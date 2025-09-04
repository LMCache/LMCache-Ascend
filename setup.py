# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import os
import sys

# Third Party
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

import logging
import sysconfig
import subprocess
import platform
import shutil


ROOT_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_ascend_home_path():
    # NOTE: standard Ascend CANN toolkit path
    return os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")


def _get_ascend_env_path():
    # NOTE: standard Ascend Environment variable setup path
    env_script_path = os.path.realpath(
        os.path.join(_get_ascend_home_path(), "..", "set_env.sh")
    )
    if not os.path.exists(env_script_path):
        raise ValueError(
            f"The file '{env_script_path}' is not found, "
            "please make sure environment variable 'ASCEND_HOME_PATH' is set correctly."
        )
    return env_script_path


def _get_npu_soc():
    _soc_version = os.getenv("SOC_VERSION", None)
    if _soc_version is None:
        npu_smi_cmd = [
            "bash",
            "-c",
            "npu-smi info | grep OK | awk '{print $3}' | head -n 1",
        ]
        try:
            _soc_version = subprocess.check_output(npu_smi_cmd, text=True).strip()
            _soc_version = _soc_version.split("-")[0]
            _soc_version = "Ascend" + _soc_version
            return _soc_version
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Retrieve SoC version failed: {e}")
    return _soc_version

class custom_build_info(build_py):

    def run(self):
        soc_version = _get_npu_soc()
        if not soc_version:
            raise ValueError(
                "SOC version is not set. Please set SOC_VERSION environment variable."
            )

        package_dir = os.path.join(ROOT_DIR, "lmcache_ascend", "_build_info.py")
        with open(package_dir, "w+") as f:
            f.write('# Auto-generated file\n')
            f.write(f"__soc_version__ = '{soc_version}'\n")
        logging.info(
            f"Generated _build_info.py with SOC version: {soc_version}")
        super().run()

class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwargs) -> None:
        super().__init__(name, sources=[], py_limited_api=False, **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class custom_install(install):
    def run(self):
        self.run_command("build_ext")
        install.run(self)


class CustomAscendCmakeBuildExt(build_ext):
    def build_extension(self, ext):
        # build the so as c_ops
        ext_name = ext.name.split(".")[-1]
        so_name = ext_name + ".so"
        logger.info(f"Building {so_name} ...")
        BUILD_OPS_DIR = os.path.join(ROOT_DIR, "build")
        os.makedirs(BUILD_OPS_DIR, exist_ok=True)

        ascend_home_path = _get_ascend_home_path()
        env_path = _get_ascend_env_path()
        _soc_version = _get_npu_soc()
        _cxx_compiler = os.getenv("CXX")
        _cc_compiler = os.getenv("CC")
        python_executable = sys.executable

        try:
            # if pybind11 is installed via pip
            pybind11_cmake_path = (
                subprocess.check_output(
                    [python_executable, "-m", "pybind11", "--cmakedir"]
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError as e:
            # else specify pybind11 path installed from source code on CI container
            raise RuntimeError(f"CMake configuration failed: {e}")

        import torch_npu

        torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        import torch

        torch_path = os.path.dirname(os.path.abspath(torch.__file__))

        # python include
        python_include_path = sysconfig.get_path("include", scheme="posix_prefix")

        arch = platform.machine()
        install_path = os.path.join(BUILD_OPS_DIR, "install")
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            install_path = BUILD_OPS_DIR

        cmake_cmd = [
            f"source {env_path} && "
            f"cmake -S {ROOT_DIR} -B {BUILD_OPS_DIR}"
            f"  -DSOC_VERSION={_soc_version}"
            f"  -DARCH={arch}"
            "  -DUSE_ASCEND=1"
            f"  -DPYTHON_EXECUTABLE={python_executable}"
            f"  -DCMAKE_PREFIX_PATH={pybind11_cmake_path}"
            f"  -DCMAKE_BUILD_TYPE=Release"
            f"  -DCMAKE_INSTALL_PREFIX={install_path}"
            f"  -DPYTHON_INCLUDE_PATH={python_include_path}"
            f"  -DTORCH_NPU_PATH={torch_npu_path}"
            f"  -DTORCH_PATH={torch_path}"
            f"  -DASCEND_CANN_PACKAGE_PATH={ascend_home_path}"
            "  -DCMAKE_VERBOSE_MAKEFILE=ON"
        ]

        if _cxx_compiler is not None:
            cmake_cmd += [f"  -DCMAKE_CXX_COMPILER={_cxx_compiler}"]

        if _cc_compiler is not None:
            cmake_cmd += [f"  -DCMAKE_C_COMPILER={_cc_compiler}"]

        cmake_cmd += [f" && cmake --build {BUILD_OPS_DIR} -j --verbose"]
        cmake_cmd += [f" && cmake --install {BUILD_OPS_DIR}"]
        cmake_cmd = "".join(cmake_cmd)

        logger.info(f"Start running CMake commands:\n{cmake_cmd}")
        try:
            _ = subprocess.run(
                cmake_cmd, cwd=ROOT_DIR, text=True, shell=True, check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build {so_name}: {e}")

        build_lib_dir = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(build_lib_dir), exist_ok=True)

        package_name = ext.name.split(".")[0]  # e.g., 'lmcache'
        src_dir = os.path.join(ROOT_DIR, package_name)

        for root, _, files in os.walk(install_path):
            for file in files:
                if file.endswith(".so"):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(os.path.dirname(build_lib_dir), file)
                    if os.path.exists(dst_path):
                        os.remove(dst_path)

                    if isinstance(
                        self.distribution.get_command_obj("develop"), develop
                    ):
                        # For the ascend kernels
                        src_dir_file = os.path.join(src_dir, file)
                        shutil.copy(src_path, src_dir_file)
                    shutil.copy(src_path, dst_path)

                    logger.info(f"Copied {file} to {dst_path}")


def ascend_extension():
    print("Building Ascend extensions")
    return [CMakeExtension(name="lmcache_ascend.c_ops")], {
        "build_py": custom_build_info,
        "build_ext": CustomAscendCmakeBuildExt
    }


if __name__ == "__main__":
    ext_modules, cmdclass = ascend_extension()

    setup(
        packages=find_packages(
            exclude=("csrc",)
        ),  # Ensure csrc is excluded if it only contains sources
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
    )
