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
import configparser 
import glob

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

def _get_aicore_arch_number(ascend_path, soc_version, host_arch):
    platform_config_path = os.path.join(ascend_path, f"{host_arch}-linux/data/platform_config")
    ini_file = os.path.join(platform_config_path, f"{soc_version}.ini")

    if not os.path.exists(ini_file):
        raise ValueError(f"The file '{ini_file}' is not found, please check your SOC_VERSION")

    # read the file and extract
    logger.info(f"Extracting AIC Version from: {ini_file}")
    cp = configparser.ConfigParser()
    cp.read(ini_file)

    aic_version = cp.get("version", "AIC_version")
    logger.info(f"AIC Version: {aic_version}")

    version_number = aic_version.split("-")[-1]
    return version_number

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
        arch = platform.machine()
        _aicore_arch = _get_aicore_arch_number(ascend_home_path, _soc_version, arch)
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

        install_path = os.path.join(BUILD_OPS_DIR, "install")
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            install_path = BUILD_OPS_DIR

        cmake_cmd = [
            f"source {env_path} && "
            f"cmake -S {ROOT_DIR} -B {BUILD_OPS_DIR}"
            f"  -DSOC_VERSION={_soc_version}"
            f"  -DASCEND_AICORE_ARCH={_aicore_arch}"
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

        # Expected file patterns (using glob patterns for flexibility)
        expected_patterns = [
            "c_ops*.so", 
            "libcache_kernels.so"
        ]

        # Search for files matching our patterns
        so_files = []
        for pattern in expected_patterns:
            # Search in main directory and common subdirectories
            search_paths = [
                install_path,
                os.path.join(install_path, "lib"),
                os.path.join(install_path, "lib64")
            ]
            
            for search_path in search_paths:
                if os.path.exists(search_path):
                    matches = glob.glob(os.path.join(search_path, pattern))
                    so_files.extend(matches)

        # For develop mode, also copy to source directory
        is_develop_mode = isinstance(
            self.distribution.get_command_obj("develop"), develop
        )
        # Remove duplicates
        so_files = list(dict.fromkeys(so_files))

        if not so_files:
            raise RuntimeError(f"No .so files found matching patterns {expected_patterns}")

        logger.info(f"Found {len(so_files)} .so files:")
        for so_file in so_files:
            logger.info(f"  - {so_file}")

        # Copy each file (same copying logic as above)
        for src_path in so_files:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(os.path.dirname(build_lib_dir), filename)
            
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {filename} to {dst_path}")
            
            if is_develop_mode:
                src_dir_file = os.path.join(src_dir, filename)
                if os.path.abspath(src_path) != os.path.abspath(src_dir_file):
                    if os.path.exists(src_dir_file):
                        os.remove(src_dir_file)
                    shutil.copy2(src_path, src_dir_file)
                    logger.info(f"Copied {filename} to source directory: {src_dir_file}")

        logger.info("All files copied successfully")



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
