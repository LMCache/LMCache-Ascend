# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import configparser
import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig

# Third Party
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

ROOT_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_MINDSPORE = os.getenv("USE_MINDSPORE", "False").lower() in ("true", "1")


# NOTE: Apply platform-specific patches during installation
# to resolve environment issues
def run_patches():
    """Execute the patch script after installation."""
    try:
        sys.path.append(str(ROOT_DIR))
        # First Party
        from lmcache_ascend.integration.patch.apply_patch import run_integration_patches

        run_integration_patches()
    except Exception as e:
        logger.error(f"Post-install patch system encountered an error: {e}")
        return


def _get_ascend_home_path():
    # NOTE: standard Ascend CANN toolkit path
    return os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")


def _get_cann_version():
    """Read the CANN toolkit version from ascend_toolkit_install.info."""
    ascend_home = _get_ascend_home_path()
    arch = platform.machine()
    info_path = os.path.join(
        ascend_home, f"{arch}-linux", "ascend_toolkit_install.info"
    )
    if not os.path.exists(info_path):
        logger.warning(f"ascend_toolkit_install.info not found at {info_path}")
        return None
    with open(info_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("version="):
                version = line.split("=", 1)[1].strip()
                logger.info(f"Detected CANN toolkit version: {version}")
                return version
    logger.warning("Could not parse version from ascend_toolkit_install.info")
    return None


def _is_cann_85_or_later(cann_version_tuple):
    """Determine whether the CANN environment is 8.5+.

    Checks (in order):
      1. Env var override: USE_HIXL=1 forces 8.5+ mode.
      2. Parsed version tuple from ascend_toolkit_install.info.
      3. Secondary heuristic: set_env.sh lives directly under
         ASCEND_HOME_PATH on CANN 8.5+ vs one level up on older versions.
    """
    env_override = os.getenv("USE_HIXL", "").strip()
    if env_override.lower() in ("1", "true", "on"):
        logger.info("USE_HIXL env var set — forcing CANN >= 8.5 build mode")
        return True
    if env_override.lower() in ("0", "false", "off"):
        logger.info("USE_HIXL env var explicitly disabled — forcing CANN < 8.5 build mode")
        return False

    if cann_version_tuple:
        return cann_version_tuple >= (8, 5, 0)

    ascend_home = _get_ascend_home_path()
    new_path = os.path.join(ascend_home, "set_env.sh")
    old_path = os.path.join(ascend_home, "..", "set_env.sh")
    if os.path.exists(new_path) and not os.path.exists(old_path):
        logger.warning(
            "CANN version detection failed but set_env.sh location "
            "suggests CANN >= 8.5 — building with HIXL/hcomm_onesided. "
            "Set USE_HIXL=0 to override."
        )
        return True

    logger.warning(
        "CANN version detection failed — defaulting to HCCL (pre-8.5) build. "
        "Set USE_HIXL=1 to force CANN >= 8.5 mode."
    )
    return False


def _get_ascend_env_path(cann_85_or_later):
    _ascend_home_path = _get_ascend_home_path()
    if cann_85_or_later:
        env_script_path = os.path.join(_ascend_home_path, "set_env.sh")
    else:
        env_script_path = os.path.join(_ascend_home_path, "..", "set_env.sh")

    if not os.path.exists(env_script_path):
        raise ValueError(
            f"The file '{env_script_path}' is not found, "
            "please make sure environment variable 'ASCEND_HOME_PATH' is set correctly."
        )
    return env_script_path


def _get_npu_soc():
    """
    Retrieves the NPU SoC version by parsing the output of the npu-smi command.

    This function handles two known output formats:
    1. Standard format with "Chip Name" (e.g., Ascend910B4).
    2. A newer format with both "Chip Name" and "NPU Name" (e.g., Ascend910_9392).

    Returns:
        str: The determined SoC version string.

    Raises:
        RuntimeError: If the npu-smi command fails or the output is malformed.
    """
    _soc_version = os.getenv("SOC_VERSION")
    if _soc_version:
        return (
            "Ascend" + _soc_version[6:]
            if _soc_version.lower().startswith("ascend")
            else _soc_version
        )

    try:
        npu_smi_cmd = ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"]
        full_output = subprocess.check_output(npu_smi_cmd, text=True)

        npu_info = {}
        for line in full_output.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                npu_info[key.strip()] = value.strip()

        chip_name = npu_info.get("Chip Name", None)
        npu_name = npu_info.get("NPU Name", None)

        if not chip_name:
            raise RuntimeError("Could not find 'Chip Name' in npu-smi output.")

        if npu_name:
            # New Format for npu-smi info on 910C machines: "Ascend910_9392"
            _soc_version = f"{chip_name}_{npu_name}"
        else:
            # Old Format for npu-smi info on 910B machines: "Ascend910B4"
            if chip_name.startswith("Ascend"):
                _soc_version = chip_name
            else:
                _soc_version = "Ascend" + chip_name

        return _soc_version

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            f"Failed to execute npu-smi command and retrieve SoC version: {e}"
        ) from e


def _get_aicore_arch_number(ascend_path, soc_version, host_arch):
    platform_config_path = os.path.join(
        ascend_path, f"{host_arch}-linux/data/platform_config"
    )
    ini_file = os.path.join(platform_config_path, f"{soc_version}.ini")

    if not os.path.exists(ini_file):
        raise ValueError(
            f"The file '{ini_file}' is not found, please check your SOC_VERSION"
        )

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

        cann_version = _get_cann_version() or "unknown"

        package_dir = os.path.join(ROOT_DIR, "lmcache_ascend", "_build_info.py")
        with open(package_dir, "w+") as f:
            f.write("# Auto-generated file\n")
            f.write(f"__soc_version__ = '{soc_version}'\n")
            if USE_MINDSPORE:
                framework_name = "mindspore"
            else:
                framework_name = "pytorch"
            f.write(f"__framework_name__ = '{framework_name}'\n")
            f.write(f"__cann_version__ = '{cann_version}'\n")
            f.write("\n")
            f.write("def cann_version_tuple() -> tuple[int, ...]:\n")
            f.write("    import re\n")
            f.write("    parts = re.findall(r'\\d+', __cann_version__)\n")
            f.write("    return tuple(int(p) for p in parts)\n")
        logging.info(
            f"Generated _build_info.py with SOC version: {soc_version}, "
            f"CANN version: {cann_version}"
        )
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
        cann_version = _get_cann_version()
        cann_version_tuple = tuple(
            int(p) for p in re.findall(r"\d+", cann_version or "")
        )

        self._cann_version_no_hccl = _is_cann_85_or_later(cann_version_tuple)
        env_path = _get_ascend_env_path(self._cann_version_no_hccl)

        _soc_version = _get_npu_soc()
        arch = platform.machine()
        _aicore_arch = _get_aicore_arch_number(ascend_home_path, _soc_version, arch)
        _cxx_compiler = os.getenv("CXX")
        _cc_compiler = os.getenv("CC")
        python_executable = sys.executable

        if self._cann_version_no_hccl:
            logger.info(f"CANN {cann_version}: building HIXL transfer channel")
            logger.info(f"CANN {cann_version}: building hcomm one-sided channel")
        else:
            logger.info(f"CANN {cann_version}: building HCCL transfer channel")

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
            raise RuntimeError(f"CMake configuration failed: {e}") from e

        # python include
        python_include_path = sysconfig.get_path("include", scheme="posix_prefix")

        install_path = os.path.join(BUILD_OPS_DIR, "install")
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            install_path = BUILD_OPS_DIR

        cmake_cmd = [
            f". {env_path} && "
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
            f"  -DASCEND_CANN_PACKAGE_PATH={ascend_home_path}"
            "  -DCMAKE_VERBOSE_MAKEFILE=ON"
        ]

        if USE_MINDSPORE:
            # Third Party
            import mindspore

            ms_path = os.path.dirname(os.path.abspath(mindspore.__file__))
            cmake_cmd += [f"  -DMINDSPORE_PATH={ms_path}"]
        else:
            # Third Party
            import torch
            import torch_npu

            torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
            torch_cxx11_abi = int(torch.compiled_with_cxx11_abi())
            torch_path = os.path.dirname(os.path.abspath(torch.__file__))
            cmake_cmd += [f"  -DTORCH_NPU_PATH={torch_npu_path}"]
            cmake_cmd += [f"  -DTORCH_PATH={torch_path}"]
            cmake_cmd += [f"  -DGLIBCXX_USE_CXX11_ABI={torch_cxx11_abi}"]
            torch_cmake_dir = os.path.join(torch.utils.cmake_prefix_path, "Torch")
            cmake_cmd += [f"  -DTorch_DIR={torch_cmake_dir}"]

        if self._cann_version_no_hccl:
            cmake_cmd += ["  -DUSE_HIXL=ON"]
            cmake_cmd += ["  -DUSE_HCOMM_ONESIDED=ON"]

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
            raise RuntimeError(f"Failed to build {so_name}: {e}") from e
        build_lib_dir = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(build_lib_dir), exist_ok=True)

        src_dir = os.path.join(ROOT_DIR, "lmcache_ascend")

        # Expected file patterns (using glob patterns for flexibility)
        expected_patterns = ["c_ops*.so", "libcache_kernels.so"]
        if self._cann_version_no_hccl:
            expected_patterns.append("hixl_npu_comms*.so")
            expected_patterns.append("hcomm_onesided*.so")
        else:
            expected_patterns.append("hccl_npu_comms*.so")

        # Search for files matching our patterns
        so_files = []
        for pattern in expected_patterns:
            # Search in main directory and common subdirectories
            search_paths = [
                install_path,
                os.path.join(install_path, "lib"),
                os.path.join(install_path, "lib64"),
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
            raise RuntimeError(
                f"No .so files found matching patterns {expected_patterns}"
            )

        logger.info(f"Found {len(so_files)} .so files:")
        for so_file in so_files:
            logger.info(f"  - {so_file}")

        # Copy each file with improved path validation and duplicate handling
        #  compared to previous implementation
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
                    logger.info(
                        f"Copied {filename} to source directory: {src_dir_file}"
                    )

        logger.info("All files copied successfully")
        run_patches()


def ascend_extension():
    print("Building Ascend extensions")
    return [CMakeExtension(name="lmcache_ascend.c_ops")], {
        "build_py": custom_build_info,
        "build_ext": CustomAscendCmakeBuildExt,
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
