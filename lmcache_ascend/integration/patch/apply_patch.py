#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

os.environ["SKIP_LMCACHE_PATCH"] = "1"


def is_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def run_integration_patches():
    logger.info("Initializing LMCache-Ascend patch manager...")

    base_path = "lmcache_ascend.integration.patch"
    patch_tasks = [
        # ("sglang", f"{base_path}.sglang.sglang_patch", "SglangPatcher"),
        ("vllm_ascend", f"{base_path}.vllm.cacheblend_patch", "CacheBlendPatcher"),
        # ("mindspore", "...", "MindSporePatcher"),
    ]

    for package_name, module_path, class_name in patch_tasks:
        if is_installed(package_name):
            logger.info(f"Detected {package_name}. Applying patches...")
            try:
                module = importlib.import_module(module_path)
                patcher_cls = getattr(module, class_name)

                if patcher_cls.apply_all():
                    logger.info(f"Successfully patched {package_name}.")
                else:
                    logger.error(f"Failed to apply patches for {package_name}.")
            except Exception as e:
                logger.error(f"Error while patching {package_name}: {e}")
        else:
            logger.debug(f"{package_name} is not installed, skipping.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_integration_patches()
    del os.environ["SKIP_LMCACHE_PATCH"]
