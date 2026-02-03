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

    if is_installed("mindspore"):
        logger.info("MindSpore environment confirmed. Applying patches...")
        # TODO: apply_mindspore_patches()
        return

    if is_installed("sglang"):
        # TODO: apply_sglang_patches()
        return

    if is_installed("vllm_ascend"):
        logger.info("vLLM-Ascend environment confirmed. Applying patches...")

        from lmcache_ascend.integration.patch.vllm.cacheblend_patch import (
            CacheBlendPatcher,
        )
        success = CacheBlendPatcher.apply_all()
        if success:
            logger.info("vLLM-Ascend patches applied successfully.")
        else:
            logger.error("vLLM-Ascend patches failed to apply.")
        return

    logger.info(
        "No supported inference framework (MindSpore, SGLang, or vLLM-Ascend) "
        "found in current environment."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_integration_patches()
