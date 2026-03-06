#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Patch vLLM-Ascend worker for 310P platform.

This script modifies the KV cache allocation logic to ensure 310P
always uses the default initialization path(NZ).
"""

# Standard
from pathlib import Path

# First Party
from lmcache_ascend.integration.patch.base_patcher import (
    BasePatcher,
    VersionRange,
    logger,
)


class VllmAscend0100rc1Patcher(BasePatcher):
    """Patcher for 310P in vllm-ascend v0.10.0rc1."""

    # Version list where 310P KV cache patch is required
    VERSION_SERIES = [VersionRange("0.10.0rc1")]

    @classmethod
    def apply_all(cls) -> bool:
        try:
            version = cls.get_version("vllm-ascend")
            logger.info(
                f"vLLM-Ascend environment confirmed (version: {version}). "
                "Applying patches..."
            )

            tasks = [
                {
                    "name": "Vllm-Ascend v0.10.0rc1 310p Adapt Patch",
                    "module": "vllm_ascend.worker.model_runner_v1",
                    "func": cls._patch_worker_file,
                    "required_versions": cls.VERSION_SERIES,
                },
            ]

            return cls.run_patch_tasks(version, tasks)
        except Exception as e:
            logger.error(
                f"Unexpected error during patching process: {e}", exc_info=True
            )
            return False

    @classmethod
    def _patch_worker_file(cls, path: Path):
        """
        This patch adds is_310p() check to KV transfer config conditions,
        ensuring proper cache behavior on Ascend 310P devices.

        --- a/vllm_ascend/worker/model_runner_v1.py
        +++ b/vllm_ascend/worker/model_runner_v1.py
        @@ -2257,7 +2257,7 @@ class NPUModelRunner(LoRAModelRunnerMixin):
                            num_kv_heads, nope_dim)
            rope_cache_shape = (num_blocks, block_size,
                                num_kv_heads, rope_dim)
        -       if self.vllm_config.kv_transfer_config is None:
        +       if is_310p() or self.vllm_config.kv_transfer_config is None:
        @@ -2302,7 +2302,7 @@ class NPUModelRunner(LoRAModelRunnerMixin):
            kv_cache_list = []
            for i in range(num_caches):
                cache_shape = kv_cache_shape[1:]
        -             if self.vllm_config.kv_transfer_config is None:
        +              if is_310p() or self.vllm_config.kv_transfer_config is None:
        """
        _TARGET_PATTERN = "if self.vllm_config.kv_transfer_config is None:"
        _REPLACEMENT = "if is_310p() or self.vllm_config.kv_transfer_config is None:"
        _EXPECTED_OCCURRENCES = 2

        content = path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)

        modification_count = 0

        for i, line in enumerate(lines):
            if _TARGET_PATTERN in line and "is_310p()" not in line:
                lines[i] = line.replace(_TARGET_PATTERN, _REPLACEMENT)
                modification_count += 1
                logger.debug(
                    f"Modified KV cache condition at line {i + 1} "
                    f"(occurrence #{modification_count})"
                )

        if modification_count == 0:
            logger.info(f"310P KV cache patch already applied in {path}")
            return

        if modification_count == _EXPECTED_OCCURRENCES:
            logger.debug(
                f"Successfully modified all {_EXPECTED_OCCURRENCES} KV cache conditions"
            )
        else:
            logger.warning(
                f"Modified {modification_count} occurrence(s), "
                f"expected {_EXPECTED_OCCURRENCES}"
            )

        cls._backup_file(path)
        path.write_text("".join(lines), encoding="utf-8")
