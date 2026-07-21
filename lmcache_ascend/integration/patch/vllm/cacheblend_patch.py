#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
1. Patch vLLM-Ascend worker for LMCache model tracking.

This script:
  - locates vllm_ascend.worker.worker via import
  - applies the LMCache model registration + KV transfer init changes to load_model
  - comments out ensure_kv_transfer_initialized in initialize_from_config
  - creates a backup of the original file

2. Patch vLLM-Ascend for Rotary Embedding
Redirecting logic to _npu_rotary_embedding as per the 0.9.2rc1 stable implementation.
This ensures we avoid the issues identified in the newer version.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path

# First Party
from lmcache_ascend.integration.patch.base_patcher import (
    BasePatcher,
    VersionRange,
    logger,
)


class CacheBlendPatcher(BasePatcher):
    VERSION_SERIES = [VersionRange("0.9.2rc1", "0.11.0")]

    ROPE_PATCH_VERSIONS = [VersionRange("0.10.2rc1", "0.11.0")]

    @classmethod
    def apply_all(cls) -> bool:
        """Main entry point: apply all vLLM-Ascend specific patches."""
        try:
            version = cls.get_version("vllm-ascend")
            logger.info(
                f"vLLM-Ascend environment confirmed (version: {version}). "
                "Applying patches..."
            )

            tasks = [
                {
                    "name": "Worker Model Tracking Patch",
                    "module": "vllm_ascend.worker.worker",
                    "func": cls._patch_worker_file,
                    "required_versions": cls.VERSION_SERIES,
                },
                {
                    "name": "RoPE Fallback Patch",
                    "module": "vllm_ascend.ops.rotary_embedding",
                    "func": cls._patch_rope_file,
                    "required_versions": cls.ROPE_PATCH_VERSIONS,
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
        Injects LMCache tracking into the vLLM-Ascend worker.

        This patch registers the model tracker during initialization and
        realigns the KV cache connector setup to ensure metadata is
        available for CacheBlend queries.

        --- a/vllm_ascend/worker/worker.py
        +++ b/vllm_ascend/worker/worker.py
        @@ -17,6 +17,8 @@
        +from lmcache.integration.vllm.utils import ENGINE_NAME
        +from lmcache.v1.compute.models.utils import VLLMModelTracker

        @@ -312,6 +314,9 @@ class NPUWorker(WorkerBase):
            def load_model():
                ...
                with context:
                    self.model_runner.load_model()

        +        VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)
        +        ensure_kv_transfer_initialized(self.vllm_config)
        +
        @@ -391,7 +396,7 @@ class NPUWorker(WorkerBase):
            def initialize_from_config():
                ...
                ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        -        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        +        # ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        """
        _IMPORTS_TO_ADD = [
            "from lmcache.integration.vllm.utils import ENGINE_NAME\n",
            "from lmcache.v1.compute.models.utils import VLLMModelTracker\n",
        ]
        _EARLY_KV_INIT_CALL = (
            "ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)"
        )

        content = path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        changed = False

        # Phase 1: Injection of Imports
        if not any("VLLMModelTracker" in line for line in lines):
            insert_at = 0
            for i, line in enumerate(lines):
                if "import" in line:
                    insert_at = i
                    break
            lines = lines[:insert_at] + _IMPORTS_TO_ADD + lines[insert_at:]
            changed = True
            logger.debug("Injected LMCache imports.")

        # Phase 2: Comment out early KV initialization in initialize_from_config
        block_init = cls._find_function_block(lines, "initialize_from_config")
        if not block_init:
            raise RuntimeError(
                "Critical: Could not find 'initialize_from_config' in worker."
            )

        early_init_found = False
        commented_early_init_found = False
        for i in range(block_init[0], block_init[1]):
            line = lines[i]
            line_stripped = line.lstrip()
            if _EARLY_KV_INIT_CALL in line and not line_stripped.startswith("#"):
                lines[i] = line.replace(
                    _EARLY_KV_INIT_CALL,
                    f"# {_EARLY_KV_INIT_CALL}",
                    1,
                )
                changed = True
                early_init_found = True
                logger.debug(
                    "Commented out ensure_kv_transfer_initialized at %d", i + 1
                )
                break
            if f"# {_EARLY_KV_INIT_CALL}" in line:
                commented_early_init_found = True

        if not early_init_found and not commented_early_init_found:
            raise RuntimeError(
                "Critical: Could not find the exact early "
                "'ensure_kv_transfer_initialized(self.vllm_config, "
                "kv_cache_config)' call inside initialize_from_config."
            )

        # Phase 3: Add model registration in load_model
        if not any("VLLMModelTracker.register_model" in line for line in lines):
            block_load = cls._find_function_block(lines, "load_model")
            if not block_load:
                raise RuntimeError("Critical: Could not find 'load_model' in worker.")

            last_idx = block_load[1] - 1
            while last_idx > block_load[0] and not lines[last_idx].strip():
                last_idx -= 1

            # Logic: Find indentation of the function body
            indent = "        "  # Default for vLLM class methods
            reg_msg = (
                "VLLMModelTracker.register_model(ENGINE_NAME, "
                "self.model_runner.model)"
            )
            snippet = [
                "\n",
                f"{indent}{reg_msg}\n",
                f"{indent}ensure_kv_transfer_initialized(self.vllm_config)\n",
            ]
            for i, s in enumerate(snippet):
                lines.insert(last_idx + 1 + i, s)
            changed = True
            logger.debug(f"Injected VLLMModelTracker at line {last_idx + 2}")

        if changed:
            cls._backup_file(path)
            path.write_text("".join(lines), encoding="utf-8")
            logger.info(f"Successfully applied worker patches to {path}")
        else:
            logger.info(f"Worker file {path} is already patched.")

    @classmethod
    def _patch_rope_file(cls, path: Path):
        """
        Forces RoPE fallback to _npu_rotary_embedding by setting self.cos to None.

        Precomputing cos/sin can lead to dimension mismatches with query (q)
        tensors. Setting this to None ensures the model retrieves data from
        cos_sin_cache dynamically instead.

        --- a/vllm_ascend/ops/rotary_embedding.py
        +++ b/vllm_ascend/ops/rotary_embedding.py
        @@ -42,6 +42,7 @@ def _rope_forward_oot(
             is_neox_style: bool,
             offsets: Optional[torch.Tensor] = None
         ) -> Tuple[torch.Tensor, torch.Tensor]:
        +     self.cos = None  # Force fallback - Added by LMCache
             query_shape, key_shape = query.shape, key.shape
             if self.cos_sin_cache.device != query.device:
                 self.cos_sin_cache = self.cos_sin_cache.to(query.device)
        """
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        func_name = "_rope_forward_oot"

        block = cls._find_function_block(lines, func_name)
        if not block:
            logger.error(f"Critical: Function {func_name} not found in {path}.")
            return

        if any("self.cos = None" in lines[i] for i in range(block[0], block[1])):
            logger.info(f"RoPE file {path} already contains the fallback patch.")
            return

        # Locate insertion point after function arguments
        insert_pos = block[0]
        found_closing_paren = False
        while insert_pos < len(lines):
            if ")" in lines[insert_pos]:
                found_closing_paren = True
                break
            insert_pos += 1

        if not found_closing_paren:
            logger.error(f"Could not find signature end for {func_name}")
            return

        insert_pos += 1  # Move to the first line of the function body

        # Calculate indentation based on the next non-empty line
        indent = "    "
        if insert_pos < len(lines):
            for i in range(insert_pos, block[1]):
                stripped = lines[i].lstrip()
                if stripped and not stripped.startswith(('"', "'")):
                    indent = lines[i][: len(lines[i]) - len(stripped)]
                    break

        patch_text = f"{indent}self.cos = None  # Force fallback - Added by LMCache\n"
        lines.insert(insert_pos, patch_text)
        cls._backup_file(path)
        path.write_text("".join(lines), encoding="utf-8")
        logger.info(f"Successfully applied RoPE fallback patch to {path}")
