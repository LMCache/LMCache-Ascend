#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

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


class SglangPatcher(BasePatcher):
    SGLANG_VERSIONS = [VersionRange("0.5.2", "0.5.8")]

    @classmethod
    def apply_all(cls) -> bool:
        """Main entry point: apply all SGLang specific patches."""
        try:
            version = cls.get_version("sglang")

            logger.info(
                f"SGLang environment confirmed (version: {version}). "
                "Applying patches..."
            )

            # Resolve the file path dynamically based on installed sglang package
            model_runner_path = cls._find_sglang_file()
            if not model_runner_path:
                logger.error("Could not locate sglang model_runner or mixin file.")
                return False

            tasks = [
                {
                    "name": "SGLang Memory Pool Patch",
                    "module": model_runner_path,
                    "func": cls._patch_init_memory_pool,
                    "required_versions": cls.SGLANG_VERSIONS,
                }
            ]

            return cls.run_patch_tasks(version, tasks)

        except Exception as e:
            logger.error(f"Unexpected error during SGLang patching: {e}")
            return False

    @classmethod
    def _find_sglang_file(cls) -> Path | None:
        """Find the relevant file across different SGLang versions."""
        candidates = [
            "sglang.srt.model_executor.model_runner_kv_cache_mixin",
            "sglang.srt.model_executor.model_runner",
        ]
        for mod_name in candidates:
            try:
                if cls._find_module_path(mod_name):
                    return mod_name
            except Exception:
                continue
        return None

    @classmethod
    def _patch_init_memory_pool(cls, path: Path) -> bool:
        """
        Inject lmcache_ascend import into SGLang's init_memory_pool.

        --- a/SG/sglang/python/sglang/srt/model_executor/model_runner.py
        +++ b/SG/sglang/python/sglang/srt/model_executor/model_runner.py

        @@ -1635,6 +1636,8 @@ class ModelRunner:
                # Initialize token_to_kv_pool
                is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
                if self.server_args.attention_backend == "ascend":
        +            if self.server_args.enable_lmcache:
        +                import lmcache_ascend
                    if self.use_mla_backend:
                        self.token_to_kv_pool = AscendMLAPagedTokenToKVPool(
                            self.max_total_num_tokens,
        """
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        changed = False

        block = cls._find_function_block(lines, "init_memory_pool")
        if not block:
            logger.error(f"Function 'init_memory_pool' not found in {path}")
            return False

        target_trigger = 'if self.server_args.attention_backend == "ascend":'

        for i in range(block[0], block[1]):
            if target_trigger in lines[i]:
                # Avoid double patching
                if i + 1 < len(lines) and "enable_lmcache" in lines[i + 1]:
                    logger.info(f"File {path} is already patched.")
                    return True

                line_content = lines[i]
                current_indent = line_content[
                    : len(line_content) - len(line_content.lstrip())
                ]
                inner_indent = current_indent + "    "

                patch_code = [
                    f"{inner_indent}if self.server_args.enable_lmcache:\n",
                    f"{inner_indent}    import lmcache_ascend\n",
                ]

                lines = lines[: i + 1] + patch_code + lines[i + 1 :]
                changed = True
                break

        if changed:
            cls._backup_file(path)
            path.write_text("".join(lines), encoding="utf-8")
            logger.info(f"Applied patch to {path}")
            return True
        return False
