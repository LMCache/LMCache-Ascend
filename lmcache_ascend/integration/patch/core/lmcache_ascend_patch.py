#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Patch `lmcache` to import `lmcache_ascend`.

This script:
  - locates `lmcache.__init__` via import
  - adds `import lmcache_ascend` to it to trigger Ascend-specific patches
  - creates a backup of the original file
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path

# First Party
from lmcache_ascend.integration.patch.base_patcher import (
    BasePatcher,
    logger,
)


class LMCacheAscendPatcher(BasePatcher):
    @classmethod
    def apply_all(cls) -> bool:
        try:
            logger.info("Applying LMCacheAscendPatcher patches...")

            tasks = [
                {
                    "name": "LMCache-Ascend core Patch",
                    "module": "lmcache.__init__",
                    "func": cls._patch_init_import,
                    "required_versions": None,
                }
            ]

            return cls.run_patch_tasks(None, tasks)
        except Exception as e:
            logger.error(
                f"Unexpected error during patching process: {e}", exc_info=True
            )
            return False

    @classmethod
    def _patch_init_import(cls, path: Path):
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)

        if any("import lmcache_ascend" in line for line in lines):
            return

        lines = lines + ["import lmcache_ascend\n"]
        logger.debug("Apply LMCacheAscendPatcher done.")

        cls._backup_file(path)
        path.write_text("".join(lines), encoding="utf-8")
        logger.info(f"Successfully applied worker patches to {path}")
