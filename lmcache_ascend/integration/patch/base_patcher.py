#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import importlib.metadata
import importlib.util
import logging
import shutil
import time

# Third Party
from packaging import version

logger = logging.getLogger(__name__)


class VersionRange:
    def __init__(self, start: str, end: str | None = None):
        self.start = version.parse(start.lstrip("v"))
        self.end = version.parse(end.lstrip("v")) if end else self.start

    def __contains__(self, ver_str: str) -> bool:
        if not ver_str:
            return False
        try:
            current_v = version.parse(ver_str)
            return self.start <= current_v <= self.end
        except Exception:
            return False


class BasePatcher:
    @staticmethod
    def get_version(package_name: str) -> str | None:
        """Retrieve version from installed package metadata."""
        try:
            ver = importlib.metadata.version(package_name)
            return ver.lstrip("v") if ver else None
        except importlib.metadata.PackageNotFoundError:
            logger.debug("Package %s metadata not found.", package_name)
            return None

    @staticmethod
    def is_version_in_range(
        current_ver: str, version_ranges: list[VersionRange | str]
    ) -> bool:
        """Verify is version in range"""
        if not current_ver:
            return False
        for r in version_ranges:
            if isinstance(r, VersionRange):
                if current_ver in r:
                    return True
            else:
                if current_ver == r:
                    return True
        return False

    @classmethod
    def run_patch_tasks(cls, current_ver: str | None, tasks: list[dict]):
        """
        tasks format: {
            "name": str,
            "module": str,
            "func": callable,
            "required_versions": list | None
        }
        """
        success_count = 0
        enabled_tasks = [
            t
            for t in tasks
            if t.get("required_versions") is None
            or cls.is_version_in_range(current_ver, t["required_versions"])
        ]

        for task in enabled_tasks:
            try:
                path = cls._find_module_path(task["module"])
                task["func"](path)
                logger.info("Successfully applied: %s", task["name"])
                success_count += 1
            except Exception as e:
                logger.error("Failed to apply %s: %s", task["name"], e)

        return success_count == len(enabled_tasks)

    @staticmethod
    def _find_module_path(module_name: str) -> Path:
        """Locate the physical file path of a python module."""
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"Spec origin not found for {module_name}")

            path = Path(spec.origin).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Resolved path does not exist: {path}")

            logger.debug("Located %s at %s", module_name, path)
            return path
        except Exception as e:
            raise RuntimeError(f"Locating module {module_name} failed: {e}") from e

    @staticmethod
    def _backup_file(path: Path):
        """Create a timestamped backup of the target file."""
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        try:
            shutil.copy2(path, backup)
            logger.info("Backup created: %s", backup)
        except Exception as e:
            raise RuntimeError(f"Failed to create backup for {path}: {e}") from e

    @staticmethod
    def _find_function_block(
        lines: list[str], func_name: str
    ) -> tuple[int, int] | None:
        """Find the start and end line indices of a function definition."""
        start = None
        indent = 0
        search_pattern = f"def {func_name}("

        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(search_pattern):
                start = idx
                indent = len(line) - len(stripped)
                logger.debug(
                    "Found function '%s' at line %s with indent %s",
                    func_name,
                    idx + 1,
                    indent,
                )
                break

        if start is None:
            logger.warning("Function '%s' not found in the provided lines.", func_name)
            return None

        end = len(lines)
        for idx in range(start + 1, len(lines)):
            stripped = lines[idx].lstrip()
            if not stripped:
                continue

            curr_indent = len(lines[idx]) - len(stripped)
            if stripped.startswith(("def ", "class ", "@")) and curr_indent <= indent:
                end = idx
                break

        logger.debug("Function '%s' block: lines %s to %s", func_name, start + 1, end)
        return start, end
