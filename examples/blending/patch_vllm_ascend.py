#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
1. Patch vLLM-Ascend worker for LMCache model tracking.

This script:
  - locates vllm_ascend.worker.worker_v1 via import
  - applies the LMCache model registration + KV transfer init changes to load_model
  - comments out ensure_kv_transfer_initialized in _init_worker_distributed_environment
  - creates a backup of the original file

2. Patch vLLM-Ascend for Rotary Embedding
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import importlib
import shutil
import sys
import time

_IMPORTS_TO_ADD = [
    "from lmcache.integration.vllm.utils import ENGINE_NAME\n",
    "from lmcache.v1.compute.models.utils import VLLMModelTracker\n",
]

TARGET_MODULE = "vllm_ascend.ops.rotary_embedding"
TARGET_FUNC = "_rope_forward_oot"
REQUIRED_VERSIONS = [
    "0.10.2rc1",
    "0.11.0rc0",
    "0.11.0rc1",
    "0.11.0rc2",
    "0.11.0rc3",
    "0.11.0",
]


def get_vllm_ascend_version():
    try:
        return importlib.metadata.version("vllm-ascend")
    except importlib.metadata.PackageNotFoundError:
        return None


def _find_module_path(module_name: str) -> Path:
    """Find the file system path of a python module."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        # Try fallback if the module is named differently in the environment
        if module_name == "vllm_ascend.worker.worker_v1":
            # Some environments might use underscores/hyphens differently
            module = importlib.import_module("vllm_ascend.worker.worker_v1")

    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"Unable to resolve file path for {module_name}")
    return Path(module_file).resolve()


def _backup_file(path: Path) -> Path:
    """Create a backup of the target file."""
    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists():
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, backup)
    return backup


def _find_function_block(lines: list[str], func_name: str) -> tuple[int, int] | None:
    """Find the start and end line indices of a function definition."""
    start = None
    indent = 0
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"def {func_name}("):
            start = idx
            indent = len(line) - len(stripped)
            break
    if start is None:
        return None

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].lstrip()
        if not stripped:
            continue
        # If we hit another definition at same or lower indent, the block ends
        current_indent = len(lines[idx]) - len(stripped)
        if (stripped.startswith(("def ", "class ", "@"))) and current_indent <= indent:
            end = idx
            break
    return start, end


def _ensure_imports(lines: list[str]) -> tuple[list[str], bool]:
    """Ensure required LMCache imports exist in the file."""
    content = "".join(lines)
    if all(line in content for line in _IMPORTS_TO_ADD):
        return lines, False

    insert_at = 0
    for idx, line in enumerate(lines):
        if "import" in line:
            insert_at = idx
            break

    new_lines = lines[:insert_at] + _IMPORTS_TO_ADD + lines[insert_at:]
    return new_lines, True


def _comment_kv_init_in_worker_env(lines: list[str]) -> tuple[list[str], bool]:
    """Comment out ensure_kv_transfer_initialized in the distributed env setup."""
    block = _find_function_block(lines, "_init_worker_distributed_environment")
    if block is None:
        return lines, False

    start, end = block
    changed = False
    for idx in range(start, end):
        line = lines[idx]
        stripped = line.lstrip()
        # Look for the specific call mentioned in README
        if "ensure_kv_transfer_initialized(" in stripped and not stripped.startswith(
            "#"
        ):
            leading = line[: len(line) - len(stripped)]
            lines[idx] = f"{leading}# {stripped}"
            changed = True
            # In distributed env, there is usually only one such call
            if block:
                break

    return lines, changed


def _patch_load_model(lines: list[str]) -> tuple[list[str], bool]:
    """Add model registration and KV init at the end of load_model function."""
    block = _find_function_block(lines, "load_model")
    if block is None:
        raise RuntimeError("Unable to find load_model in worker_v1.py")

    start, end = block

    # Check if already patched
    if any("VLLMModelTracker.register_model" in line for line in lines[start:end]):
        return lines, False

    # Find the last line of the function to append
    # We look for the last non-empty line within the block
    last_content_idx = end - 1
    while last_content_idx > start and not lines[last_content_idx].strip():
        last_content_idx -= 1

    # Determine indentation from the function body
    # Typically 8 spaces (4 for class, 4 for def)
    indent = "        "

    registration_snippet = [
        "\n",
        f"{indent}VLLMModelTracker.register_model("
        f"ENGINE_NAME, self.model_runner.model)\n",
        f"{indent}ensure_kv_transfer_initialized(self.vllm_config)\n",
    ]

    # Insert before the start of the next block
    for i, line in enumerate(registration_snippet):
        lines.insert(last_content_idx + 1 + i, line)

    return lines, True


def patch_ascend_worker(path: Path) -> bool:
    """Apply all necessary patches to the worker file."""
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    changed = False

    # 1. Add Imports
    lines, imports_changed = _ensure_imports(lines)
    changed = changed or imports_changed

    # 2. Comment out the distributed init call
    lines, env_changed = _comment_kv_init_in_worker_env(lines)
    changed = changed or env_changed

    # 3. Patch load_model function
    lines, init_changed = _patch_load_model(lines)
    changed = changed or init_changed

    if not changed:
        return False

    _backup_file(path)
    path.write_text("".join(lines), encoding="utf-8")
    return True


def patch_rope_forward(path: Path) -> bool:
    """
    Applies the 'self.cos = None' patch to _rope_forward_oot.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    start_idx = -1
    indent = ""

    for i, line in enumerate(lines):
        if f"def {TARGET_FUNC}(" in line:
            start_idx = i
            for j in range(i + 1, len(lines)):
                stripped = lines[j].lstrip()
                if stripped and not stripped.startswith('"""'):
                    indent = lines[j][: len(lines[j]) - len(stripped)]
                    break
            break

    if start_idx == -1:
        print(f"Error: Could not find function {TARGET_FUNC}")
        return False

    already_patched = False
    for i in range(start_idx + 1, start_idx + 10):
        if i < len(lines) and "self.cos = None" in lines[i]:
            already_patched = True
            break

    if already_patched:
        return False

    insert_pos = start_idx
    while ")" not in lines[insert_pos]:
        insert_pos += 1
    insert_pos += 1

    patch_line = f"{indent}self.cos = None  # Added by Patch to force fallback\n"
    lines.insert(insert_pos, patch_line)

    _backup_file(path)
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default="vllm_ascend.worker.worker_v1",
        help="Module path to patch (default: vllm_ascend.worker.worker_v1)",
    )
    parser.add_argument(
        "--rope-module",
        default="vllm_ascend.ops.rotary_embedding",
        help="Rotary embedding module path to patch",
    )
    args = parser.parse_args()

    # --- 1. Patch Worker ---
    try:
        target_path = _find_module_path(args.module)
    except Exception as exc:
        print(f"Error locating module {args.module}: {exc}", file=sys.stderr)
        return 1

    try:
        did_patch = patch_ascend_worker(target_path)

        if did_patch:
            print(f"Successfully patched ascend_worker: {target_path}")
        else:
            print(f"Ascend_worker already patched or no changes needed: {target_path}")
    except Exception as exc:
        print(f"Error patching {target_path}: {exc}", file=sys.stderr)
        # Standard
        import traceback

        traceback.print_exc()
        return 1

    # --- 2. Patch Rotary Embedding ---
    try:
        current_version = get_vllm_ascend_version()
        if current_version not in REQUIRED_VERSIONS:
            print(
                f"Skipping RoPE patch: vllm-ascend version {current_version} "
                "is not in the required list."
            )
        else:
            rope_path = _find_module_path(args.rope_module)
            did_patch_rope = patch_rope_forward(rope_path)

            if did_patch_rope:
                print(f"Successfully patched rope_forward: {rope_path}")
            else:
                print(f"Rope forward already patched or no changes needed: {rope_path}")

    except Exception as exc:
        print(f"Error patching rope module: {exc}", file=sys.stderr)
        return 1

    print("\nAll patch operations completed.")
    print("Please restart vLLM-Ascend workers to apply changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
