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
import logging
import shutil
import sys
import time

# Configure local logger
logger = logging.getLogger(__name__)

class VLLMAscendPatcher:
    # Version list where RoPE patch is mandatory
    REQUIRED_ROPE_VERSIONS = [
        "0.10.2rc1", "0.11.0rc0", "0.11.0rc1", 
        "0.11.0rc2", "0.11.0rc3", "0.11.0",
    ]
    
    _IMPORTS_TO_ADD = [
        "from lmcache.integration.vllm.utils import ENGINE_NAME\n",
        "from lmcache.v1.compute.models.utils import VLLMModelTracker\n",
    ]

    @classmethod
    def apply_all(cls) -> bool:
        """Main entry point: apply all vLLM-Ascend specific patches."""
        try:
            version = cls.get_version()
            if version:
                logger.info(f"Detected vllm-ascend version: {version}")
            else:
                logger.warning("Could not detect vllm-ascend version via metadata.")

            # 1. Patch Worker for model tracking
            try:
                worker_path = cls._find_module_path("vllm_ascend.worker.worker_v1")
                cls._patch_worker_file(worker_path)
            except Exception as e:
                logger.error(f"Failed to patch worker file: {e}")
                return False

            # 2. Patch RoPE (Conditional based on version)
            if version in cls.REQUIRED_ROPE_VERSIONS:
                try:
                    rope_path = cls._find_module_path("vllm_ascend.ops.rotary_embedding")
                    cls._patch_rope_file(rope_path)
                except Exception as e:
                    logger.error(f"Failed to patch RoPE file: {e}")
                    return False
            else:
                logger.info(f"Version {version} not in REQUIRED_ROPE_VERSIONS. Skipping RoPE patch.")

            return True
        except Exception as e:
            logger.error(f"Unexpected error during patching process: {e}", exc_info=True)
            return False

    @staticmethod
    def get_version() -> str | None:
        """Retrieve vllm-ascend version from installed package metadata."""
        try:
            return importlib.metadata.version("vllm-ascend")
        except importlib.metadata.PackageNotFoundError:
            logger.debug("vllm-ascend package metadata not found.")
            return None

    
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
            
            logger.debug(f"Located {module_name} at {path}")
            return path
        except Exception as e:
            raise RuntimeError(f"Locating module {module_name} failed: {e}")

    @staticmethod
    def _backup_file(path: Path):
        """Create a timestamped backup of the target file before modification."""
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        try:
            shutil.copy2(path, backup)
            logger.info(f"Backup created: {backup}")
        except Exception as e:
            raise RuntimeError(f"Failed to create backup for {path}: {e}")

    @staticmethod
    def _find_function_block(lines: list[str], func_name: str) -> tuple[int, int] | None:
        """Find the start and end line indices of a function definition."""
        start = None
        indent = 0
        search_pattern = f"def {func_name}("
        
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(search_pattern):
                start = idx
                indent = len(line) - len(stripped)
                logger.debug(f"Found function '{func_name}' at line {idx+1} with indent {indent}")
                break
        
        if start is None:
            logger.warning(f"Function '{func_name}' not found in the provided lines.")
            return None

        # Determine the end of the block based on indentation
        end = len(lines)
        for idx in range(start + 1, len(lines)):
            stripped = lines[idx].lstrip()
            if not stripped: # Skip empty lines
                continue
            
            curr_indent = len(lines[idx]) - len(stripped)
            # If line starts with def/class/@ and indent is <= function start indent, block ended
            if stripped.startswith(("def ", "class ", "@")) and curr_indent <= indent:
                end = idx
                break
        
        logger.debug(f"Function '{func_name}' block: lines {start+1} to {end}")
        return start, end


    @classmethod
    def _patch_worker_file(cls, path: Path):
        """Inject LMCache tracking into vLLM-Ascend worker."""
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
            lines = lines[:insert_at] + cls._IMPORTS_TO_ADD + lines[insert_at:]
            changed = True
            logger.debug("Injected LMCache imports.")

        # Phase 2: Comment out distributed KV initialization
        block_dist = cls._find_function_block(lines, "_init_worker_distributed_environment")
        if block_dist:
            for i in range(block_dist[0], block_dist[1]):
                if "ensure_kv_transfer_initialized(" in lines[i] and not lines[i].lstrip().startswith("#"):
                    lines[i] = lines[i].replace("ensure_kv_transfer_initialized(", "# ensure_kv_transfer_initialized(")
                    changed = True
                    logger.debug(f"Commented out ensure_kv_transfer_initialized at line {i+1}")
        else:
            logger.warning("Could not find _init_worker_distributed_environment. Skipping this sub-patch.")

        # Phase 3: Add model registration in load_model
        if not any("VLLMModelTracker.register_model" in line for line in lines):
            block_load = cls._find_function_block(lines, "load_model")
            if block_load:
                last_idx = block_load[1] - 1
                while last_idx > block_load[0] and not lines[last_idx].strip(): 
                    last_idx -= 1
                
                # Logic: Find indentation of the function body
                indent = "        " # Default for vLLM class methods
                snippet = [
                    "\n",
                    f"{indent}VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)\n",
                    f"{indent}ensure_kv_transfer_initialized(self.vllm_config)\n"
                ]
                for i, s in enumerate(snippet):
                    lines.insert(last_idx + 1 + i, s)
                changed = True
                logger.debug(f"Injected VLLMModelTracker into load_model at line {last_idx+2}")
            else:
                logger.error("Critical: Could not find 'load_model' function in worker.")

        if changed:
            cls._backup_file(path)
            path.write_text("".join(lines), encoding="utf-8")
            logger.info(f"Successfully applied worker patches to {path}")
        else:
            logger.info(f"Worker file {path} is already patched. No changes made.")

    @classmethod
    def _patch_rope_file(cls, path: Path):
        """Force RoPE fallback by setting self.cos to None."""
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        func_name = "_rope_forward_oot"
        
        block = cls._find_function_block(lines, func_name)
        if not block:
            logger.error(f"Critical: Could not find function {func_name} in RoPE file.")
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
            logger.error(f"Could not find end of function signature for {func_name}")
            return
            
        insert_pos += 1 # Move to the first line of the function body
        
        # Calculate indentation based on the next non-empty line
        indent = "    " 
        if insert_pos < len(lines):
            for i in range(insert_pos, block[1]):
                stripped = lines[i].lstrip()
                if stripped and not stripped.startswith(('"', "'")):
                    indent = lines[i][:len(lines[i]) - len(stripped)]
                    break

        lines.insert(insert_pos, f"{indent}self.cos = None  # Force fallback - Added by LMCache\n")
        cls._backup_file(path)
        path.write_text("".join(lines), encoding="utf-8")
        logger.info(f"Successfully applied RoPE fallback patch to {path}")