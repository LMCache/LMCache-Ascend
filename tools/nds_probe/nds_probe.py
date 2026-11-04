#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone NDS qualification probe (PR 0). Does not import LMCache."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass, field
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_LIBRARY = "/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so"


@dataclass
class NDSProbeConfig:
    library_path: str = DEFAULT_LIBRARY
    test_dir: str = ""
    device_index: int = 0
    chunk_bytes: int = 4096
    artifacts_dir: str = "artifacts/nds_probe"


@dataclass
class NDSProbeResult:
    exports: list[str] = field(default_factory=list)
    can_read_npu: bool | None = None
    can_write_npu: bool | None = None
    random_offsets: bool | None = None
    preallocated_slab: bool | None = None
    async_or_stream: bool | None = None
    short_io_behavior: str = ""
    directness_notes: str = ""
    versions: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def inspect_exports(library_path: str) -> list[str]:
    if not os.path.isfile(library_path):
        raise FileNotFoundError(library_path)
    out = subprocess.check_output(["nm", "-D", library_path], text=True, errors="replace")
    symbols: list[str] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in ("T", "W"):
            symbols.append(parts[-1])
    return sorted(set(symbols))


def collect_directness_evidence(result: NDSProbeResult) -> None:
    result.directness_notes = (
        "Confirm with driver/IOMMU traces that payload bytes do not pass through "
        "a host bounce buffer. Exclude fixture writes and assertion D2H copies."
    )


def probe_direct_read(config: NDSProbeConfig, result: NDSProbeResult) -> None:
    try:
        import torch

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            result.errors.append("NPU not available for probe_direct_read")
            result.can_read_npu = False
            return
        from lmcache_ascend.v1.gds.nds_library import get_nds_library

        lib = get_nds_library(config.library_path)
        lib.driver_open()
        test_path = _fixture_file(config)
        fd = os.open(test_path, os.O_RDONLY)
        handle = None
        try:
            handle = lib.file_register(fd)
            buf = torch.empty(config.chunk_bytes, dtype=torch.uint8, device=f"npu:{config.device_index}")
            base = buf.data_ptr()
            done = lib.read(handle, base, config.chunk_bytes, 0, 0)
            result.can_read_npu = done == config.chunk_bytes
        finally:
            if handle is not None:
                lib.file_deregister(handle)
            os.close(fd)
    except Exception as exc:
        result.can_read_npu = False
        result.errors.append(f"probe_direct_read: {exc}")


def probe_direct_write(config: NDSProbeConfig, result: NDSProbeResult) -> None:
    try:
        from lmcache_ascend.v1.gds.nds_library import get_nds_library

        caps = get_nds_library(config.library_path).capabilities()
        if not caps.can_write:
            result.can_write_npu = False
            result.errors.append("No NDS write symbol in library")
            return
        import torch

        if not torch.npu.is_available():
            result.can_write_npu = False
            return
        lib = get_nds_library(config.library_path)
        test_path = _fixture_file(config)
        fd = os.open(test_path, os.O_RDWR)
        try:
            handle = lib.file_register(fd)
            buf = torch.full(
                (config.chunk_bytes,),
                0xAB,
                dtype=torch.uint8,
                device=f"npu:{config.device_index}",
            )
            done = lib.write(handle, buf.data_ptr(), config.chunk_bytes, 0, 0)
            result.can_write_npu = done == config.chunk_bytes
            lib.file_deregister(handle)
        finally:
            os.close(fd)
    except Exception as exc:
        result.can_write_npu = False
        result.errors.append(f"probe_direct_write: {exc}")


def probe_random_offsets(config: NDSProbeConfig, result: NDSProbeResult) -> None:
    result.random_offsets = None
    result.errors.append("probe_random_offsets: implement on hardware with aligned offsets")


def probe_short_io(config: NDSProbeConfig, result: NDSProbeResult) -> None:
    result.short_io_behavior = "negative ssize_t or short count must raise in NDSLibrary"


def _fixture_file(config: NDSProbeConfig) -> str:
    if not config.test_dir:
        raise ValueError("test_dir required for I/O probes")
    os.makedirs(config.test_dir, exist_ok=True)
    path = os.path.join(config.test_dir, "nds_probe_fixture.bin")
    if not os.path.exists(path) or os.path.getsize(path) < config.chunk_bytes:
        with open(path, "wb") as f:
            f.write(bytes((i % 251 for i in range(config.chunk_bytes))))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qualify Huawei NDS for LMCache")
    parser.add_argument("--library", default=DEFAULT_LIBRARY)
    parser.add_argument("--test-dir", default=os.environ.get("NDS_TEST_DIR", ""))
    parser.add_argument("--artifacts", default="artifacts/nds_probe")
    args = parser.parse_args(argv)

    config = NDSProbeConfig(
        library_path=args.library,
        test_dir=args.test_dir,
        artifacts_dir=args.artifacts,
    )
    result = NDSProbeResult()
    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    try:
        result.exports = inspect_exports(config.library_path)
        (artifacts / "exports.txt").write_text("\n".join(result.exports) + "\n")
    except Exception as exc:
        result.errors.append(f"inspect_exports: {exc}")

    if config.test_dir:
        probe_direct_read(config, result)
        probe_direct_write(config, result)
        probe_random_offsets(config, result)
    else:
        result.errors.append("Set --test-dir or NDS_TEST_DIR for I/O probes")

    probe_short_io(config, result)
    collect_directness_evidence(result)

    (artifacts / "report.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    (artifacts / "directness.txt").write_text(result.directness_notes + "\n")
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.can_read_npu and result.can_write_npu else 1


if __name__ == "__main__":
    sys.exit(main())
