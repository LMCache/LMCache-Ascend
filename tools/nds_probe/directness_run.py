#!/usr/bin/env python3
"""Directness run: NDS read, then NDS write of that file's blocks and read back.

The write target is the block device at the file's FIEMAP physical offset.
A regular-file PWRITE is rejected by the kernel. The script refuses to write
unless the file is one sector-aligned extent and FIBMAP agrees with FIEMAP.

Host-buffer evidence is sampled only around register / write / read:
process RssAnon, file-page residency (mincore), and this process's
kmalloc and mm_page_alloc trace records. Fixture writes and the final
D2H compare sit outside those windows.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import mmap
import os
import stat
import subprocess
import sys
from pathlib import Path

SIZE = 32 * 1024 * 1024
CANARY = 1 * 1024 * 1024
MOUNT = os.environ.get("NDS_TEST_MOUNT", "/mnt/disk1")
OUT_DIR = Path(os.environ.get("NDS_DIRECTNESS_OUT", "/tmp/nds-directness"))
TRACE = "/sys/kernel/debug/tracing"

FIEMAP_EXTENT_LAST = 0x00000001
FIEMAP_EXTENT_UNKNOWN = 0x00000002
FIEMAP_EXTENT_DELALLOC = 0x00000004
FIEMAP_EXTENT_ENCODED = 0x00000008
FIEMAP_EXTENT_DATA_ENCRYPTED = 0x00000080
FIEMAP_EXTENT_NOT_ALIGNED = 0x00000100
FIEMAP_EXTENT_DATA_INLINE = 0x00000200
FIEMAP_EXTENT_DATA_TAIL = 0x00000400
FIEMAP_EXTENT_UNWRITTEN = 0x00000800
UNSUPPORTED_FLAGS = (
    FIEMAP_EXTENT_UNKNOWN
    | FIEMAP_EXTENT_DELALLOC
    | FIEMAP_EXTENT_ENCODED
    | FIEMAP_EXTENT_DATA_ENCRYPTED
    | FIEMAP_EXTENT_NOT_ALIGNED
    | FIEMAP_EXTENT_DATA_INLINE
    | FIEMAP_EXTENT_DATA_TAIL
    | FIEMAP_EXTENT_UNWRITTEN
)
FS_IOC_FIEMAP = 0xC020660B
FIGETBSZ = 2
FIBMAP = 1


def pattern_bytes(size: int, seed: int, offset: int = 0) -> bytes:
    import torch

    idx = torch.arange(offset, offset + size, dtype=torch.int64)
    return ((idx * 131 + seed) & 0xFF).to(torch.uint8).numpy().tobytes()


def fill_npu(buf, seed: int) -> None:
    import torch

    n = buf.numel()
    step = 1 * 1024 * 1024
    for off in range(0, n, step):
        count = min(step, n - off)
        idx = torch.arange(off, off + count, device=buf.device, dtype=torch.int64)
        buf[off : off + count] = ((idx * 131 + seed) & 0xFF).to(torch.uint8)


def write_file(path: Path, size: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.posix_fallocate(fd, 0, size)
        off = 0
        step = 1 * 1024 * 1024
        while off < size:
            chunk = pattern_bytes(min(step, size - off), seed, off)
            os.write(fd, chunk)
            off += len(chunk)
        os.fsync(fd)
    finally:
        os.close(fd)


def fiemap(fd: int, size: int) -> list[dict]:
    # Header is 32 bytes. Each extent is 40 bytes. Ask for a handful.
    extent_count = 8
    buf = bytearray(32 + extent_count * 40)
    struct_fmt_head = "<QQIIII"
    import struct

    struct.pack_into(struct_fmt_head, buf, 0, 0, size, 0, 0, extent_count, 0)
    import fcntl

    fcntl.ioctl(fd, FS_IOC_FIEMAP, buf, True)
    _, _, _, mapped, count, _ = struct.unpack_from(struct_fmt_head, buf, 0)
    extents = []
    for i in range(mapped):
        logical, physical, length, flags, _, _, _ = struct.unpack_from(
            "<QQQIIII", buf, 32 + i * 40
        )
        extents.append(
            {
                "logical": logical,
                "physical": physical,
                "length": length,
                "flags": flags,
            }
        )
    if count != extent_count:
        raise RuntimeError(f"fiemap extent_count changed to {count}")
    return extents


def fibmap_first_physical(fd: int) -> int:
    import array
    import fcntl

    size_buf = array.array("i", [0])
    fcntl.ioctl(fd, FIGETBSZ, size_buf, True)
    block_size = int(size_buf[0])
    if block_size <= 0:
        raise RuntimeError(f"FIGETBSZ returned {block_size}")
    block = array.array("I", [0])
    fcntl.ioctl(fd, FIBMAP, block, True)
    return int(block[0]) * block_size


def block_device_for(path: Path) -> str:
    dev = os.stat(path).st_dev
    for node in Path("/dev").iterdir():
        try:
            st = node.stat()
        except OSError:
            continue
        if stat.S_ISBLK(st.st_mode) and st.st_rdev == dev:
            return str(node)
    raise RuntimeError(f"no block device for {path} dev {os.major(dev)}:{os.minor(dev)}")


def require_one_extent(path: Path, size: int) -> dict:
    fd = os.open(path, os.O_RDONLY)
    try:
        extents = fiemap(fd, size)
        fib = fibmap_first_physical(fd)
    finally:
        os.close(fd)
    if len(extents) != 1:
        raise RuntimeError(f"{path} has {len(extents)} extents, write needs one")
    ext = extents[0]
    if ext["logical"] != 0 or ext["length"] != size:
        raise RuntimeError(f"extent does not cover the file: {ext}")
    if ext["flags"] & UNSUPPORTED_FLAGS:
        raise RuntimeError(f"unsupported extent flags 0x{ext['flags']:x}")
    if ext["physical"] <= 0 or ext["physical"] % 512 or ext["length"] % 512:
        raise RuntimeError(f"extent is not a non-zero 512-byte aligned range: {ext}")
    if fib != ext["physical"]:
        raise RuntimeError(
            f"FIBMAP {fib} disagrees with FIEMAP physical {ext['physical']}"
        )
    dev = block_device_for(path)
    dev_size = os.stat(dev).st_size or int(Path(f"/sys/class/block/{Path(dev).name}/size").read_text()) * 512
    if ext["physical"] + ext["length"] > dev_size:
        raise RuntimeError("extent is past the end of the block device")
    ext["device"] = dev
    ext["device_size"] = dev_size
    return ext


def evict_file_cache(path: Path, size: int) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def resident_bytes(path: Path, size: int) -> int:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    prot_read = 1
    map_shared = 1
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, prot_read, map_shared, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value, 2**64 - 1):
            raise OSError(ctypes.get_errno(), "mmap")
        try:
            pages = (size + 4095) // 4096
            vec = (ctypes.c_ubyte * pages)()
            rc = libc.mincore(addr, size, vec)
            if rc != 0:
                raise OSError(ctypes.get_errno(), "mincore")
            return sum(1 for byte in vec if byte & 1) * 4096
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def proc_kb() -> dict[str, int]:
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS:", "RssAnon:", "RssFile:")):
            key, value = line.split()[:2]
            out[key.rstrip(":")] = int(value)
    return out


class KernelTrace:
    EVENTS = ("kmem/kmalloc", "kmem/kmalloc_node", "kmem/mm_page_alloc")

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.raw = ""
        if not os.path.isdir(TRACE):
            subprocess.check_call(["mount", "-t", "debugfs", "none", "/sys/kernel/debug"])

    def _write(self, rel: str, text: str) -> None:
        path = os.path.join(TRACE, rel)
        with open(path, "w") as fh:
            fh.write(text)

    def __enter__(self) -> "KernelTrace":
        self._write("tracing_on", "0")
        self._write("current_tracer", "nop")
        self._write("set_event", "")
        self._write("trace", "")
        filt = f"common_pid == {self.pid}"
        for event in self.EVENTS:
            base = os.path.join(TRACE, "events", event)
            if not os.path.isdir(base):
                continue
            self._write(f"events/{event}/filter", filt)
            self._write(f"events/{event}/enable", "1")
        self._write("tracing_on", "1")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._write("tracing_on", "0")
        with open(os.path.join(TRACE, "trace"), "r", errors="replace") as fh:
            self.raw = fh.read()
        for event in self.EVENTS:
            base = os.path.join(TRACE, "events", event)
            if os.path.isdir(base):
                self._write(f"events/{event}/enable", "0")
        self._write("set_event", "")

    def summary(self) -> dict:
        kmalloc_max = 0
        kmalloc_bytes = 0
        kmalloc_count = 0
        page_bytes = 0
        page_count = 0
        large = []
        entries = ""
        for line in self.raw.splitlines():
            if line.startswith("# entries-in-buffer"):
                entries = line.split(":", 1)[-1].strip()
            if "kmalloc" in line and "bytes_alloc=" in line:
                kmalloc_count += 1
                for token in line.split():
                    if token.startswith("bytes_alloc="):
                        nbytes = int(token.split("=", 1)[1])
                        kmalloc_bytes += nbytes
                        kmalloc_max = max(kmalloc_max, nbytes)
                        if nbytes >= 1024 * 1024:
                            large.append(line.strip()[:240])
            elif "mm_page_alloc:" in line and "order=" in line:
                page_count += 1
                for token in line.split():
                    if token.startswith("order="):
                        order = int(token.split("=", 1)[1])
                        page_bytes += (1 << order) * 4096
                        if order >= 8:
                            large.append(line.strip()[:240])
        return {
            "entries": entries,
            "kmalloc_count": kmalloc_count,
            "kmalloc_bytes": kmalloc_bytes,
            "kmalloc_max": kmalloc_max,
            "page_alloc_count": page_count,
            "page_alloc_bytes": page_bytes,
            "large_records": large[:12],
        }


def measure(fn) -> dict:
    before = proc_kb()
    with KernelTrace() as trace:
        fn()
    after = proc_kb()
    summary = trace.summary()
    summary["rss_anon_before_kb"] = before.get("RssAnon")
    summary["rss_anon_after_kb"] = after.get("RssAnon")
    summary["rss_anon_delta_kb"] = after.get("RssAnon", 0) - before.get("RssAnon", 0)
    summary["vmrss_delta_kb"] = after.get("VmRSS", 0) - before.get("VmRSS", 0)
    return summary


def direct_host_read(path: Path, size: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        mapping = mmap.mmap(-1, size)
        view = memoryview(mapping)
        try:
            got = os.preadv(fd, [view], 0)
            if got != size:
                raise RuntimeError(f"O_DIRECT read {got} of {size}")
            return bytes(view)
        finally:
            view.release()
            mapping.close()
    finally:
        os.close(fd)


def _save(report: dict) -> None:
    text = json.dumps(report, indent=2)
    (OUT_DIR / "report.json").write_text(text + "\n")
    print(text, flush=True)


def main() -> int:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "size": SIZE,
        "device": f"npu:0",
        "npu_available": bool(torch.npu.is_available()),
    }
    if not report["npu_available"]:
        report["error"] = "torch.npu is not available"
        print(json.dumps(report, indent=2))
        return 1

    from lmcache_ascend.v1.gds.openeuler_nds_library import OpenEulerNDSLibrary

    report["trace_self_check"] = trace_self_check()
    lib = OpenEulerNDSLibrary(library_path=os.environ["NDS_LIBRARY_PATH"])
    lib.driver_open()
    root = Path(MOUNT) / "m00926961" / "nds_directness"
    try:
        report["read_only"] = run_read_only(lib, root)
        _save(report)
        report["canary_write_read"] = run_write_read(
            lib, root / "canary.bin", CANARY, seed_file=3, seed_npu=4
        )
        _save(report)
        if not report["canary_write_read"]["match"]:
            report["write_read"] = {
                "skipped": "canary did not match; 32 MiB block-device write was not run"
            }
        else:
            report["write_read"] = run_write_read(
                lib, root / "write_read.bin", SIZE, seed_file=5, seed_npu=6
            )
        report["rss_control"] = rss_control()
    finally:
        lib.driver_close()
        _save(report)

    _save(report)
    ok = report["read_only"]["match"] and report.get("write_read", {}).get("match") is True
    return 0 if ok else 1


def trace_self_check() -> dict:
    path = "/tmp/nds-trace-self-check.bin"
    summary = measure(lambda: open(path, "wb").write(b"x" * 4096))
    summary["saw_kernel_alloc"] = bool(
        summary["kmalloc_count"] or summary["page_alloc_count"]
    )
    return summary


def rss_control() -> dict:
    before = proc_kb()
    blob = bytearray(SIZE)
    after = proc_kb()
    del blob
    return {
        "bytes": SIZE,
        "rss_anon_delta_kb": after.get("RssAnon", 0) - before.get("RssAnon", 0),
    }


def run_read_only(lib, root: Path) -> dict:
    path = root / "read_only.bin"
    write_file(path, SIZE, seed=1)
    evict_file_cache(path, SIZE)
    cold = resident_bytes(path, SIZE)
    fd = os.open(path, os.O_RDONLY)
    lib.file_register(fd)
    import torch

    buf = torch.zeros(SIZE, dtype=torch.uint8, device="npu:0")
    reg = measure(lambda: lib.register_memory(buf.data_ptr(), SIZE))
    evict_file_cache(path, SIZE)
    before_resident = resident_bytes(path, SIZE)
    io = measure(lambda: lib.read(fd, buf.data_ptr(), SIZE, 0, 0))
    after_resident = resident_bytes(path, SIZE)
    got = buf.cpu()
    expect = ((torch.arange(SIZE) * 131 + 1) & 0xFF).to(torch.uint8)
    match = bool(torch.equal(got, expect))
    lib.unregister_memory(buf.data_ptr(), SIZE)
    os.close(fd)
    return {
        "path": str(path),
        "match": match,
        "page_cache_after_fixture_drop": cold,
        "page_cache_before_read": before_resident,
        "page_cache_after_read": after_resident,
        "register": reg,
        "read": io,
    }


def run_write_read(lib, path: Path, size: int, seed_file: int, seed_npu: int) -> dict:
    write_file(path, size, seed=seed_file)
    ext = require_one_extent(path, size)
    import torch

    src = torch.empty(size, dtype=torch.uint8, device="npu:0")
    fill_npu(src, seed_npu)
    dst = torch.zeros(size, dtype=torch.uint8, device="npu:0")
    reg_src = measure(lambda: lib.register_memory(src.data_ptr(), size))
    reg_dst = measure(lambda: lib.register_memory(dst.data_ptr(), size))
    file_fd = os.open(path, os.O_RDONLY)
    dev_fd = os.open(ext["device"], os.O_RDWR)
    lib.file_register(file_fd)
    lib.file_register(dev_fd)
    try:
        written = measure(
            lambda: lib.write(dev_fd, src.data_ptr(), size, ext["physical"], 0)
        )
        evict_file_cache(path, size)
        before_resident = resident_bytes(path, size)
        read = measure(lambda: lib.read(file_fd, dst.data_ptr(), size, 0, 0))
        after_resident = resident_bytes(path, size)
        got = dst.cpu()
        expect = ((torch.arange(size) * 131 + seed_npu) & 0xFF).to(torch.uint8)
        match = bool(torch.equal(got, expect))
        disk = direct_host_read(path, size)
        disk_match = disk == pattern_bytes(size, seed_npu)
    finally:
        lib.unregister_memory(src.data_ptr(), size)
        lib.unregister_memory(dst.data_ptr(), size)
        os.close(file_fd)
        os.close(dev_fd)
    return {
        "path": str(path),
        "extent": ext,
        "match": match,
        "o_direct_matches_npu_pattern": disk_match,
        "page_cache_before_read": before_resident,
        "page_cache_after_read": after_resident,
        "register_src": reg_src,
        "register_dst": reg_dst,
        "write": written,
        "read": read,
    }


if __name__ == "__main__":
    sys.exit(main())
