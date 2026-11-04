# SPDX-License-Identifier: Apache-2.0
"""Hardware NDS direct read tests (skip without NPU and NDS_TEST_DIR)."""

# Standard
import os

# Third Party
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("NDS_TEST_DIR"),
    reason="set NDS_TEST_DIR to a scratch DPC/DTFS directory",
)


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


requires_npu = pytest.mark.skipif(not _npu_available(), reason="NPU not available")


@requires_npu
def test_nds_read_file_into_npu_buffer_without_lmcache(tmp_path):
    from lmcache_ascend.v1.gds.nds_library import get_nds_library

    test_dir = os.environ["NDS_TEST_DIR"]
    chunk = 4096
    path = os.path.join(test_dir, "nds_direct_read_test.bin")
    with open(path, "wb") as f:
        f.write(bytes((i % 251 for i in range(chunk))))

    lib = get_nds_library()
    lib.driver_open()
    fd = os.open(path, os.O_RDONLY)
    try:
        handle = lib.file_register(fd)
        buf = torch.empty(chunk, dtype=torch.uint8, device="npu:0")
        done = lib.read(handle, buf.data_ptr(), chunk, 0, 0)
        assert done == chunk
        expected = (torch.arange(chunk, dtype=torch.int32) % 251).to(torch.uint8)
        assert torch.equal(buf.cpu(), expected)
        lib.file_deregister(handle)
    finally:
        os.close(fd)
