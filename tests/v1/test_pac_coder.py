# SPDX-License-Identifier: Apache-2.0
# Module under test
# Third Party
from lmcache.storage_backend.serde.cachegen_basics import (
    CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK,
)
from lmcache.storage_backend.serde.cachegen_encoder import (
    torch_quant_vectorized,
)
import pytest
import torch
import torch_npu

# First Party
import lmcache_ascend.c_ops as lmc_ops


@pytest.mark.parametrize("layers", [1, 4])
@pytest.mark.parametrize("channels", [32, 128])
@pytest.mark.parametrize("tokens", [20, 256, 1000])
@pytest.mark.parametrize("chunk_size", [CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK])
def test_basic_encode_ii(layers, channels, tokens, chunk_size):
    device = "npu"
    dtype = torch.bfloat16

    shape = [layers, tokens, channels]
    full_x = torch.normal(0, 2, shape).to(dtype=dtype).to(device=device)

    bins = [32] + [32] * (layers - 1)
    bins = torch.tensor(bins, dtype=dtype).to(device=device)

    syms_in, _ = torch_quant_vectorized(bins, full_x)

    meta_data = torch.zeros([layers, channels, 32], dtype=torch.uint16).to(device="npu")
    lmc_ops.pac_prepare_enc_metadata(syms_in, meta_data)
    # Sync ensures kernel crashes are attributed to a consistent line
    torch_npu.npu.synchronize()

    for chunk_start in range(0, tokens, chunk_size):
        chunk_end = min(tokens, chunk_start + chunk_size)
        t_in_chunk = chunk_end - chunk_start

        bytesstream = torch.zeros([layers, channels, chunk_size], dtype=torch.uint8).to(
            device="npu"
        )
        cum_lens = torch.zeros([layers, channels], dtype=torch.int32).to(device="npu")
        syms_out = torch.zeros([layers, t_in_chunk, channels], dtype=torch.uint8).to(
            device="npu"
        )
        s_in = syms_in[:, chunk_start:chunk_end, :].clone()
        lmc_ops.pac_encode(s_in, meta_data, bytesstream, cum_lens)
        # Sync ensures kernel crashes are attributed to a consistent line
        torch_npu.npu.synchronize()

        lmc_ops.pac_decode(meta_data, bytesstream, cum_lens, syms_out)
        # Sync ensures kernel crashes are attributed to a consistent line
        torch_npu.npu.synchronize()

        assert (syms_out == syms_in[:, chunk_start:chunk_end, :]).all()
