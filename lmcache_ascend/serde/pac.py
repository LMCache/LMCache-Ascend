# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List

# Third Party
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_basics import (
    CacheGenConfig,
    CacheGenGPUBytestream,
    CacheGenGPUEncoderOutput,
)
from lmcache.storage_backend.serde.cachegen_encoder import (
    _split_kv,
    torch_quant_vectorized,
)
from lmcache.utils import _lmcache_nvtx_annotate
import torch
import torch_npu

# First Party
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
def pac_decode_function(
    meta_data: torch.Tensor,
    data_chunks: List[CacheGenGPUBytestream],
    layers_in_key: int,
    n_tokens: int,
    output: torch.Tensor,
):
    chunk_size = 256  # No support for variable cachengen chunk size
    nlayers, nchannels, _ = meta_data.shape  # [layer, channels, bins]

    decode_stream = torch_npu.npu.Stream()
    e = None
    with torch_npu.npu.stream(decode_stream):
        output = output.flatten()
        chunked_output = torch.split(output, chunk_size * nchannels * nlayers, dim=0)
        shaped_chunked_output = []
        for c in chunked_output:
            c = c.reshape((nlayers, c.size()[0] // (nlayers * nchannels), nchannels))
            shaped_chunked_output.append(c)

        in_out_pairs = zip(data_chunks, shaped_chunked_output, strict=True)
        for data_chunk, output_chunk in in_out_pairs:
            bytes_tensor = data_chunk.bytestream
            cum_lens = data_chunk.bytestream_lengths.flatten().reshape(
                (nlayers, nchannels)
            )
            lmc_ops.pac_decode(meta_data, bytes_tensor, cum_lens, output_chunk)
            e = decode_stream.record_event(e)

        output = torch.cat(shaped_chunked_output, 1)
        out = output.reshape((2, layers_in_key, n_tokens, nchannels))
        key, value = out.float()

    e = decode_stream.record_event(e)
    e.synchronize()

    return key, value


@_lmcache_nvtx_annotate
def pac_encode_function(
    kv: torch.Tensor,
    config: CacheGenConfig,
    key_bins: torch.Tensor,
    value_bins: torch.Tensor,
    n_tokens: int,
) -> CacheGenGPUEncoderOutput:
    chunk_size = 256  # No support for variable cachengen chunk size
    n_chunks = (
        n_tokens // chunk_size
        if n_tokens % chunk_size == 0
        else (n_tokens // chunk_size) + 1
    )
    n_bins = int(max(key_bins.max(), value_bins.max()).item())

    num_heads, head_size = kv.shape[-2:]
    fp_k, fp_v = _split_kv(kv)
    nchannels = num_heads * head_size
    nlayers = fp_k.shape[0] + fp_v.shape[0]

    new_key, max_tensors_key = torch_quant_vectorized(key_bins, fp_k)
    new_value, max_tensors_value = torch_quant_vectorized(value_bins, fp_v)
    encode_input = torch.cat((new_key, new_value), dim=0).reshape(
        nlayers, n_tokens, nchannels
    )

    meta_data = torch.zeros(
        (nlayers, nchannels, n_bins), dtype=torch.int16, device="npu"
    )

    lmc_ops.pac_prepare_enc_metadata(encode_input, meta_data)

    output_buffer = torch.zeros(
        (nlayers, nchannels, chunk_size), dtype=torch.uint8, device="npu"
    )
    output_lengths = torch.zeros((nlayers, nchannels), dtype=torch.int32, device="npu")

    data_chunks = []
    for chunk_ii in range(0, n_chunks):
        start = chunk_ii * chunk_size
        end = min((chunk_ii + 1) * chunk_size, n_tokens)

        output_buffer.zero_()
        output_lengths.zero_()
        tmp_in = encode_input[:, start:end, :].clone()

        lmc_ops.pac_encode(tmp_in, meta_data, output_buffer, output_lengths)
        max_len = output_lengths[-1, -1]

        data_chunks.append(
            CacheGenGPUBytestream(
                bytestream=output_buffer.flatten()[0:max_len].clone(),
                bytestream_lengths=output_lengths.clone(),
                ntokens=end - start,
            )
        )

    return CacheGenGPUEncoderOutput(
        data_chunks,
        meta_data,
        max_tensors_key=max_tensors_key,
        max_tensors_value=max_tensors_value,
        num_heads=num_heads,
        head_size=head_size,
    )
