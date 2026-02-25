# SPDX-License-Identifier: Apache-2.0
# Module under test
# Third Party
from lmcache.storage_backend.serde.cachegen_basics import (
    CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK,
    CacheGenConfig,
    CacheGenGPUEncoderOutput,
    QuantizationSpec,
)
from lmcache.storage_backend.serde.cachegen_decoder import decode_function_gpu
from lmcache.storage_backend.serde.cachegen_encoder import (
    _split_kv,
    encode_function,
    torch_quant_vectorized,
)
import pytest
import torch
import torch_npu


@pytest.mark.parametrize("n_tokens", [20, 256, 1000])
@pytest.mark.parametrize("n_layers", [1, 28])
@pytest.mark.parametrize("chunk_size", [CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK])
@pytest.mark.parametrize("num_heads", [3, 8])
@pytest.mark.parametrize("head_size", [64, 128])
def test_encode_into_decode(n_tokens, n_layers, chunk_size, num_heads, head_size):
    # This test relies on reproducing the context provided by CacheGenSerializer and
    # CacheGenDeserializer before they encode/decode.
    device = "npu"
    dtype = torch.bfloat16

    # Minor test limitation - this construction of cachegen config assumes that we're in
    # one of 2 worlds (n_layers == 1 or >10). If you need to test n_layers 2 - 10,
    # update the config construction step
    assert n_layers == 1 or n_layers > 10
    if n_layers == 1:
        cache_gen_config = CacheGenConfig(
            nlayers=n_layers,
            kspecs=[QuantizationSpec(start_layer=0, end_layer=n_layers, bins=32)],
            vspecs=[QuantizationSpec(start_layer=0, end_layer=n_layers, bins=32)],
        )
    else:
        cache_gen_config = CacheGenConfig(
            nlayers=n_layers,
            kspecs=[
                QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                QuantizationSpec(start_layer=10, end_layer=n_layers, bins=16),
            ],
            vspecs=[
                QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                QuantizationSpec(start_layer=2, end_layer=n_layers, bins=16),
            ],
        )

    v_bins = torch.zeros(n_layers).to(device=device)
    for spec in cache_gen_config.vspecs:
        v_bins[spec.start_layer : spec.end_layer] = spec.bins

    k_bins = torch.zeros(n_layers).to(device=device)
    for spec in cache_gen_config.kspecs:
        k_bins[spec.start_layer : spec.end_layer] = spec.bins

    shape = [n_layers, 2, n_tokens, num_heads, head_size]
    kv_full = torch.normal(0, 2, shape).to(dtype=dtype).to(device=device)
    torch_npu.npu.synchronize()

    expected_chunks = []
    byte_stream_full = []

    for chunk_start in range(0, n_tokens, chunk_size):
        chunk_end = min(n_tokens, chunk_start + chunk_size)
        kv_chunk = kv_full[:, :, chunk_start:chunk_end, :, :]

        fp_k, fp_v = _split_kv(kv_chunk)
        expected_key, _ = torch_quant_vectorized(k_bins, fp_k)
        expected_value, _ = torch_quant_vectorized(v_bins, fp_v)
        expected_chunks.append((expected_key, expected_value))

        encode_output = encode_function(
            kv_chunk, cache_gen_config, k_bins, v_bins, chunk_end - chunk_start
        )

        torch_npu.npu.synchronize()

        byte_stream_full.append(encode_output.to_bytes())

    decoded_chunks = []

    for byte_stream in byte_stream_full:
        decoder_input = CacheGenGPUEncoderOutput.from_bytes(byte_stream)
        decode_n_tokens = decoder_input.max_tensors_key.shape[1]
        decode_layers = decoder_input.cdf.shape[0] // 2
        decode_channels = decoder_input.cdf.shape[1]

        decoder_output_size = chunk_size * 2 * decode_layers * decode_channels
        decoder_output_T = (
            torch.zeros(decoder_output_size, dtype=torch.uint8)
            .to(device="npu")
            .reshape((chunk_size, 2 * decode_layers * decode_channels))
        )

        torch_npu.npu.synchronize()

        decoded_chunks.append(
            decode_function_gpu(
                decoder_input.cdf,
                decoder_input.data_chunks,
                decode_layers,
                decode_n_tokens,
                decoder_output_T,
            )
        )

        torch_npu.npu.synchronize()

    for expected_chunk, decoded_chunk in zip(
        expected_chunks, decoded_chunks, strict=False
    ):
        decoded_key, decoded_value = decoded_chunk
        expected_key, expected_value = expected_chunk
        assert (expected_key == decoded_key).all()
        assert (expected_value == decoded_value).all()
