# SPDX-License-Identifier: Apache-2.0
# Module under test
# Third Party
from lmcache.storage_backend.serde.cachegen_basics import (
    CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK,
    CacheGenConfig,
    CacheGenGPUEncoderOutput,
    QuantizationSpec,
)
from lmcache.storage_backend.serde.cachegen_decoder import (
    decode_function_gpu,
)
from lmcache.storage_backend.serde.cachegen_encoder import (
    _split_kv,
    encode_function,
    torch_quant_vectorized,
)
import pytest
import torch
import torch_npu

# First Party
import lmcache_ascend.c_ops as lmc_ops


# Generating relastic and obviously correct input/output for encode is hard in all but
# the most trivial cases. Fortunately what we really care about is that
#  - decode recovers the symbols that were input to encode
#  - the intermediate representation is compressed at roughly the right ratio
#
# The testing here covers some very simplified test cases of encode/decode in isolation.
# See later tests for tests of encode/decode as a pair in more realistic cases
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("channels", [1, 32, 256])
@pytest.mark.parametrize("bits_for_symbol", [2, 3, 4, 5, 6])
def test_basic_encode(layers, channels, bits_for_symbol):
    # With 2^n symbols uniformly distributed symbols, we expect exactly n bits per
    # symbol (+ some tail bits from a minor detail of the implemented algorithm). This
    # is because the encoding ends up being the binary representation.  So "0" = 0b00,
    # "1" = 0b01, "2" = 0b10 and "3" = 0b11 if we're in the 2 bit case.
    #
    # More interesting CDFs lead to more interesting encoding which this test does not
    # cover.

    # Arrange - symbols in the 2^n range
    bits = bits_for_symbol
    max_sym = 2**bits  # 0 through max_sym - 1
    symbols = (
        list([sym for sym in range(0, max_sym) for channel in range(0, channels)])
        * layers
    )
    n_syms = int(len(symbols) / (layers * channels))
    shape = (layers, n_syms, channels)
    symbols_T = torch.tensor(symbols).to(device="npu").to(torch.int8).reshape(shape)

    # Arrange - perfectly even CDF
    bins = max_sym + 1
    cdf = [bin * (0x10000 / (bins - 1)) for bin in range(0, bins)] * layers * channels
    cdf_shape = (layers, channels, bins)
    cdf_T = torch.Tensor(cdf).to(device="npu").to(torch.int16).reshape(cdf_shape)

    # Arrange - output buffers
    output_shape = (layers, channels, CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK)
    output_buf_T = (
        torch.zeros(
            CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK * layers * channels, dtype=torch.uint8
        )
        .to(device="npu")
        .reshape(output_shape)
    )

    output_lens = [0] * layers * channels
    output_lens_shape = (layers, channels)
    output_lens_T = (
        torch.Tensor(output_lens)
        .to(device="npu")
        .to(torch.int32)
        .reshape(output_lens_shape)
    )

    torch_npu.npu.synchronize()

    # Act
    lmc_ops.encode_fast_new(cdf_T, symbols_T, output_buf_T, output_lens_T)
    torch_npu.npu.synchronize()

    # Assert - see above about the expected encoding of each symbol being the binary
    # representation of the the symbol
    str_rep = ""
    for token in range(0, max_sym):
        if bits == 2:
            str_rep += f"{token:02b}"  # 4
        elif bits == 3:
            str_rep += f"{token:03b}"  # 8
        elif bits == 4:
            str_rep += f"{token:04b}"  # 16
        elif bits == 5:
            str_rep += f"{token:05b}"  # 32
        elif bits == 6:
            str_rep += f"{token:06b}"  # 64
        else:
            raise AssertionError()
    if len(str_rep) % 8 != 0:
        str_rep += "0" * (8 - len(str_rep) % 8)
    expected = list(
        [int(str_rep[ii : ii + 8], 2) for ii in range(0, len(str_rep), 8)]
    )  # Break it into bytes
    expected_len = len(expected)
    expected += [0] * (
        CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK - len(expected)
    )  # Pad to match output size

    expected = (
        torch.Tensor(expected)
        .to(device="npu")
        .to(torch.uint8)
        .broadcast_to(output_shape)
    )
    assert (expected == output_buf_T).all()

    expected_lens_T = (
        torch.full_like(output_lens_T, expected_len).to(device="npu").to(torch.int32)
    )
    assert (expected_lens_T == output_lens_T).all()


def test_basic_non_uniform_encode():
    # When the symbols are not uniformly distributed. For example 0: 50%, 1: 0%, 2: 25%,
    # 3: 25%.
    #
    # Fewer bits are used to encode those more common symbols. In this simplified case
    # it works out as 0 -> '0', 1 -> n/a, 2 -> '10', 3 -> '11'

    # Arrange
    max_sym = 4
    n_syms = 4
    symbols = [0, 0, 2, 3]
    symbols_T = (
        torch.Tensor(symbols).to(device="npu").to(torch.int8).reshape((1, n_syms, 1))
    )

    bins = max_sym + 1
    cdf = [0x0000, 0x8000, 0x8000, 0xC000, 0x10000]
    cdf_T = torch.Tensor(cdf).to(device="npu").to(torch.int16).reshape((1, 1, bins))

    output_buf_T = (
        torch.zeros(n_syms, dtype=torch.uint8).to(device="npu").reshape((1, 1, n_syms))
    )

    output_lens = [0]
    output_lens_T = (
        torch.Tensor(output_lens).to(device="npu").to(torch.int32).reshape((1, 1))
    )

    # Act
    lmc_ops.encode_fast_new(cdf_T, symbols_T, output_buf_T, output_lens_T)
    torch_npu.npu.synchronize()

    # Assert
    #
    # Given coding: 0 -> '0', 1 -> n/a, 2 -> '10', 3 -> '11', [0, 0, 2, 3] becomes.
    # '001011' + '00' to complete the byte
    assert output_buf_T[0][0][0].tolist() == int("00101100", 2)


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("channels", [32, 256])  # Restriction channels % 32 == 0
@pytest.mark.parametrize("bits_for_symbol", [2, 3, 4, 5])
def test_basic_decode(layers, channels, bits_for_symbol):
    # Broadly, this is the inverse of test_basic_encode. By using the simplifying
    # assumption of 2^n_bits equally probable symbols the encoded form simplifies to
    # become the binary representation of the symbols

    # Arrange - create a string of the encoded form given the above assumption
    max_sym = 2**bits_for_symbol  # 0 through max_sym - 1
    n_syms = max_sym  # per channel
    str_rep = ""
    for token in range(0, max_sym):
        if bits_for_symbol == 2:
            str_rep += f"{token:02b}"  # 4
        elif bits_for_symbol == 3:
            str_rep += f"{token:03b}"  # 8
        elif bits_for_symbol == 4:
            str_rep += f"{token:04b}"  # 16
        elif bits_for_symbol == 5:
            str_rep += f"{token:05b}"  # 32
        elif bits_for_symbol == 6:
            str_rep += f"{token:06b}"  # 64 - not supported by v2 decode kernel
        else:
            raise AssertionError()
    if len(str_rep) % 8 != 0:
        str_rep += "0" * (8 - len(str_rep) % 8)
    byte_stream = list(
        [int(str_rep[ii : ii + 8], 2) for ii in range(0, len(str_rep), 8)]
    )
    byte_stream_T = (
        torch.Tensor(byte_stream * layers * channels).to(device="npu").to(torch.uint8)
    )

    lens_T = (
        torch.Tensor([len(byte_stream)] * layers * channels)
        .to(device="npu")
        .to(torch.int32)
    )
    lens_T = lens_T.cumsum(0).reshape((layers, channels))

    bins = max_sym + 1
    factor = 0x10000 / (bins - 1)
    cdf = [int(b * factor) | bits_for_symbol for b in range(0, bins)]
    cdf = cdf * layers * channels
    cdf_T = (
        torch.Tensor(cdf)
        .to(device="npu")
        .to(torch.int16)
        .reshape((layers, channels, bins))
    )

    output_shape = (layers, n_syms, channels)
    output_buf_T = (
        torch.zeros(n_syms * layers * channels, dtype=torch.uint8)
        .to(device="npu")
        .reshape(output_shape)
    )

    # Act
    lmc_ops.decode_fast_prefsum(cdf_T, byte_stream_T, lens_T, output_buf_T)
    torch_npu.npu.synchronize()

    # Assert
    expected_symbols = (
        list([sym for sym in range(0, max_sym) for channel in range(0, channels)])
        * layers
    )
    expected_symbols_T = (
        torch.Tensor(expected_symbols)
        .to(device="npu")
        .to(torch.int8)
        .reshape(output_shape)
    )

    assert (expected_symbols_T == output_buf_T).all()


@pytest.mark.parametrize("n_tokens", [1, 20, 256, 8000])
@pytest.mark.parametrize("n_layers", [1, 28])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("head_size", [128])
def test_encode_into_decode(n_tokens, n_layers, chunk_size, num_heads, head_size):
    # Unlike the preceding tests which are aiming to be focussed, simplifying tests,
    # this test aims to be a representative test of the CacheGen process
    #
    # It relies a little on reproducing the context provided by CacheGenSerializer and
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

        decoder_output_size = decode_n_tokens * 2 * decode_layers * decode_channels
        decoder_output_T = (
            torch.zeros(decoder_output_size, dtype=torch.uint8)
            .to(device="npu")
            .reshape((decode_n_tokens, 2 * decode_layers * decode_channels))
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
