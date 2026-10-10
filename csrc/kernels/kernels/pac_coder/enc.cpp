#include "kernel_operator.h"
using namespace AscendC;

namespace kvcache_ops {
namespace pac_coder {

constexpr int32_t DATABLOCK_BYTES = 32;
constexpr int32_t N_C_PER_BLOCK = 32;
static_assert(N_C_PER_BLOCK % DATABLOCK_BYTES == 0);

// Some intializations implicitly assume 32 channels - update as appropriate before changing this assert
static_assert(N_C_PER_BLOCK == 32);

constexpr int32_t N_DBs_PER_BLOCK = N_C_PER_BLOCK / DATABLOCK_BYTES; // Data blocks per copy block (assuming u8)
constexpr int32_t N_C_MAX = 4096;

constexpr int32_t N_T_MAX = 256;
static_assert(N_T_MAX % 256 == 0);

constexpr int32_t N_B_MAX = 32;
static_assert(N_B_MAX % 32 == 0);

constexpr int32_t N_T_PER_BATCH = N_T_MAX * N_C_PER_BLOCK;

namespace impl {
__aicore__ inline auto ceil_32(int32_t size) -> uint32_t {
    return size % 32 == 0 ? size : 32 * (1 + (size / 32));
};

class PacEncoder {
public:
    __aicore__ inline PacEncoder(
        GM_ADDR in_syms, // In symbols [n_layers, n_tokens, n_channels], uint8
        GM_ADDR out_meta, // Out meta [n_layers, n_channels, n_bins], uint16
        GM_ADDR out_bytes, // Out bytes [n_layers, n_channels, batch_size], uint8
        GM_ADDR out_lens, // Out lengths [n_layers, n_channels], uint32

        TPipe& pipe,

        int32_t n_tokens,
        int32_t n_layers,
        int32_t n_channels,
        uint32_t n_bins,
        int32_t chunk_size,
        half scale_factor);

    __aicore__ inline void meta_data_calc(int layer_id, int channel_start_id);
    __aicore__ inline void encode(int layer_id, int channel_id);
private:
    // Input Queues
    TQue<TPosition::VECIN, 1> symInQ;
    GlobalTensor<uint8_t> g_syms;

    // Output Queues
    TQue<TPosition::VECOUT, 1> byteStreamOutQ;
    GlobalTensor<uint8_t> g_out_bytes;

    TQue<TPosition::VECOUT, 1> lensOutQ;
    GlobalTensor<uint32_t> g_out_lens;

    TQue<TPosition::VECOUT, 1> metaDataOutQ;
    TQue<TPosition::VECIN, 1> metaDataInQ;
    GlobalTensor<uint16_t> gm_meta_data;

    TPipe& pipe;

    // Dimensionality
    int32_t n_tokens;
    int32_t n_layers;
    int32_t n_channels;
    uint32_t n_bins;
    int32_t chunk_size;

    half scale_factor;

    int32_t n_tokens_per_layer;

    // Intermediate buffers
    TBuf<TPosition::VECCALC> calcBuf; // For transient (per `encode()`) tensors
    TBuf<TPosition::VECCALC> utilsCalcBuf; // For class lifetime tensors

    // General use registers that can be reused
    LocalTensor<int32_t> register_bins_x_channels; // N_BINS x N_CHANNELS
    LocalTensor<int32_t> register_bins_x_channels_x_2; // N_BINS x N_CHANNELS
    LocalTensor<int32_t> register_bins; // N_BINS
    LocalTensor<int32_t> register_bins_2; // N_BINS

    // Setup for gathering from i16
    LocalTensor<int32_t> indexes_0_31_x_32; // N_BINS x N_CHANNELS [0, 1, 2, 3, ... 31 | 0, 1, 2, ... | ... ]
    LocalTensor<int32_t> inverter_32_elem_x32_i16_idxs; // N_BINS x N_CHANNELS [31, 30, 29, ... 0 | 63, 62, 61, ... 32 | ... ] x sizeof(i16)
    LocalTensor<int32_t> swap_channel_bin_i16_idxs; // N_BINS x N_CHANNELS [0, 32, 64, ... | 1, 33, 65, ... | ... ] x sizeof(i16)
    LocalTensor<int32_t> swap_channel_token_i16_idxs; // N_TOKENS [0, 32, 64, ... ] x sizeof(i16)
    LocalTensor<int32_t> broadcast_i32_over_8_idxs; // 8 * N_C_PER_BLOCK [0, 0, 0, ... | 1, 1, 1  |  ] x sizeof(i32)

    LocalTensor<int32_t> l_pows_2_arr; // N_C_PER_BLOCK * 8 (max enc length) [2^7, 2^6, ... 2^0| 2^7, 2^6, ... ]
    LocalTensor<int16_t> l_pows_2_mask_arr; // N_C_PER_BLOCK * 2 * 8 (max enc length) [0b1..10, 0b1..10 | 0b1..100, 0b1..100, ... ]
    LocalTensor<int16_t> l_pows_2_mask_arr_ii; // N_C_PER_BLOCK * 2 * 8 (max enc length) [0b1..10, 0b1..100, 0b1..1000, ... | 0b1..10, ... ]
    LocalTensor<half> ones; // 256 (arbitrary) [1, 1, 1, ... ]
    LocalTensor<half> zeroes; // 256 (arbitrary) [0, 0, 0, ... ]
    LocalTensor<half> bin_floats; // N_B_MAX [0.0, 1.0, 2.0, ... ]

    LocalTensor<int32_t> p2s_32_arr; // 32 [2^0, 2^1, 2^2 ... ]
    LocalTensor<int32_t> duplicating_gather_i16_idxs; // 2 * N_C_PER_BLOCK [0, 0, 2, 2, 4, 4, ... ] - only care about every other elem
    LocalTensor<int32_t> reducing_gather_i16_idxs; // N_C_PER_BLOCK [2, 6, 10, ... ] - only care about every other elem

    // Class has no known need to support move or copy operations
    PacEncoder(const PacEncoder&) = delete;
    PacEncoder& operator=(const PacEncoder&) = delete;
    PacEncoder(PacEncoder&&) = delete;
    PacEncoder& operator=(PacEncoder&&) = delete;
};

__aicore__ inline PacEncoder::PacEncoder(
    GM_ADDR in_syms,
    GM_ADDR out_meta,
    GM_ADDR out_bytes,
    GM_ADDR out_lens,
    TPipe& _pipe,
    int32_t n_tokens,
    int32_t n_layers,
    int32_t n_channels,
    uint32_t n_bins,
    int32_t chunk_size,
    half scale_factor):
        pipe(_pipe),
        n_tokens(n_tokens),
        n_layers(n_layers),
        n_channels(n_channels),
        n_bins(n_bins),
        chunk_size(chunk_size),
        scale_factor(scale_factor) {

    half deq_scale = 1.0;
    SetDeqScale(deq_scale);

    n_tokens_per_layer = n_channels * n_tokens;

    g_syms.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(in_syms), n_layers * n_tokens * n_channels);
    gm_meta_data.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(out_meta), n_layers * n_channels * n_bins);
    g_out_bytes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(out_bytes), n_layers * n_channels * chunk_size);
    g_out_lens.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(out_lens), n_layers * n_channels);

    pipe.InitBuffer(symInQ, 1, N_T_PER_BATCH);
    pipe.InitBuffer(byteStreamOutQ, 1, N_T_PER_BATCH);
    pipe.InitBuffer(lensOutQ, 1, N_C_PER_BLOCK * sizeof(int32_t));
    pipe.InitBuffer(metaDataOutQ, 1, N_C_PER_BLOCK * N_B_MAX * sizeof(int16_t));
    pipe.InitBuffer(metaDataInQ, 1, N_C_PER_BLOCK * N_B_MAX * sizeof(int16_t));

    uint32_t calc_buf_sz_aligned = 0x1D000;
    pipe.InitBuffer(calcBuf, calc_buf_sz_aligned);
    pipe.InitBuffer(utilsCalcBuf, 0xA000);

    uint32_t utils_calc_buf_offset = 0;

    uint32_t register_bins_x_channels_count = N_B_MAX * N_C_PER_BLOCK;
    uint32_t register_bins_x_channels_sz = ceil_32(register_bins_x_channels_count * sizeof(int32_t));
    register_bins_x_channels = utilsCalcBuf.GetWithOffset<int32_t>(register_bins_x_channels_count, utils_calc_buf_offset);
    utils_calc_buf_offset += register_bins_x_channels_sz;

    register_bins_x_channels_x_2 = utilsCalcBuf.GetWithOffset<int32_t>(2 * register_bins_x_channels_count, utils_calc_buf_offset);
    utils_calc_buf_offset += 2 * register_bins_x_channels_sz;

    uint32_t register_bins_count = 2 * N_B_MAX;
    uint32_t register_bins_sz = ceil_32(register_bins_count * sizeof(int32_t));
    register_bins = utilsCalcBuf.GetWithOffset<int32_t>(register_bins_count, utils_calc_buf_offset);
    utils_calc_buf_offset += register_bins_sz;

    register_bins_2 = utilsCalcBuf.GetWithOffset<int32_t>(register_bins_count, utils_calc_buf_offset);
    utils_calc_buf_offset += register_bins_sz;

    // LocalTensor<int32_t> indexes_0_31_x_32; // N_BINS x N_CHANNELS [0, 1, 2, 3, ... 31 | 0, 1, 2, ... | ... ]
    uint32_t indexes_0_31_count = N_B_MAX * N_C_PER_BLOCK;
    uint32_t indexes_0_31_sz = ceil_32(indexes_0_31_count * sizeof(int32_t));
    indexes_0_31_x_32 = utilsCalcBuf.GetWithOffset<int32_t>(indexes_0_31_count, utils_calc_buf_offset);
    utils_calc_buf_offset += indexes_0_31_sz;
    CreateVecIndex(indexes_0_31_x_32, 0, N_B_MAX);
    Copy(indexes_0_31_x_32[32], indexes_0_31_x_32, 32, 1, {1, 1, 8, 8});
    Copy(indexes_0_31_x_32[64], indexes_0_31_x_32, 64, (N_B_MAX / 2) - 1, {1, 1, 8, 0});

    // LocalTensor<int32_t> inverter_32_elem_x32_i16_idxs; // N_BINS x N_CHANNELS sizeof(i16) x [31, 30, 29, ... 0 | 63, 62, 61, ... 32 | ... ]
    uint32_t inverter_32_elem_x32_i16_idxs_count = N_B_MAX * N_C_PER_BLOCK;
    uint32_t inverter_32_elem_x32_i16_idxs_sz = ceil_32(inverter_32_elem_x32_i16_idxs_count * sizeof(int32_t));
    inverter_32_elem_x32_i16_idxs = utilsCalcBuf.GetWithOffset<int32_t>(inverter_32_elem_x32_i16_idxs_count, utils_calc_buf_offset);
    utils_calc_buf_offset += inverter_32_elem_x32_i16_idxs_sz;
    Duplicate(register_bins_x_channels, 2 * (N_B_MAX - 1), N_B_MAX); // sizeof(i16) x [31, 31, 31 ... ]
    Adds(register_bins_x_channels[1 * N_B_MAX], register_bins_x_channels, 2 * 1 * N_B_MAX, 1 * N_B_MAX); // 0 - 1 set - // sizeof(i16) x [31, 31, 31 ... | 63, 63, ... ]
    Adds(register_bins_x_channels[2 * N_B_MAX], register_bins_x_channels, 2 * 2 * N_B_MAX, 2 * N_B_MAX); // 0 - 3 set
    Adds(register_bins_x_channels[4 * N_B_MAX], register_bins_x_channels, 2 * 4 * N_B_MAX, 4 * N_B_MAX); // 0 - 7 set
    Adds(register_bins_x_channels[8 * N_B_MAX], register_bins_x_channels, 2 * 8 * N_B_MAX, 8 * N_B_MAX); // 0 - 15 set
    Adds(register_bins_x_channels[16 * N_B_MAX], register_bins_x_channels, 2 * 16 * N_B_MAX, 16 * N_B_MAX); // 0 - 31 set
    // Sub indexes sizeof(i16) times (twice)
    Sub(inverter_32_elem_x32_i16_idxs, register_bins_x_channels, indexes_0_31_x_32, 64, N_B_MAX / 2, { 1, 1, 1, 8, 8, 8 });
    Sub(inverter_32_elem_x32_i16_idxs, inverter_32_elem_x32_i16_idxs, indexes_0_31_x_32, 64, N_B_MAX / 2, { 1, 1, 1, 8, 8, 8 }); // inverter_32_elem_x32_i16_idxs = [62, 60, 58, ... 0 | 126, 124, ...]

    // LocalTensor<int32_t> swap_channel_bin_i16_idxs; // N_BINS x N_CHANNELS sizeof(i16) x [0, 32, 64, ... | 1, 33, 65, ... | ... ]
    uint32_t swap_channel_bin_i16_idxs_count = N_B_MAX * N_C_PER_BLOCK;
    uint32_t swap_channel_bin_i16_idxs_sz = ceil_32(swap_channel_bin_i16_idxs_count * sizeof(int32_t));
    swap_channel_bin_i16_idxs = utilsCalcBuf.GetWithOffset<int32_t>(swap_channel_bin_i16_idxs_count, utils_calc_buf_offset);
    utils_calc_buf_offset += swap_channel_bin_i16_idxs_sz;
    Muls(swap_channel_bin_i16_idxs, indexes_0_31_x_32, 2 * N_B_MAX, N_B_MAX);
    Adds(swap_channel_bin_i16_idxs[1 * N_B_MAX], swap_channel_bin_i16_idxs, 2 * 1, 1 * N_B_MAX); // 0 - 1 set
    Adds(swap_channel_bin_i16_idxs[2 * N_B_MAX], swap_channel_bin_i16_idxs, 2 * 2, 2 * N_B_MAX); // 0 - 3 set
    Adds(swap_channel_bin_i16_idxs[4 * N_B_MAX], swap_channel_bin_i16_idxs, 2 * 4, 4 * N_B_MAX); // 0 - 7 set
    Adds(swap_channel_bin_i16_idxs[8 * N_B_MAX], swap_channel_bin_i16_idxs, 2 * 8, 8 * N_B_MAX); // 0 - 15 set
    Adds(swap_channel_bin_i16_idxs[16 * N_B_MAX], swap_channel_bin_i16_idxs, 2 * 16, 16 * N_B_MAX); // 0 - 31 set

    // LocalTensor<int32_t> swap_channel_token_i16_idxs; // N_TOKENS [0, 32, 64, ... ] x sizeof(i16)
    uint32_t swap_channel_token_i16_idxs_count = N_T_MAX;
    uint32_t swap_channel_token_i16_idxs_sz = ceil_32(swap_channel_token_i16_idxs_count * sizeof(int32_t));
    swap_channel_token_i16_idxs = utilsCalcBuf.GetWithOffset<int32_t>(swap_channel_token_i16_idxs_count, utils_calc_buf_offset);
    utils_calc_buf_offset += swap_channel_token_i16_idxs_sz;
    CreateVecIndex(swap_channel_token_i16_idxs, 0, N_T_MAX);
    Muls(swap_channel_token_i16_idxs, swap_channel_token_i16_idxs, static_cast<int32_t>(N_C_PER_BLOCK * sizeof(half)), N_T_MAX);

    // LocalTensor<int32_t> l_pows_2_arr; // N_C_PER_BLOCK * 8 (max enc length) [2^0, 2^1, ... 2^7 | 2^0, 2^1, ... ]
    uint32_t l_pows_2_arr_count = 8 * N_C_PER_BLOCK;
    uint32_t l_pows_2_arr_sz = ceil_32(l_pows_2_arr_count * sizeof(int32_t));
    l_pows_2_arr = utilsCalcBuf.GetWithOffset<int32_t>(l_pows_2_arr_sz, utils_calc_buf_offset);
    utils_calc_buf_offset += l_pows_2_arr_sz;
    for (auto b_ii = 0; b_ii < 8; ++b_ii) {
        int32_t p2 = (0x1 << b_ii) >> 1;
        l_pows_2_arr.SetValue(b_ii, p2);
    }
    Copy(l_pows_2_arr[1 * 8], l_pows_2_arr, 8, 1, {1, 1, 8, 8});
    Copy(l_pows_2_arr[2 * 8], l_pows_2_arr, 16, 1, {1, 1, 8, 8});
    Copy(l_pows_2_arr[4 * 8], l_pows_2_arr, 32, 1, {1, 1, 8, 8});
    Copy(l_pows_2_arr[8 * 8], l_pows_2_arr, 32, 1, {1, 1, 8, 8});
    Copy(l_pows_2_arr[12 * 8], l_pows_2_arr, 32, 1, {1, 1, 8, 8});
    Copy(l_pows_2_arr[16 * 8], l_pows_2_arr, 32, 1, {1, 1, 8, 8});
    Copy(l_pows_2_arr[20 * 8], l_pows_2_arr, 32, 1, {1, 1, 8, 8});
    Copy(l_pows_2_arr[24 * 8], l_pows_2_arr, 32, 1, {1, 1, 8, 8});
    Copy(l_pows_2_arr[28 * 8], l_pows_2_arr, 32, 1, {1, 1, 8, 8});

    // LocalTensor<int16_t> l_pows_2_mask_arr; // N_C_PER_BLOCK * 2 * 8 (max enc length * 2 because of an i32/i16) [0b1..11, 0b1..11 | 0b1..110, 0b1..110, ... ]
    uint32_t l_pows_2_mask_arr_count = 2 * 8 * N_C_PER_BLOCK;
    uint32_t l_pows_2_mask_arr_sz = ceil_32(l_pows_2_mask_arr_count * sizeof(int16_t));
    l_pows_2_mask_arr = utilsCalcBuf.GetWithOffset<int16_t>(l_pows_2_mask_arr_count, utils_calc_buf_offset);
    utils_calc_buf_offset += l_pows_2_mask_arr_sz;
    for (auto b_ii = 0; b_ii < 8; ++b_ii) {
        int16_t mask = (0xFFFFFFFF >> b_ii) << b_ii;
        // To Brcb to N_C_PER_BLOCK positions. Repeat the mask N_C_PER_BLOCK / 16 times given N_C_PER_BLOCK == 32 per
        // module assertion
        register_bins.ReinterpretCast<int16_t>().SetValue((4 * b_ii) + 0, mask);
        register_bins.ReinterpretCast<int16_t>().SetValue((4 * b_ii) + 1, mask);
        register_bins.ReinterpretCast<int16_t>().SetValue((4 * b_ii) + 2, mask);
        register_bins.ReinterpretCast<int16_t>().SetValue((4 * b_ii) + 3, mask);
    }
    Brcb(l_pows_2_mask_arr, register_bins.ReinterpretCast<int16_t>(), 8, {1, 8});

    // LocalTensor<int16_t> l_pows_2_mask_arr_ii; // N_C_PER_BLOCK * 2 * 8 (max enc length) [0b1..10, 0b1..100, 0b1..1000, ... | 0b1..10, ... ]
    uint32_t l_pows_2_mask_arr_ii_count = 2 * 8 * N_C_PER_BLOCK;
    uint32_t l_pows_2_mask_arr_ii_sz = ceil_32(l_pows_2_mask_arr_ii_count * sizeof(int16_t));
    l_pows_2_mask_arr_ii = utilsCalcBuf.GetWithOffset<int16_t>(l_pows_2_mask_arr_ii_count, utils_calc_buf_offset);
    utils_calc_buf_offset += l_pows_2_mask_arr_ii_sz;
    // Set the first 32 B, then copy into the rest. Target is 0b111111111, 0b111111110, 0b111111100 ...
    int16_t base_m = 0xffff;
    for (auto b_ii = 0; b_ii < 8; ++b_ii) {
        l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>().SetValue(b_ii + 0 * 8, (base_m >> b_ii) << b_ii);
        l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>().SetValue(b_ii + 1 * 8, (base_m >> b_ii) << b_ii);
    }

    Copy(l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>()[2 * 8], l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>(), 16, 1, {1, 1, 8, 8});
    Copy(l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>()[4 * 8], l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>(), 32, 1, {1, 1, 8, 8});
    Copy(l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>()[8 * 8], l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>(), 64, 1, {1, 1, 8, 8});
    Copy(l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>()[8 * 8], l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>(), 64, 1, {1, 1, 8, 8});
    Copy(l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>()[16 * 8], l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>(), 64, 1, {1, 1, 8, 8});
    Copy(l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>()[24 * 8], l_pows_2_mask_arr_ii.ReinterpretCast<int32_t>(), 64, 1, {1, 1, 8, 8});

    uint32_t zeroes_count = 256;
    uint32_t zeroes_sz = ceil_32(zeroes_count * sizeof(half));
    zeroes = utilsCalcBuf.GetWithOffset<half>(zeroes_count, utils_calc_buf_offset);
    utils_calc_buf_offset += zeroes_sz;
    half zero = 0.0;
    Duplicate(zeroes, zero, zeroes_count);

    uint32_t ones_count = 256;
    uint32_t ones_sz = ceil_32(ones_count * sizeof(half));
    ones = utilsCalcBuf.GetWithOffset<half>(ones_count, utils_calc_buf_offset);
    utils_calc_buf_offset += ones_sz;
    half one = 1.0;
    Duplicate(ones, one, ones_count);

    uint32_t bin_floats_count = N_B_MAX;
    uint32_t bin_floats_sz = ceil_32(bin_floats_count * sizeof(half));
    bin_floats = utilsCalcBuf.GetWithOffset<half>(bin_floats_count, utils_calc_buf_offset);
    utils_calc_buf_offset += bin_floats_sz;
    Cast(bin_floats, indexes_0_31_x_32, RoundMode::CAST_NONE, N_B_MAX);

    uint32_t broadcast_i32_over_8_idxs_count = 8 * N_C_PER_BLOCK;
    uint32_t broadcast_i32_over_8_idxs_sz = ceil_32(broadcast_i32_over_8_idxs_count * sizeof(int32_t));
    broadcast_i32_over_8_idxs = utilsCalcBuf.GetWithOffset<int32_t>(broadcast_i32_over_8_idxs_count, utils_calc_buf_offset);
    utils_calc_buf_offset += broadcast_i32_over_8_idxs_sz;
    Copy(register_bins, indexes_0_31_x_32, 32, 1, {1, 1, 8, 8});
    Muls(register_bins, register_bins, 4, N_C_PER_BLOCK);
    Brcb(broadcast_i32_over_8_idxs, register_bins, N_C_PER_BLOCK, {1, 8});

    uint32_t p2s_32_arr_count = 32;
    uint32_t p2s_32_arr_sz = ceil_32(p2s_32_arr_count * sizeof(int32_t));
    p2s_32_arr = utilsCalcBuf.GetWithOffset<int32_t>(p2s_32_arr_count, utils_calc_buf_offset);
    utils_calc_buf_offset += p2s_32_arr_sz;
    uint32_t base = 1;
    for (auto b_ii = 0; b_ii < 32; ++b_ii) {
        p2s_32_arr.SetValue(b_ii, base << (b_ii - 1));
    }

    uint32_t duplicating_gather_i16_idxs_count = 2 * N_C_PER_BLOCK;
    uint32_t duplicating_gather_i16_idxs_sz = ceil_32(duplicating_gather_i16_idxs_count * sizeof(int32_t));
    duplicating_gather_i16_idxs = utilsCalcBuf.GetWithOffset<int32_t>(duplicating_gather_i16_idxs_count, utils_calc_buf_offset);
    utils_calc_buf_offset += duplicating_gather_i16_idxs_sz;
    CreateVecIndex(duplicating_gather_i16_idxs, -1, 2 * N_C_PER_BLOCK); // -1, 0, 1, 2, ...
    uint64_t everyother_mask[1] = {0x5555555555555555}; // Every other element for 64
    Adds(duplicating_gather_i16_idxs, duplicating_gather_i16_idxs, 1, everyother_mask, 1, {1, 1, 8, 8}); // 0, 0, 2, 2, 4, 4

    uint32_t reducing_gather_i16_idxs_count = N_C_PER_BLOCK;
    uint32_t reducing_gather_i16_idxs_sz = ceil_32(reducing_gather_i16_idxs_count * sizeof(int32_t));
    reducing_gather_i16_idxs = utilsCalcBuf.GetWithOffset<int32_t>(reducing_gather_i16_idxs_count, utils_calc_buf_offset);
    CreateVecIndex(reducing_gather_i16_idxs, 0, N_C_PER_BLOCK); // 0, 1, 2, 3 ...
    Muls(reducing_gather_i16_idxs, reducing_gather_i16_idxs, static_cast<int32_t>(2 * sizeof(uint16_t)), N_C_PER_BLOCK);
    Adds(reducing_gather_i16_idxs, reducing_gather_i16_idxs, static_cast<int32_t>(sizeof(uint16_t)), N_C_PER_BLOCK);
}

__aicore__ inline void PacEncoder::meta_data_calc(int layer_id, int channel_start_id) {
    uint32_t n_T_chunks = n_tokens / N_T_MAX;
    n_T_chunks = n_tokens % N_T_MAX == 0 ? n_T_chunks : n_T_chunks + 1;

    uint32_t calc_buf_offset = 0;
    uint32_t cast_input_count = N_T_MAX * N_C_PER_BLOCK;
    uint32_t cast_input_sz = ceil_32(cast_input_count * sizeof(half));
    LocalTensor<half> cast_input = calcBuf.GetWithOffset<half>(cast_input_count, calc_buf_offset);
    calc_buf_offset += cast_input_sz;

    uint32_t swapped_input_count = N_T_MAX * N_C_PER_BLOCK;
    uint32_t swapped_input_sz = ceil_32(swapped_input_count * sizeof(half));
    LocalTensor<half> swapped_input = calcBuf.GetWithOffset<half>(swapped_input_count, calc_buf_offset);
    calc_buf_offset += swapped_input_sz;

    uint32_t bin_cmp_mask_sz = ceil_32(N_T_PER_BATCH / (8 * sizeof(int8_t)));
    LocalTensor<uint8_t> bin_cmp_mask = calcBuf.GetWithOffset<uint8_t>(bin_cmp_mask_sz, calc_buf_offset);
    calc_buf_offset += bin_cmp_mask_sz;

    uint32_t bin_cmp_as_elem_count = N_T_PER_BATCH;
    uint32_t bin_cmp_as_elem_sz = ceil_32(bin_cmp_as_elem_count  * sizeof(half));
    LocalTensor<half> bin_cmp_as_elem = calcBuf.GetWithOffset<half>(bin_cmp_as_elem_count, calc_buf_offset);
    calc_buf_offset += bin_cmp_as_elem_sz;

    uint32_t count_count = N_B_MAX * N_C_PER_BLOCK;
    uint32_t count_sz = ceil_32(count_count * sizeof(half));
    LocalTensor<half> count = calcBuf.GetWithOffset<half>(count_count, calc_buf_offset);
    calc_buf_offset += count_sz;

    uint32_t swapped_count_count = N_B_MAX * N_C_PER_BLOCK;
    uint32_t swapped_count_sz = ceil_32(swapped_count_count * sizeof(half));
    LocalTensor<half> swapped_count_tmp = calcBuf.GetWithOffset<half>(swapped_count_count, calc_buf_offset);
    calc_buf_offset += swapped_count_sz;
    LocalTensor<half> swapped_count = calcBuf.GetWithOffset<half>(swapped_count_count, calc_buf_offset);
    calc_buf_offset += swapped_count_sz;
    half zero = 0.;
    Duplicate(swapped_count, zero, N_B_MAX * N_C_PER_BLOCK);

    uint32_t intermediate_count_count = 2 * N_C_PER_BLOCK;
    uint32_t intermediate_count_sz = ceil_32(intermediate_count_count * sizeof(half));
    LocalTensor<half> intermediate_count = calcBuf.GetWithOffset<half>(intermediate_count_count, calc_buf_offset);
    calc_buf_offset += intermediate_count_sz;

    // Tally all tokens in layer and chunk of channels startign from channel start index
    for (uint32_t T_chunk_idx = 0; T_chunk_idx < n_T_chunks; ++T_chunk_idx) {

        int32_t n_tokens_in_chunk = N_T_MAX * (T_chunk_idx + 1) > n_tokens ? n_tokens - (N_T_MAX * T_chunk_idx) : N_T_MAX;

        LocalTensor<uint8_t> l_syms = symInQ.AllocTensor<uint8_t>();
        auto sym_offset =
            n_tokens_per_layer * layer_id +
            n_channels * N_T_MAX * T_chunk_idx +
            channel_start_id;

        uint16_t src_stride = N_DBs_PER_BLOCK * ((n_channels / N_C_PER_BLOCK) - 1);
        DataCopyParams repeatParams = {static_cast<uint16_t>(n_tokens_in_chunk), N_DBs_PER_BLOCK, src_stride, 0};
        DataCopy(l_syms, g_syms[sym_offset], repeatParams);

        symInQ.EnQue(l_syms);
        l_syms = symInQ.DeQue<uint8_t>();

        Cast(cast_input, l_syms, RoundMode::CAST_NONE, n_tokens_in_chunk * N_C_PER_BLOCK);
        symInQ.FreeTensor(l_syms);

        // Defensive, ensure any dummy tokens are intialized to invalid token values
        half inval = 999.;
        Duplicate(cast_input[n_tokens_in_chunk * N_C_PER_BLOCK], inval, (N_T_MAX - n_tokens_in_chunk) * N_C_PER_BLOCK);

        for (int32_t c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
            Gather(swapped_input[N_T_MAX * c_ii], cast_input, swap_channel_token_i16_idxs.ReinterpretCast<uint32_t>(), sizeof(half) * c_ii, N_T_MAX);
        }

        for (int b_ii = 0; b_ii < n_bins; ++b_ii) {
            CompareScalar(bin_cmp_mask, swapped_input, bin_floats(b_ii), CMPMODE::EQ, N_T_PER_BATCH);

            BinaryRepeatParams select_repeat_params = BinaryRepeatParams(
                1, // dstBlkStrideIn
                1, // src0BlkStrideIn /
                1, // src1BlkStrideIn
                8, // dstRepStrideIn
                0, // src0RepStrideIn
                0 // src1RepStrideIn
            );

            Select(bin_cmp_as_elem, bin_cmp_mask, ones, zeroes, SELMODE::VSEL_TENSOR_TENSOR_MODE, 128, N_T_PER_BATCH / 128, select_repeat_params);
            RepeatReduceSum(intermediate_count, bin_cmp_as_elem, N_T_PER_BATCH / 128, 128, 0, 1, 1, 8);
            PairReduceSum(count[b_ii * N_B_MAX], intermediate_count, 1, 64, 1, 1, 8);
        }
        Gather(swapped_count_tmp, count, swap_channel_bin_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_B_MAX * N_C_PER_BLOCK);
        Add(swapped_count, swapped_count, swapped_count_tmp, N_B_MAX * N_C_PER_BLOCK);
    }
    Muls(swapped_count, swapped_count, scale_factor, N_B_MAX * N_C_PER_BLOCK);

    // The above may include fractions - anything in the 0-1 range must end up encodable this is achieved with a
    // a ceil rounding.
    Cast(swapped_count.ReinterpretCast<int16_t>(), swapped_count, RoundMode::CAST_CEIL, N_C_PER_BLOCK * N_B_MAX);
    Cast(swapped_count, swapped_count.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, N_C_PER_BLOCK * N_B_MAX);

    // --------
    // Phase: Sort bins (pre-requisite for aligning bin boundaries)
    // --------
    // Needs 8B per elem for storing val + idx + dummy as input to Sort32
    uint32_t sorted_count_sz = ceil_32(N_B_MAX * N_C_PER_BLOCK * 8);
    LocalTensor<half> sorted_count = calcBuf.GetWithOffset<half>(sorted_count_sz / sizeof(half), calc_buf_offset);
    calc_buf_offset += sorted_count_sz;

    for (uint32_t b_ii = 0; b_ii < N_B_MAX; ++b_ii) {
        Sort32(sorted_count[b_ii * N_B_MAX * (8 / sizeof(half))], swapped_count[b_ii * N_B_MAX], indexes_0_31_x_32.ReinterpretCast<uint32_t>(), N_B_MAX / 32);
    }

    uint32_t sorted_count_idx_sz = ceil_32(N_C_PER_BLOCK * N_B_MAX * sizeof(uint32_t));
    LocalTensor<int32_t> sorted_count_idx = calcBuf.GetWithOffset<int32_t>(N_C_PER_BLOCK * N_B_MAX, calc_buf_offset);
    calc_buf_offset += sorted_count_idx_sz;

    uint64_t _rsvd = 0;
    auto elems_per_256 = 256 / 8;
    auto repeats = (N_B_MAX * N_C_PER_BLOCK) / elems_per_256;
    GatherMaskParams gmp = {
        1, // src0BlockStride. 1 - Continuous data
        static_cast<uint8_t>(repeats),
        8, // src0RepeatStride - Continuous Data
        0 // src1RepeatStride - not used
    };
    GatherMask(sorted_count_idx, sorted_count.ReinterpretCast<int32_t>(), 2, false, 0, gmp, _rsvd);

    _rsvd = 0;
    gmp = {
        1, // src0BlockStride. 1 - Continuous data
        static_cast<uint8_t>(repeats),
        8, // src0RepeatStride - Continuous Data
        0 // src1RepeatStride - not used
    };
    GatherMask(swapped_count, sorted_count, 3, false, 0, gmp, _rsvd);

    static constexpr CumSumConfig cum_sum_cfg = {true, true, false};
    CumSumInfo cum_sum_info = {N_C_PER_BLOCK, N_B_MAX};
    CumSum<half, cum_sum_cfg>(swapped_count, swapped_count, swapped_count, cum_sum_info);

    // Ensure the full number line is used
    auto swapped_count_i16 = swapped_count.ReinterpretCast<int16_t>();
    Cast(swapped_count_i16, swapped_count, RoundMode::CAST_CEIL, N_C_PER_BLOCK * N_B_MAX);
    for (auto c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        int16_t correction = 256 - swapped_count_i16.GetValue(N_B_MAX * (c_ii + 1) - 1);
        Adds(swapped_count_i16[N_B_MAX * c_ii], swapped_count_i16[N_B_MAX * c_ii], correction, N_B_MAX);
    }
    Cast(swapped_count, swapped_count_i16, RoundMode::CAST_NONE, N_C_PER_BLOCK * N_B_MAX);

    Gather(swapped_count, swapped_count, inverter_32_elem_x32_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_B_MAX);
    half min_1 = -1.0;
    Muls(swapped_count, swapped_count, min_1, N_C_PER_BLOCK * N_B_MAX);
    half correct = 256.0;
    Adds(swapped_count, swapped_count, correct, N_C_PER_BLOCK * N_B_MAX);

    Cast(swapped_count_i16, swapped_count, RoundMode::CAST_RINT, N_C_PER_BLOCK * N_B_MAX);

    // --------
    // Phase: Align boundaries to 2^n boundaries - pass 1 (least common to most common)
    // --------
    uint32_t low_o_sz = ceil_32(N_C_PER_BLOCK * N_B_MAX * sizeof(int16_t));
    LocalTensor<int16_t> low_o = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK * N_B_MAX, calc_buf_offset);
    calc_buf_offset += low_o_sz;

    LocalTensor<int16_t> low_o_ii = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK * N_B_MAX, calc_buf_offset);
    calc_buf_offset += low_o_sz;

    uint32_t low_sz = ceil_32(N_C_PER_BLOCK  * sizeof(int16_t));
    LocalTensor<int16_t> low = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK , calc_buf_offset);
    calc_buf_offset += low_sz;

    uint32_t high_sz = ceil_32(N_C_PER_BLOCK * sizeof(int16_t));
    LocalTensor<int16_t> high = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += high_sz;

    uint32_t low_bcast_sz = ceil_32(8 * N_C_PER_BLOCK * sizeof(int32_t));
    LocalTensor<int32_t> low_bcast = calcBuf.GetWithOffset<int32_t>(8 * N_C_PER_BLOCK , calc_buf_offset);
    calc_buf_offset += low_bcast_sz;

    uint32_t high_bcast_sz = ceil_32(8 * N_C_PER_BLOCK * sizeof(int16_t));
    LocalTensor<int16_t> high_bcast = calcBuf.GetWithOffset<int16_t>(8 * N_C_PER_BLOCK , calc_buf_offset);
    calc_buf_offset += high_bcast_sz;

    uint32_t c_block_sz = ceil_32(N_C_PER_BLOCK * sizeof(int16_t));
    LocalTensor<int8_t> alignment_mask = calcBuf.GetWithOffset<int8_t>(N_C_PER_BLOCK , calc_buf_offset);
    calc_buf_offset += c_block_sz;
    LocalTensor<int16_t> alignment_mask_i16 = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK , calc_buf_offset);
    calc_buf_offset += c_block_sz;
    LocalTensor<int8_t> exceeds_high_mask = calcBuf.GetWithOffset<int8_t>(N_C_PER_BLOCK , calc_buf_offset);
    calc_buf_offset += c_block_sz;
    LocalTensor<int16_t> exceeds_high_mask_i16 = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK , calc_buf_offset);
    calc_buf_offset += c_block_sz;

    Gather(low_o, swapped_count_i16, swap_channel_bin_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);

    for (auto bin_ii = 0; bin_ii < (N_B_MAX - 1); ++bin_ii) {
        Copy(low, low_o[N_C_PER_BLOCK * bin_ii], N_C_PER_BLOCK, 1, {1, 1, 8, 8});
        Copy(high, low_o[N_C_PER_BLOCK * (bin_ii + 1)], N_C_PER_BLOCK, 1, {1, 1, 8, 8});

        Brcb(register_bins_x_channels, indexes_0_31_x_32, N_C_PER_BLOCK / 8, {1, 8});

        Muls(register_bins_x_channels, register_bins_x_channels, 2, N_C_PER_BLOCK * (32 / sizeof(int32_t)));
        Gather(high_bcast, high, register_bins_x_channels.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * (32 / sizeof(int32_t)));

        // Broadcast Low over 8 positions. One for each possible power 2 alignment. It will be involved in a compare so,
        // owing to type cosntraints, needs to be i32. (half would work for the compare but we also want to do bitwise
        // logical operations)
        Cast(low.ReinterpretCast<half>(), low, RoundMode::CAST_NONE, N_C_PER_BLOCK);
        Cast(register_bins, low.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);
        Muls(register_bins_x_channels, register_bins_x_channels, 2, N_C_PER_BLOCK * (32 / sizeof(int32_t)));
        Gather(low_bcast, register_bins, register_bins_x_channels.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * (32 / sizeof(int32_t)));

        // Identify trailing zero's of low - while tight packing bins this bounds the bin width that can be allocated
        And(register_bins_x_channels.ReinterpretCast<int16_t>(), low_bcast.ReinterpretCast<int16_t>(), l_pows_2_mask_arr_ii, 2 * 8 * N_C_PER_BLOCK);
        Compare(alignment_mask, low_bcast, register_bins_x_channels, CMPMODE::EQ, N_C_PER_BLOCK * 8);

        Cast(alignment_mask_i16.ReinterpretCast<half>(), alignment_mask.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, N_C_PER_BLOCK);
        Cast(alignment_mask_i16, alignment_mask_i16.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);
        ShiftLeft(alignment_mask_i16, alignment_mask_i16, static_cast<int16_t>(1), N_C_PER_BLOCK);
        Adds(alignment_mask_i16, alignment_mask_i16, static_cast<int16_t>(1), N_C_PER_BLOCK);

        // Propose a series of new highs by adding 2^n to low
        auto proposed_high = low_bcast; // Buffer re-use
        Add(proposed_high, low_bcast, l_pows_2_arr, 8 * N_C_PER_BLOCK);

        Cast(proposed_high.ReinterpretCast<half>(), proposed_high, RoundMode::CAST_NONE, N_C_PER_BLOCK * 8);
        Cast(high_bcast.ReinterpretCast<half>(), high_bcast, RoundMode::CAST_NONE, N_C_PER_BLOCK * 8);

        // Pick the proposed bin that doesn't exceed high and satisfies alignment requirements
        Compare(exceeds_high_mask, proposed_high.ReinterpretCast<half>(), high_bcast.ReinterpretCast<half>(), CMPMODE::LE, N_C_PER_BLOCK * 8);

        Cast(exceeds_high_mask_i16.ReinterpretCast<half>(), exceeds_high_mask.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, N_C_PER_BLOCK);
        Cast(exceeds_high_mask_i16, exceeds_high_mask_i16.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);

        And(exceeds_high_mask_i16, exceeds_high_mask_i16, alignment_mask_i16, N_C_PER_BLOCK);
        Adds(exceeds_high_mask_i16, exceeds_high_mask_i16, static_cast<int16_t>(0x0001), N_C_PER_BLOCK);
        ShiftRight(exceeds_high_mask_i16, exceeds_high_mask_i16, static_cast<int16_t>(1), N_C_PER_BLOCK);

        Cast(exceeds_high_mask_i16.ReinterpretCast<half>(), exceeds_high_mask_i16, RoundMode::CAST_NONE, N_C_PER_BLOCK);
        Cast(exceeds_high_mask.ReinterpretCast<uint8_t>(), exceeds_high_mask_i16.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);

        // Update the bins and move on to next low/high pair
        uint64_t _rsvd = 0;
        auto repeats = 1;
        gmp = {
            1, // src0BlockStride. 1 - Continuous data
            static_cast<uint8_t>(repeats),
            8, // src0RepeatStride - Continuous Data
            0 // src1RepeatStride - not used
        };
        auto new_high = low; // Buffer reuse
        GatherMask(new_high.ReinterpretCast<half>(), proposed_high.ReinterpretCast<half>(), exceeds_high_mask.ReinterpretCast<uint16_t>(), true, 256, gmp, _rsvd);

        Cast(low_o[N_C_PER_BLOCK * (bin_ii + 1)], new_high.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);
    }
    Gather(low_o_ii, low_o, swap_channel_bin_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);
    Gather(low_o_ii, low_o_ii, inverter_32_elem_x32_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_B_MAX);

    int16_t min_1_i16 = -1;
    Muls(low_o_ii, low_o_ii, min_1_i16, N_C_PER_BLOCK * N_B_MAX);
    int16_t correct_i16 = 256;
    Adds(low_o_ii, low_o_ii, correct_i16, N_C_PER_BLOCK * N_B_MAX);

    Gather(low_o, low_o_ii, swap_channel_bin_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);

    // --------
    // Phase: Align boundaries to 2^n boundaries - pass 2 (most common to least common)
    // --------
    for (auto bin_ii = 0; bin_ii < N_B_MAX; ++bin_ii) {
        if (bin_ii == 0) {
            Duplicate(low, static_cast<int16_t>(0), N_C_PER_BLOCK);
        } else {
            Copy(low, low_o[N_C_PER_BLOCK * (bin_ii - 1)], N_C_PER_BLOCK, 1, {1, 1, 8, 8});
        }

        if (bin_ii == (N_B_MAX - 1)) {
            Duplicate(high, static_cast<int16_t>(256), N_C_PER_BLOCK);
        } else {
            Copy(high, low_o[N_C_PER_BLOCK * bin_ii], N_C_PER_BLOCK, 1, {1, 1, 8, 8});
        }

        Brcb(register_bins_x_channels, indexes_0_31_x_32, N_C_PER_BLOCK / 8, {1, 8});

        Muls(register_bins_x_channels, register_bins_x_channels, 2, N_C_PER_BLOCK * (32 / sizeof(int32_t)));
        Gather(high_bcast, high, register_bins_x_channels.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * (32 / sizeof(int32_t)));

        Muls(register_bins_x_channels, register_bins_x_channels, 2, N_C_PER_BLOCK * (32 / sizeof(int32_t)));
        Cast(low.ReinterpretCast<half>(), low, RoundMode::CAST_NONE, N_C_PER_BLOCK);
        Cast(register_bins, low.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);
        Gather(low_bcast, register_bins, register_bins_x_channels.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * (32 / sizeof(int32_t)));

        // Identify trailing zero's of low - while tight packing bins this bounds the bin width that can be allocated
        And(register_bins_x_channels.ReinterpretCast<int16_t>(), low_bcast.ReinterpretCast<int16_t>(), l_pows_2_mask_arr_ii, 2 * 8 * N_C_PER_BLOCK);
        Compare(alignment_mask, low_bcast, register_bins_x_channels, CMPMODE::EQ, N_C_PER_BLOCK * 8);

        Cast(alignment_mask_i16.ReinterpretCast<half>(), alignment_mask.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, N_C_PER_BLOCK);
        Cast(alignment_mask_i16, alignment_mask_i16.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);
        ShiftLeft(alignment_mask_i16, alignment_mask_i16, static_cast<int16_t>(1), N_C_PER_BLOCK);
        Adds(alignment_mask_i16, alignment_mask_i16, static_cast<int16_t>(1), N_C_PER_BLOCK);

        // Propose a series of new highs by adding 2^n to low
        auto proposed_high = low_bcast; // Buffer re-use
        Add(proposed_high, proposed_high, l_pows_2_arr, 8 * N_C_PER_BLOCK);

        Cast(proposed_high.ReinterpretCast<half>(), low_bcast, RoundMode::CAST_NONE, N_C_PER_BLOCK * 8);
        Cast(high_bcast.ReinterpretCast<half>(), high_bcast, RoundMode::CAST_NONE, N_C_PER_BLOCK * 8);

        // Pick the proposed bin that doesn't exceed high and satisfies alignment requirements
        Compare(exceeds_high_mask, proposed_high.ReinterpretCast<half>(), high_bcast.ReinterpretCast<half>(), CMPMODE::LE, N_C_PER_BLOCK * 8);

        Cast(exceeds_high_mask_i16.ReinterpretCast<half>(), exceeds_high_mask.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, N_C_PER_BLOCK);
        Cast(exceeds_high_mask_i16, exceeds_high_mask_i16.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);

        And(exceeds_high_mask_i16, exceeds_high_mask_i16, alignment_mask_i16, N_C_PER_BLOCK);
        Adds(exceeds_high_mask_i16, exceeds_high_mask_i16, static_cast<int16_t>(0x0001), N_C_PER_BLOCK);
        ShiftRight(exceeds_high_mask_i16, exceeds_high_mask_i16, static_cast<int16_t>(1), N_C_PER_BLOCK);

        Cast(exceeds_high_mask_i16.ReinterpretCast<half>(), exceeds_high_mask_i16, RoundMode::CAST_NONE, N_C_PER_BLOCK);
        Cast(exceeds_high_mask.ReinterpretCast<uint8_t>(), exceeds_high_mask_i16.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);

        // Update the bins and move on to next low/high pair
        uint64_t _rsvd = 0;
        auto repeats = 1;
        gmp = {
            1, // src0BlockStride. 1 - Continuous data
            static_cast<uint8_t>(repeats),
            8, // src0RepeatStride - Continuous Data
            0 // src1RepeatStride - not used
        };
        auto new_high = low; // Buffer reuse
        GatherMask(new_high.ReinterpretCast<half>(), proposed_high.ReinterpretCast<half>(), exceeds_high_mask.ReinterpretCast<uint16_t>(), true, 256, gmp, _rsvd);
        Cast(low_o[N_C_PER_BLOCK * (bin_ii)], new_high.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);
    }

    // --------
    // Phase: Determine minimum encode lengths given aligned bins
    // --------
    uint32_t lens_sz = ceil_32(N_C_PER_BLOCK * N_B_MAX * sizeof(int16_t));
    LocalTensor<int16_t> lens = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK * N_B_MAX , calc_buf_offset);
    calc_buf_offset += lens_sz;

    const int max_symbol = N_B_MAX - 1;
    for (auto c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        for (uint8_t mock_sym = 0; mock_sym < N_B_MAX; mock_sym++) {
            uint16_t c_low = mock_sym == 0 ? 0 : low_o(c_ii + N_B_MAX * (mock_sym - 1));
            uint16_t c_high = low_o(c_ii + N_B_MAX * mock_sym);

            if (c_low == static_cast<uint16_t>(256)) {
                // All remaining symbols are un-encodable
                for (; mock_sym < N_B_MAX; mock_sym++)
                {
                    lens.SetValue(c_ii + N_B_MAX * mock_sym, 9);
                    low_o.SetValue(c_ii + N_B_MAX * mock_sym, static_cast<int16_t>(512)); // Break sorting ties
                }
                continue;
            }

            c_low <<= 8;
            c_high <<= 8;

            uint16_t next_pos = AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(c_high & (~c_low))) - 47;
            uint8_t len_base = static_cast<uint8_t>(next_pos);

            uint16_t base_encode = (c_low >> (16 - len_base)) << (16 - len_base);
            uint16_t lower = base_encode + ((1 << (16 - len_base))) - 1;
            uint8_t len_break_lower = AscendC::ScalarCountLeadingZero(static_cast<uint64_t>(lower & ~c_low)) - 47;

            if (mock_sym == 0 || len_break_lower >= lens.GetValue(c_ii + N_B_MAX * (mock_sym - 1))) {
                len_break_lower -= 1;
            }

            lens.SetValue(c_ii + N_B_MAX * mock_sym, len_break_lower);
        }
    }

    Gather(low_o_ii, low_o, swap_channel_bin_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);
    Gather(register_bins_x_channels.ReinterpretCast<int16_t>(), lens, swap_channel_bin_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);

    // --------
    // Phase: Prepare for encoding by reverting the sorting so aligned bins, and encoded length are in symbol order
    // rather than frequency order
    // --------
    auto resorted_counts = register_bins_x_channels_x_2.ReinterpretCast<half>();

    for(auto c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        Cast(register_bins.ReinterpretCast<half>(), sorted_count_idx[c_ii * N_B_MAX], RoundMode::CAST_RINT, N_C_PER_BLOCK);
        Sort32(resorted_counts[N_B_MAX * c_ii * (8 / sizeof(half))], register_bins.ReinterpretCast<half>(), indexes_0_31_x_32.ReinterpretCast<uint32_t>(), N_B_MAX / 32);
    }

    _rsvd = 0;
    elems_per_256 = 256 / 8;
    repeats = (N_B_MAX * N_C_PER_BLOCK) / elems_per_256;
    gmp = {
        1, // src0BlockStride. 1 - Continuous data
        static_cast<uint8_t>(1),
        8, // src0RepeatStride - Continuous Data
        0 // src1RepeatStride - not used
    };
    auto tmp_rsvd = _rsvd;
    for (auto rep_ii = 0; rep_ii < repeats; ++rep_ii) {
        GatherMask(sorted_count_idx[elems_per_256 * rep_ii], resorted_counts.ReinterpretCast<int32_t>()[2 * elems_per_256 * rep_ii], 2, false, 0, gmp, tmp_rsvd);
        _rsvd += tmp_rsvd;
    }

    for (auto c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        Adds(sorted_count_idx[c_ii * N_C_PER_BLOCK], sorted_count_idx[c_ii * N_C_PER_BLOCK], 32 * c_ii, N_C_PER_BLOCK);
    }
    Muls(sorted_count_idx, sorted_count_idx, 2, N_C_PER_BLOCK * N_C_PER_BLOCK);
    Gather(lens, register_bins_x_channels.ReinterpretCast<int16_t>(), sorted_count_idx.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);
    Gather(lens, lens, inverter_32_elem_x32_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);

    Gather(low_o, low_o_ii, sorted_count_idx.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);
    Gather(low_o, low_o, inverter_32_elem_x32_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);

    for (auto c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        int16_t next = 0;
        auto last = 0;
        for (auto b_ii = 0; b_ii < N_B_MAX; ++b_ii) {
            next = last;
            last = low_o_ii.GetValue(c_ii * N_B_MAX + b_ii);
            low_o_ii.SetValue(c_ii * N_B_MAX + b_ii, next);
        }
    }

    Gather(low_o, low_o_ii, sorted_count_idx.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);
    Gather(low_o, low_o, inverter_32_elem_x32_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK * N_C_PER_BLOCK);
    auto enc = low_o;

    auto meta_out = metaDataOutQ.AllocTensor<int16_t>();
    ShiftLeft(meta_out, enc, static_cast<int16_t>(8), N_C_PER_BLOCK * N_B_MAX);
    Or(meta_out, meta_out, lens, N_C_PER_BLOCK * N_B_MAX);

    metaDataOutQ.EnQue(meta_out);
    meta_out = metaDataOutQ.DeQue<int16_t>();
    DataCopy(gm_meta_data[layer_id * n_channels * N_B_MAX + channel_start_id * N_B_MAX], meta_out.ReinterpretCast<uint16_t>(), N_C_PER_BLOCK * N_B_MAX);
    metaDataOutQ.FreeTensor(meta_out);
}


__aicore__ inline void PacEncoder::encode(int layer_id, int channel_start_id) {
    // --------
    // Phase: Copy IN
    // --------
    LocalTensor<uint8_t> l_syms = symInQ.AllocTensor<uint8_t>();
    auto sym_offset = n_tokens_per_layer * layer_id + channel_start_id;
    uint16_t src_stride = N_DBs_PER_BLOCK * ((n_channels / N_C_PER_BLOCK) - 1);
    DataCopyParams repeatParams = {static_cast<uint16_t>(n_tokens), N_DBs_PER_BLOCK, src_stride, 0};
    DataCopy(l_syms, g_syms[sym_offset], repeatParams);

    symInQ.EnQue(l_syms);
    l_syms = symInQ.DeQue<uint8_t>();

    uint32_t calc_buf_offset = 0;

    uint32_t cast_input_count = N_T_MAX * N_C_PER_BLOCK;
    uint32_t cast_input_sz = ceil_32(cast_input_count * sizeof(half));
    LocalTensor<half> cast_input = calcBuf.GetWithOffset<half>(cast_input_count, calc_buf_offset);
    calc_buf_offset += cast_input_sz;
    Cast(cast_input, l_syms, RoundMode::CAST_NONE, n_tokens * N_C_PER_BLOCK);
    symInQ.FreeTensor(l_syms);

    // Defensive, ensure any dummy tokens are intialized to invalid token values
    half inval = 999.;
    Duplicate(cast_input[n_tokens * N_C_PER_BLOCK], inval, (N_T_MAX - n_tokens) * N_C_PER_BLOCK);

    uint32_t swapped_input_count = N_T_MAX * N_C_PER_BLOCK;
    uint32_t swapped_input_sz = ceil_32(swapped_input_count * sizeof(half));
    LocalTensor<half> swapped_input = calcBuf.GetWithOffset<half>(swapped_input_count, calc_buf_offset);
    calc_buf_offset += swapped_input_sz;

    for (int32_t c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        Gather(swapped_input[N_T_MAX * c_ii], cast_input, swap_channel_token_i16_idxs.ReinterpretCast<uint32_t>(), sizeof(half) * c_ii, N_T_MAX);
    }

    auto meta_info = metaDataInQ.AllocTensor<uint16_t>();
    DataCopy(meta_info, gm_meta_data[(layer_id * n_channels + channel_start_id) * N_B_MAX], N_C_PER_BLOCK * N_B_MAX);
    metaDataInQ.EnQue(meta_info);
    meta_info = metaDataInQ.DeQue<uint16_t>();

    uint32_t bins_per_block = N_C_PER_BLOCK * N_B_MAX;
    uint32_t encs_sz = ceil_32(bins_per_block * sizeof(int16_t));
    LocalTensor<int16_t> enc = calcBuf.GetWithOffset<int16_t>(bins_per_block, calc_buf_offset);
    calc_buf_offset += encs_sz;

    LocalTensor<int16_t> lens = calcBuf.GetWithOffset<int16_t>(bins_per_block, calc_buf_offset);
    calc_buf_offset += encs_sz;

    //  Meta info in the form [ enc pattern (sym 0),  enc len (sym 0), enc pattern (sym 1),  enc len (sym 1), ... ]
    uint16_t shift_byte = 8;
    ShiftRight(enc.ReinterpretCast<uint16_t>(), meta_info, shift_byte, bins_per_block);
    ShiftLeft(lens.ReinterpretCast<uint16_t>(), meta_info, shift_byte, bins_per_block);
    ShiftRight(lens.ReinterpretCast<uint16_t>(), lens.ReinterpretCast<uint16_t>(), shift_byte, bins_per_block);

    metaDataInQ.FreeTensor(meta_info);

    // --------
    // Phase: Prepare various buffers used in the core encoding loop
    // --------
    uint32_t tmp_out_sz = ceil_32(N_T_PER_BATCH);
    LocalTensor<uint16_t> tmp_out = calcBuf.GetWithOffset<uint16_t>(N_T_PER_BATCH / sizeof(uint16_t), calc_buf_offset);
    calc_buf_offset += tmp_out_sz;

    LocalTensor<uint16_t> bytestream_out_i16 = byteStreamOutQ.AllocTensor<uint16_t>();
    LocalTensor<uint32_t> lens_out_32 = lensOutQ.AllocTensor<uint32_t>();
    LocalTensor<int16_t> lens_out = lens_out_32.ReinterpretCast<int16_t>();
    Duplicate(lens_out, static_cast<int16_t>(2), N_C_PER_BLOCK);

    uint32_t current_write_h = 0;
    uint32_t current_T_read_h_sz = ceil_32(2 * N_C_PER_BLOCK * sizeof(uint32_t));
    LocalTensor<int32_t> current_T_read_h = calcBuf.GetWithOffset<int32_t>(2 * N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += current_T_read_h_sz;
    LocalTensor<int32_t> current_T_read_h_tmp = calcBuf.GetWithOffset<int32_t>(2 * N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += current_T_read_h_sz;

    CreateVecIndex(current_T_read_h, 0, N_C_PER_BLOCK);
    Muls(current_T_read_h, current_T_read_h, static_cast<int32_t>(sizeof(int16_t)), N_C_PER_BLOCK);

    uint32_t current_out_slot_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> current_out_slot = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += current_out_slot_sz;

    uint32_t slot_write_h_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> slot_write_h = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += slot_write_h_sz;

    uint32_t overflow_Enc_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> overflow_Enc = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += overflow_Enc_sz;
    Duplicate(overflow_Enc, static_cast<int16_t>(0), N_C_PER_BLOCK);

    uint32_t overflow_Len_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> overflow_Len = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += overflow_Len_sz;
    Duplicate(overflow_Len, static_cast<int16_t>(0), N_C_PER_BLOCK);

    uint32_t next_T_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> next_T = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += next_T_sz;

    uint32_t next_L_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> next_L = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += next_L_sz;

    uint32_t next_Enc_sz = ceil_32(2 * N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> next_Enc_16 = calcBuf.GetWithOffset<int16_t>(2 * N_C_PER_BLOCK, calc_buf_offset);
    auto next_Enc_32 = next_Enc_16.ReinterpretCast<int32_t>();
    calc_buf_offset += next_Enc_sz;

    uint32_t next_T_as_byte_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint32_t));
    LocalTensor<int32_t> next_T_as_byte = calcBuf.GetWithOffset<int32_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += next_T_as_byte_sz;

    uint32_t next_T_pair_as_byte_sz = ceil_32(2 * N_C_PER_BLOCK * sizeof(uint32_t));
    LocalTensor<int32_t> next_T_pair_as_byte = calcBuf.GetWithOffset<int32_t>(2 * N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += next_T_pair_as_byte_sz;

    uint32_t shift_factor_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint32_t));
    LocalTensor<int32_t> shift_factor = calcBuf.GetWithOffset<int32_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += shift_factor_sz;

    uint32_t next_write_16_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> next_write_16 = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += next_write_16_sz;

    uint32_t pending_enc_16_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> pending_enc_16 = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += pending_enc_16_sz;

    uint32_t pending_len_16_sz = ceil_32(N_C_PER_BLOCK * sizeof(uint16_t));
    LocalTensor<int16_t> pending_len_16 = calcBuf.GetWithOffset<int16_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += pending_len_16_sz;

    uint32_t full_sz = ceil_32(256 / 8);
    LocalTensor<uint8_t> full = calcBuf.GetWithOffset<uint8_t>(256 / 8, calc_buf_offset);
    calc_buf_offset += full_sz;
    LocalTensor<uint8_t> end = calcBuf.GetWithOffset<uint8_t>(256 / 8, calc_buf_offset);
    calc_buf_offset += full_sz;
    LocalTensor<uint8_t> complete = calcBuf.GetWithOffset<uint8_t>(256 / 8, calc_buf_offset);
    calc_buf_offset += full_sz;
    LocalTensor<uint8_t> tmp_conditional = calcBuf.GetWithOffset<uint8_t>(256 / 8, calc_buf_offset);
    calc_buf_offset += full_sz;
    Duplicate(complete.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), 256 / (8 * sizeof(int16_t)));

    // 256B because of use in compare. Assumes 256 / sizeof(int32) > N_C_PER_BLOCK
    uint32_t n_t_bcast_sz = 256;
    LocalTensor<int32_t> n_t_bcast = calcBuf.GetWithOffset<int32_t>(256 / sizeof(int32_t), calc_buf_offset);
    calc_buf_offset += n_t_bcast_sz;
    CreateVecIndex(n_t_bcast, 0, N_C_PER_BLOCK);
    Muls(n_t_bcast, n_t_bcast, static_cast<int32_t>(sizeof(int16_t)), N_C_PER_BLOCK);
    Adds(n_t_bcast, n_t_bcast, N_C_PER_BLOCK * static_cast<int32_t>((n_tokens - 1) * sizeof(int16_t)), N_C_PER_BLOCK);

    LocalTensor<int16_t> cast_input_i16 = cast_input.ReinterpretCast<int16_t>();
    Cast(cast_input_i16, cast_input, RoundMode::CAST_RINT, N_T_PER_BATCH);

    // --------
    // Phase: Combine all the previous work and encode the input symbols per the encode lengths and patterns
    // --------
    // Outer loop runs until all tokens are encoded
    Compare(end, n_t_bcast, current_T_read_h, CMPMODE::EQ, 256 / sizeof(int32_t));
    bool outer_done = false;
    while (!outer_done) {
        Duplicate(current_out_slot, static_cast<int16_t>(0), N_C_PER_BLOCK);
        Duplicate(slot_write_h, static_cast<int16_t>(16), N_C_PER_BLOCK);

        // Handle overflow from previous iteration
        Or(current_out_slot, current_out_slot, overflow_Enc, N_C_PER_BLOCK);
        Muls(overflow_Len, overflow_Len, static_cast<int16_t>(-1), N_C_PER_BLOCK);
        Add(slot_write_h, slot_write_h, overflow_Len, N_C_PER_BLOCK);

        Duplicate(overflow_Enc, static_cast<int16_t>(0), N_C_PER_BLOCK);
        Duplicate(overflow_Len, static_cast<int16_t>(0), N_C_PER_BLOCK);
        Duplicate(full.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), N_C_PER_BLOCK / (8 * sizeof(int16_t)));

        // Inner loop until current buffers are full (or all tokens are encoded)
        bool inner_done = false;
        while(!(inner_done)) {
            Or(complete.ReinterpretCast<int16_t>(), end.ReinterpretCast<int16_t>(), full.ReinterpretCast<int16_t>(), N_C_PER_BLOCK / sizeof(int16_t));

            Gather(next_T, cast_input_i16, current_T_read_h.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK);

            Adds(current_T_read_h_tmp, current_T_read_h, static_cast<int32_t>(sizeof(int16_t)) * N_C_PER_BLOCK, N_C_PER_BLOCK);

            Compare(end, n_t_bcast, current_T_read_h, CMPMODE::EQ, 256 / sizeof(int32_t));
            Or(tmp_conditional.ReinterpretCast<int16_t>(), end.ReinterpretCast<int16_t>(), complete.ReinterpretCast<int16_t>(), N_C_PER_BLOCK / sizeof(int16_t));
            Select(current_T_read_h.ReinterpretCast<float>(), tmp_conditional, current_T_read_h.ReinterpretCast<float>(), current_T_read_h_tmp.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, N_C_PER_BLOCK);

            // Next_T as byte offset in lens and encs (including channel offsets)
            Cast(register_bins.ReinterpretCast<half>(), next_T, RoundMode::CAST_NONE, N_C_PER_BLOCK);
            Cast(next_T_as_byte, register_bins.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);
            Muls(next_T_as_byte, next_T_as_byte, 2, N_C_PER_BLOCK);
            Add(next_T_as_byte, next_T_as_byte, swap_channel_token_i16_idxs, N_C_PER_BLOCK);

            Gather(next_L, lens, next_T_as_byte.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK);
            Gather(register_bins.ReinterpretCast<int16_t>(), enc, next_T_as_byte.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK); // [enc c0, enc c1, enc c2]
            Gather(next_Enc_16, register_bins.ReinterpretCast<int16_t>(), duplicating_gather_i16_idxs.ReinterpretCast<uint32_t>(), 0, 2 * N_C_PER_BLOCK); // [x, enc c0, x, enc c1, ... ]
            uint64_t mul_mask[1] = {0xAAAAAAAAAAAAAAAA}; // Every other element for 64
            Muls(next_Enc_16, next_Enc_16, static_cast<int16_t>(0), mul_mask, 1, {1, 1, 8, 8}); // [0, enc c0, 0, enc c1, ... ]

            // Note various endian-ness complexities that come into play with ReinterpretCast. Represented as u32:
            // Logical order
            // |               Symbol 0             |               Symbol 1             |
            // | Byte 0 | Byte 1 | Byte 2 | Bytes 3 | Byte 0 | Byte 1 | Byte 2 | Bytes 3 |
            // |   0    |   0    |   0    | Enc --- |   0    |   0    |   0    | Enc --- | Right aligned
            //
            // - Shift left 8 aligns to i16 boundary
            // - Shift left another slot_write_h aligns to write h
            //
            // The per symbol shift is achieved with a multiplication by 2^n. Gathering those factors needs a
            // x sizeof(int32) to convert to a byte offset into an int32 tensor
            Adds(register_bins.ReinterpretCast<int16_t>(), slot_write_h, static_cast<int16_t>(8), N_C_PER_BLOCK);
            Muls(register_bins.ReinterpretCast<int16_t>(), register_bins.ReinterpretCast<int16_t>(), static_cast<int16_t>(sizeof(int32_t)), N_C_PER_BLOCK);
            Cast(register_bins.ReinterpretCast<half>(), register_bins.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, N_C_PER_BLOCK);
            Cast(register_bins_2, register_bins.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);
            Gather(shift_factor, p2s_32_arr, register_bins_2.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK);

            // next_Enc_32 is int32 (to support muls). What does this mean when shifting a 1 into the m.s.b?
            // Rather than rely on undocumented overflow behaviour this shifts by (n - 1) and unconditionally
            // shifts left 1
            Mul(next_Enc_32, next_Enc_32, shift_factor, N_C_PER_BLOCK);
            ShiftLeft(next_Enc_32, next_Enc_32, 1, N_C_PER_BLOCK);

            Gather(next_write_16, next_Enc_16, reducing_gather_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK);
            Or(current_out_slot, current_out_slot, next_write_16, N_C_PER_BLOCK);

            // slot_write_h = slot_write_h - len // +ve => all written, 0 or -ve => full and maybe spill
            Muls(register_bins.ReinterpretCast<int16_t>(), next_L, static_cast<int16_t>(-1), N_C_PER_BLOCK);
            Add(slot_write_h, slot_write_h, register_bins.ReinterpretCast<int16_t>(), N_C_PER_BLOCK);

            // Overflow len = Max(-slot_write_h, 0);
            Muls(register_bins.ReinterpretCast<int16_t>(), slot_write_h, static_cast<int16_t>(-1), N_C_PER_BLOCK);
            Maxs(pending_len_16, register_bins.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), N_C_PER_BLOCK);

            // Correct write_h to boundary
            Maxs(slot_write_h, slot_write_h, static_cast<int16_t>(0), N_C_PER_BLOCK);
            Cast(register_bins.ReinterpretCast<half>(), slot_write_h, RoundMode::CAST_NONE, N_C_PER_BLOCK);
            Cast(register_bins_2, register_bins.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);
            CompareScalar(full, register_bins_2, 0, CMPMODE::EQ, 2 * N_C_PER_BLOCK);

            ShiftLeft(next_Enc_32, next_Enc_32, 16, N_C_PER_BLOCK);
            Gather(pending_enc_16, next_Enc_16, reducing_gather_i16_idxs.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK);

            // Select with full & !complete - stay the same or change first time full
            Not(tmp_conditional.ReinterpretCast<int16_t>(), complete.ReinterpretCast<int16_t>(), N_C_PER_BLOCK / sizeof(int16_t));
            And(tmp_conditional.ReinterpretCast<int16_t>(), tmp_conditional.ReinterpretCast<int16_t>(), full.ReinterpretCast<int16_t>(), N_C_PER_BLOCK / sizeof(int16_t));
            Select(overflow_Enc.ReinterpretCast<half>(), tmp_conditional, pending_enc_16.ReinterpretCast<half>(), overflow_Enc.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, N_C_PER_BLOCK);
            Select(overflow_Len.ReinterpretCast<half>(),  tmp_conditional, pending_len_16.ReinterpretCast<half>(), overflow_Len.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, N_C_PER_BLOCK);

            Or(tmp_conditional.ReinterpretCast<int16_t>(), full.ReinterpretCast<int16_t>(), complete.ReinterpretCast<int16_t>(), N_C_PER_BLOCK / sizeof(int16_t));

            inner_done = tmp_conditional.ReinterpretCast<int32_t>()(0) == 0xffffffff;
        }

        Copy(tmp_out[current_write_h * N_C_PER_BLOCK], current_out_slot.ReinterpretCast<uint16_t>(), 32, 1, {1, 1, 8, 8});
        ++current_write_h;

        // note, init out len to 2
        Adds(register_bins.ReinterpretCast<int16_t>(), lens_out, static_cast<int16_t>(2), N_C_PER_BLOCK);
        Select(lens_out.ReinterpretCast<half>(),  end, lens_out.ReinterpretCast<half>(), register_bins.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, N_C_PER_BLOCK);
        uint32_t full_i = full.ReinterpretCast<uint32_t>()(0);
        uint32_t end_i = end.ReinterpretCast<uint32_t>()(0);
        outer_done = (end_i & ~full_i) == 0xffffffff;
    }

    // Handle final overflow
    Duplicate(current_out_slot, static_cast<int16_t>(0), N_C_PER_BLOCK);
    Or(current_out_slot, current_out_slot, overflow_Enc, N_C_PER_BLOCK);
    Copy(tmp_out[current_write_h * N_C_PER_BLOCK], current_out_slot.ReinterpretCast<uint16_t>(), 32, 1, {1, 1, 8, 8});
    Cast(overflow_Len.ReinterpretCast<half>(), overflow_Len, RoundMode::CAST_NONE, N_C_PER_BLOCK);
    half zero = 0.;
    CompareScalar(tmp_conditional, overflow_Len.ReinterpretCast<half>(), zero , CMPMODE::EQ,  N_C_PER_BLOCK);
    Not(tmp_conditional.ReinterpretCast<int16_t>(), tmp_conditional.ReinterpretCast<int16_t>(), N_C_PER_BLOCK / sizeof(int16_t));

    Adds(register_bins.ReinterpretCast<int16_t>(), lens_out, static_cast<int16_t>(2), N_C_PER_BLOCK);
    Select(lens_out.ReinterpretCast<half>(),  tmp_conditional, lens_out.ReinterpretCast<half>(), register_bins.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, N_C_PER_BLOCK);

    // Gather together the encode bytes for each channel
    auto offset_to_layer = layer_id * n_channels * N_T_MAX;
    auto offset_into_layer = channel_start_id * N_T_MAX;
    auto base_offset = offset_to_layer + offset_into_layer;
    auto encode_cum_len = 0;
    for (int32_t c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        Gather(bytestream_out_i16[(N_T_MAX / 2) * c_ii], tmp_out, swap_channel_token_i16_idxs.ReinterpretCast<uint32_t>(), sizeof(half) * c_ii, N_T_MAX/2);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
    }

    byteStreamOutQ.EnQue(bytestream_out_i16);
    bytestream_out_i16 = byteStreamOutQ.DeQue<uint16_t>();

    // --------
    // Phase: Copy out packing the byte streams together
    // --------
    Cast(register_bins.ReinterpretCast<half>(), lens_out, RoundMode::CAST_NONE, N_C_PER_BLOCK);
    Cast(lens_out_32.ReinterpretCast<int32_t>(), register_bins.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);

    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::ALL>();
    for (int32_t c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        DataCopy(g_out_bytes[base_offset + encode_cum_len], bytestream_out_i16.ReinterpretCast<uint8_t>()[N_T_MAX * c_ii], ceil_32(lens_out_32.GetValue(c_ii)));
        encode_cum_len += lens_out_32.GetValue(c_ii);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
    }
    byteStreamOutQ.FreeTensor(bytestream_out_i16);

    lensOutQ.EnQue(lens_out_32);
    lens_out_32 = lensOutQ.DeQue<uint32_t>();
    DataCopy(g_out_lens[layer_id * n_channels + channel_start_id], lens_out_32, N_C_PER_BLOCK);
    lensOutQ.FreeTensor(lens_out_32);
}

class PacEncoderRectifier {
public:
    __aicore__ inline PacEncoderRectifier(
        GM_ADDR out_bytes, // Out bytes [n_layers, n_channels, batch_size], uint8
        GM_ADDR out_lens, // Out lengths [n_layers, n_channels], uint32

        TPipe& pipe,

        int32_t n_tokens,
        int32_t n_layers,
        int32_t n_channels,
        uint32_t n_bins,
        int32_t chunk_size);

    __aicore__ inline void rectify();

private:

    // Output Queues
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> byteStreamBoundQ;
    GlobalTensor<uint8_t> g_in_out_bytes;

    TQue<TPosition::VECIN, 2> lensInQ;
    TQue<TPosition::VECOUT, 2> lensOutQ;
    GlobalTensor<uint32_t> g_in_out_lens;

    TPipe& pipe;

    // Dimensionality
    int32_t n_tokens;
    int32_t n_layers;
    int32_t n_channels;
    uint32_t n_bins;
    int32_t chunk_size;

    // Intermediate buffers
    TBuf<TPosition::VECCALC> calcBuf; // For transient (per rectify) intermediates

    // Class has no known need to support move or copy operations
    PacEncoderRectifier(const PacEncoderRectifier&) = delete;
    PacEncoderRectifier& operator=(const PacEncoderRectifier&) = delete;
    PacEncoderRectifier(PacEncoderRectifier&&) = delete;
    PacEncoderRectifier& operator=(PacEncoderRectifier&&) = delete;
};
__aicore__ inline PacEncoderRectifier::PacEncoderRectifier(
    GM_ADDR out_bytes,
    GM_ADDR out_lens,
    TPipe& _pipe,
    int32_t n_tokens,
    int32_t n_layers,
    int32_t n_channels,
    uint32_t n_bins,
    int32_t chunk_size):
        pipe(_pipe),
        n_tokens(n_tokens),
        n_layers(n_layers),
        n_channels(n_channels),
        n_bins(n_bins),
        chunk_size(chunk_size) {
    g_in_out_bytes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(out_bytes), n_layers * n_channels * chunk_size);
    g_in_out_lens.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(out_lens), n_layers * n_channels);

    pipe.InitBuffer(byteStreamBoundQ, 2, N_T_PER_BATCH);
    pipe.InitBuffer(lensInQ, 2, N_C_MAX * sizeof(int32_t));
    pipe.InitBuffer(lensOutQ, 2, N_C_MAX * sizeof(int32_t));

    uint32_t calc_buf_sz_aligned = 0x10000;
    pipe.InitBuffer(calcBuf, calc_buf_sz_aligned);
}

__aicore__ inline void PacEncoderRectifier::rectify() {
    uint32_t calc_buf_offset = 0;
    uint32_t boundaries_per_layer = n_channels / N_C_PER_BLOCK;
    uint32_t boundaries_per_layer_aligned = ceil_32(n_channels / N_C_PER_BLOCK);
    uint32_t boundaries_count = boundaries_per_layer_aligned * n_layers;
    uint32_t boundaries_sz = ceil_32(boundaries_count * sizeof(int32_t));
    LocalTensor<uint32_t> boundaries = calcBuf.GetWithOffset<uint32_t>(boundaries_count, calc_buf_offset);
    calc_buf_offset += boundaries_sz;

    uint32_t boundaries_gather_idxs_count = n_channels / N_C_PER_BLOCK;
    uint32_t boundaries_gather_idxs_count_aligned = ceil_32(n_channels / N_C_PER_BLOCK);
    uint32_t boundaries_gather_idxs_sz = ceil_32(boundaries_gather_idxs_count_aligned * sizeof(int32_t));
    LocalTensor<int32_t> boundaries_gather_idxs = calcBuf.GetWithOffset<int32_t>(boundaries_gather_idxs_count_aligned, calc_buf_offset);
    calc_buf_offset += boundaries_gather_idxs_sz;
    CreateVecIndex(boundaries_gather_idxs, 0, boundaries_gather_idxs_count_aligned);
    Muls(boundaries_gather_idxs, boundaries_gather_idxs, 4 * N_C_PER_BLOCK, boundaries_gather_idxs_count_aligned);

    uint32_t last = 0; // Correction between layers
    for (auto l_ii = 0; l_ii < n_layers; ++l_ii) {
        LocalTensor<int32_t> lens_in = lensInQ.AllocTensor<int32_t>();

        DataCopy(lens_in.ReinterpretCast<uint32_t>(), g_in_out_lens[l_ii * n_channels], n_channels);

        lensInQ.EnQue(lens_in);
        lens_in = lensInQ.DeQue<int32_t>();

        LocalTensor<int32_t> lens_out = lensOutQ.AllocTensor<int32_t>();
        Cast(lens_out.ReinterpretCast<float>(), lens_in, RoundMode::CAST_NONE, n_channels);
        auto lens_out_f = lens_out.ReinterpretCast<float>();
        lensInQ.FreeTensor(lens_in);

        // CumSum internally applies for memory, for stability invoke it with a fixed size avoiding complexity
        // with variable n_channels requiring different footprints
        auto elem_idx = 0;
        uint32_t sum_span = 128;
        float prev_last_f = 0.; // Corrections within a layer
        for (; (elem_idx + sum_span) <= n_channels; elem_idx += sum_span) {
            auto lens_out_f_span = lens_out_f[elem_idx];
            float first = lens_out_f_span.GetValue(0);
            lens_out_f_span.SetValue(0, prev_last_f + first);

            static constexpr CumSumConfig cum_sum_cfg = {true, true, false};
            CumSumInfo cum_sum_info = {1, sum_span};
            CumSum<float, cum_sum_cfg>(lens_out_f_span, lens_out_f_span, lens_out_f_span, cum_sum_info);

            prev_last_f = lens_out_f_span.GetValue(sum_span - 1);
        }

        // Handle tail for when n_channels % sum_span != 0
        if (elem_idx < n_channels) {
            auto lens_out_f_span = lens_out_f[elem_idx];
            float first = lens_out_f_span.GetValue(0);
            lens_out_f_span.SetValue(0, prev_last_f + first);
            uint32_t cum_sum_tail = n_channels - elem_idx;

            static constexpr CumSumConfig cum_sum_cfg = {true, true, false};
            CumSumInfo cum_sum_info = {1, cum_sum_tail};
            CumSum<float, cum_sum_cfg>(lens_out_f_span, lens_out_f_span, lens_out_f_span, cum_sum_info);
        }


        Cast(lens_out, lens_out.ReinterpretCast<float>(), RoundMode::CAST_RINT, n_channels);

        Adds(lens_out, lens_out, static_cast<int32_t>(last), n_channels);
        last = lens_out.GetValue(n_channels - 1);

        Gather(boundaries[l_ii * boundaries_per_layer_aligned], lens_out.ReinterpretCast<uint32_t>(), boundaries_gather_idxs.ReinterpretCast<uint32_t>(), 4 * (N_C_PER_BLOCK - 1), boundaries_per_layer);

        lensOutQ.EnQue(lens_out);
        lens_out = lensOutQ.DeQue<int32_t>();
        DataCopy(g_in_out_lens[l_ii * n_channels], lens_out.ReinterpretCast<uint32_t>(), n_channels);
        lensOutQ.FreeTensor(lens_out);
    }

    for (auto l_ii = 0; l_ii < n_layers; ++l_ii) {
        for ( auto c_ii = 0; c_ii < n_channels; c_ii += N_C_PER_BLOCK) {
            auto c_batch_ii = c_ii / N_C_PER_BLOCK;

            LocalTensor<uint8_t> enc_bytes = byteStreamBoundQ.AllocTensor<uint8_t>();
            DataCopy(enc_bytes, g_in_out_bytes[l_ii * n_channels * N_T_MAX + c_ii * N_T_MAX], N_T_PER_BATCH);
            byteStreamBoundQ.EnQue(enc_bytes);
            enc_bytes = byteStreamBoundQ.DeQue<uint8_t>();

            auto offset = 0;
            if (c_batch_ii != 0) {
                offset = boundaries(l_ii * boundaries_per_layer_aligned + c_batch_ii - 1);
            } else if (l_ii != 0) {
                offset = boundaries((l_ii - 1) * boundaries_per_layer_aligned + boundaries_per_layer - 1);
            } // else c_batch == 0 and l_ii == 0 => leave offset as 0

            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::ALL>();
            DataCopy(g_in_out_bytes[offset], enc_bytes, N_T_PER_BATCH);

            byteStreamBoundQ.FreeTensor(enc_bytes);
        }
    }
}

} // namespace impl
} // namespace pac_coder
} // namespace kvcache_ops

extern "C" __global__ __aicore__ void pac_encode_kernel (
    GM_ADDR input_data_ptr,
    GM_ADDR meta_data_ptr,
    GM_ADDR output_data_ptr,
    GM_ADDR output_lengths_data_ptr,
    const uint32_t n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels,
    const int chunk_size,
    const float scale_factor,
    GM_ADDR workGM_ptr
) {
    TPipe pipe{};
    int32_t coreIdx = GetBlockIdx();
    int32_t launchedCores = GetBlockNum();

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    {
        kvcache_ops::pac_coder::impl::PacEncoder encoder {
            input_data_ptr,
            meta_data_ptr,
            output_data_ptr,
            output_lengths_data_ptr,
            pipe,
            n_tokens,
            n_layers,
            n_channels,
            n_bins,
            chunk_size,
            static_cast<half>(scale_factor)};

        int max_work_idx = n_layers * n_channels / kvcache_ops::pac_coder::N_C_PER_BLOCK;
        for (int work_idx = coreIdx; work_idx < max_work_idx; work_idx += launchedCores) {
            int layer_id = work_idx % n_layers;
            int channel_id = kvcache_ops::pac_coder::N_C_PER_BLOCK * (work_idx / n_layers);
            encoder.encode(layer_id, channel_id);
        }
    }

    pipe.Reset();

    GlobalTensor<int32_t> syncAllGM;
    auto DEFAULT_SYNCALL_NEED_SIZE = 32 / sizeof(int32_t);
    syncAllGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workGM_ptr), 40 * DEFAULT_SYNCALL_NEED_SIZE);

    TQue<AscendC::TPosition::VECIN, 1> workQueue;
    pipe.InitBuffer(workQueue, 1, 40 * 32);
    LocalTensor<int32_t> workLocal = workQueue.AllocTensor<int32_t>();

    SyncAll(syncAllGM, workLocal, launchedCores);
    workQueue.FreeTensor(workLocal);

    pipe.Reset();

    if (coreIdx == 0) {
        kvcache_ops::pac_coder::impl::PacEncoderRectifier rectifier {output_data_ptr,
            output_lengths_data_ptr,
            pipe,
            n_tokens,
            n_layers,
            n_channels,
            n_bins,
            chunk_size};

        rectifier.rectify();
    }
}

extern "C" __global__ __aicore__ void pac_prep_enc_metadata_kernel (
    GM_ADDR input_data_ptr,
    GM_ADDR meta_data_ptr,
    const uint32_t n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels,
    const float scale_factor
) {
    TPipe pipe{};
    int32_t coreIdx = GetBlockIdx();
    int32_t launchedCores = GetBlockNum();

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    {
        kvcache_ops::pac_coder::impl::PacEncoder encoder {
            input_data_ptr,
            meta_data_ptr,
            NULL,
            NULL,
            pipe,
            n_tokens,
            n_layers,
            n_channels,
            n_bins,
            -1,
            static_cast<half>(scale_factor)};

        int max_work_idx = n_layers * n_channels / kvcache_ops::pac_coder::N_C_PER_BLOCK;
        for (int work_idx = coreIdx; work_idx < max_work_idx; work_idx += launchedCores) {
            int layer_id = work_idx % n_layers;
            int channel_id = kvcache_ops::pac_coder::N_C_PER_BLOCK * (work_idx / n_layers);
            encoder.meta_data_calc(layer_id, channel_id);
        }
    }
}

namespace kvcache_ops {
namespace pac_coder {

void pac_encode(
    uint8_t* input_data_ptr,
    uint8_t* meta_data_ptr,
    uint8_t* output_data_ptr,
    uint8_t* output_lengths_data_ptr,
    void* stream,
    const int n_aiv,
    const int n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels,
    const int chunk_size,
    uint8_t* workGM_ptr) {

    int blockDim = n_layers * (n_channels / N_C_PER_BLOCK) < n_aiv ? n_layers * (n_channels / N_C_PER_BLOCK) : n_aiv;

    float scale_factor = static_cast<half>(kvcache_ops::pac_coder::N_T_MAX) / static_cast<half>(n_tokens);
    pac_encode_kernel<<<blockDim, nullptr, stream>>>(
        input_data_ptr,
        meta_data_ptr,
        output_data_ptr,
        output_lengths_data_ptr,
        n_bins,
        n_tokens,
        n_layers,
        n_channels,
        chunk_size,
        scale_factor,
        workGM_ptr);
}

void pac_prep_enc_metadata(
    uint8_t* input_data_ptr,
    uint8_t* meta_data_ptr,
    void* stream,
    const int n_aiv,
    const int n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels) {

    int blockDim = n_layers * (n_channels / N_C_PER_BLOCK) < n_aiv ? n_layers * (n_channels / N_C_PER_BLOCK) : n_aiv;

    float scale_factor = static_cast<half>(kvcache_ops::pac_coder::N_T_MAX - kvcache_ops::pac_coder::N_B_MAX) / static_cast<half>(n_tokens);
    pac_prep_enc_metadata_kernel<<<blockDim, nullptr, stream>>>(
        input_data_ptr,
        meta_data_ptr,
        n_bins,
        n_tokens,
        n_layers,
        n_channels,
        scale_factor);
}

} // namespace pac_coder
} // namespace kvcache_ops