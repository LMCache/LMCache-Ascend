#include "kernel_operator.h"
using namespace AscendC;

namespace kvcache_ops {
namespace pac_coder {

constexpr int32_t N_C_PER_BLOCK = 32;
// Some copy masks implicitly assume 32 channels - update as appropriate before changing this assert
static_assert(N_C_PER_BLOCK == 32);

constexpr uint32_t N_T_MAX = 256;
constexpr uint32_t N_B_MAX = 32;
constexpr int32_t DATABLOCK_BYTES = 32;
constexpr int32_t N_DBs_PER_BLOCK = N_C_PER_BLOCK / DATABLOCK_BYTES;

namespace impl {
__aicore__ inline auto ceil_32(int32_t size) -> uint32_t {
    return size % 32 == 0 ? size : 32 * (1 + (size / 32));
};

__aicore__ inline auto read_unaligned_u8_2_half(
    LocalTensor<uint8_t>& src,
    LocalTensor<uint32_t>& src_gather_idxs, // in form idx 0, idx 0, idx 1, idx 1 in bits
    LocalTensor<half>& dst,
    LocalTensor<uint32_t>& tmp_slot_1,
    LocalTensor<uint32_t>& tmp_slot_2,
    LocalTensor<int32_t>& pows_2,
    uint32_t count) -> void {
        auto src_i16 = src.ReinterpretCast<int16_t>();

        // Identify which i16 the bit index refers to and convert to a byte offset
        ShiftRight(tmp_slot_1, src_gather_idxs.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(4), 2*count);
        ShiftLeft(tmp_slot_1, tmp_slot_1, static_cast<uint32_t>(1), 2*count);

        // Gather the relevant i16 and the following i16 which provides back fill when shifting to the bits of interest
        uint64_t mask[2] = {0};
        mask[0] = 0xAAAAAAAAAAAAAAAA;
        Gather(tmp_slot_2.ReinterpretCast<int16_t>(), src_i16, tmp_slot_1, 0, mask, 1, 0);

        mask[0] = 0x5555555555555555;
        Gather(tmp_slot_2.ReinterpretCast<int16_t>(), src_i16, tmp_slot_1, 2, mask, 1, 0);

        // Identify the bit offset within the temporary buffer
        ShiftLeft(tmp_slot_1, src_gather_idxs.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(28), 2*count);
        ShiftRight(tmp_slot_1, tmp_slot_1, static_cast<uint32_t>(28), 2*count);

        // Shifting different elements by different amounts is achieved by multiplying by 2^n
        Muls(tmp_slot_1.ReinterpretCast<int32_t>(), tmp_slot_1.ReinterpretCast<int32_t>(), 4, 2*count);
        Gather(tmp_slot_1.ReinterpretCast<int32_t>(), pows_2, tmp_slot_1, 0, 2*count);
        uint64_t _rsvd = 0;
        auto repeats = 1;
        GatherMaskParams gmp = {
            1, // src0BlockStride. 1 - Continuous data
            static_cast<uint8_t>(repeats),
            8, // src0RepeatStride - Continuous Data
            0 // src1RepeatStride - not used
        };
        GatherMask(tmp_slot_1, tmp_slot_1, 2, false, 0, gmp, _rsvd);

        Mul(tmp_slot_2.ReinterpretCast<int32_t>(), tmp_slot_2.ReinterpretCast<int32_t>(), tmp_slot_1.ReinterpretCast<int32_t>(), count);

        // Reduce to range 0 - 255 (8 bit)
        ShiftRight(tmp_slot_2, tmp_slot_2, static_cast<uint32_t>(24), count);
        Cast(dst, tmp_slot_2.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
}

class PacDecoder {
public:
    __aicore__ inline PacDecoder(
        GM_ADDR meta_data_ptr, // In bytesteam [steam_length], uint8
        GM_ADDR cum_lens_ptr, // In cum lengths [n_layers, n_channels], uint64
        GM_ADDR bytestream_ptr,  // In CDF [n_layers, n_channels, n_bins], uint16
        GM_ADDR output_data_ptr, // Output symbols [n_layers, batch_size, n_channels], uint8
        AscendC::TPipe& pipe,
        int32_t n_tokens,
        int32_t n_layers,
        int32_t n_channels,
        uint32_t n_bins);

    __aicore__ inline void decode(int layer_id, int channel_id);

private:
    AscendC::TQue<AscendC::TPosition::VECIN, 2> byteStreamInQ;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> lensInQ;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> MetaInQ;

    AscendC::GlobalTensor<uint16_t> gm_meta_data;
    AscendC::GlobalTensor<uint32_t> gm_cum_lens;
    AscendC::GlobalTensor<uint8_t> gm_bytestream;

    AscendC::TQue<AscendC::TPosition::VECOUT, 2> symOutQ;
    AscendC::GlobalTensor<uint8_t> gm_output_data;

    AscendC::TPipe& pipe;

    // Dimensionality
    int32_t n_tokens;
    int32_t n_layers;
    int32_t n_channels;
    uint32_t n_bins;

    // For transient (per encode) intermediates
     TBuf<TPosition::VECCALC> calcBuf;
    uint32_t calc_buf_offset_init = 0;
    LocalTensor<int32_t> duplicating_gather_idxs;
    LocalTensor<int32_t> pows_2;

    // Class has no known need to support move or copy operations
    PacDecoder(const PacDecoder&) = delete;
    PacDecoder& operator=(const PacDecoder&) = delete;
    PacDecoder(PacDecoder&&) = delete;
    PacDecoder& operator=(PacDecoder&&) = delete;
};

__aicore__ inline PacDecoder::PacDecoder(
    GM_ADDR meta_data_ptr, // In bytesteam [steam_length], uint8
    GM_ADDR cum_lens_ptr, // In cum lengths [n_layers, n_channels], uint64
    GM_ADDR bytestream_ptr,  // In CDF [n_layers, n_channels, n_bins], uint16
    GM_ADDR output_data_ptr, // Output symbols [n_layers, batch_size, n_channels], uint8
    AscendC::TPipe& _pipe,
    int32_t n_tokens,
    int32_t n_layers,
    int32_t n_channels,
    uint32_t n_bins):
        pipe(_pipe),
        n_tokens(n_tokens),
        n_layers(n_layers),
        n_channels(n_channels),
        n_bins(n_bins) {
    half deq_scale = 1.0;
    SetDeqScale(deq_scale);

    auto MetaInQSize = N_C_PER_BLOCK * N_B_MAX * sizeof(uint16_t);
    pipe.InitBuffer(MetaInQ, 1, MetaInQSize);
    auto gm_meta_data_dim = n_layers * n_channels * N_B_MAX;
    gm_meta_data.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(meta_data_ptr), gm_meta_data_dim);

    auto lensInQSize = 2 * N_C_PER_BLOCK * sizeof(uint32_t);
    pipe.InitBuffer(lensInQ, 1, lensInQSize);
    auto gm_cum_lens_dim = n_layers * n_channels;
    gm_cum_lens.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cum_lens_ptr), gm_cum_lens_dim);

    auto byteStreamInQSize = N_C_PER_BLOCK * N_T_MAX * sizeof(uint8_t);
    pipe.InitBuffer(byteStreamInQ, 1, byteStreamInQSize);
    auto gm_bytestream_dim =  ceil_32(gm_cum_lens((n_layers * n_channels) - 1));
    gm_bytestream.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(bytestream_ptr), gm_bytestream_dim);

    auto symOutQSize = N_C_PER_BLOCK * N_T_MAX * sizeof(uint8_t);
    pipe.InitBuffer(symOutQ, 1, symOutQSize);
    auto gm_output_data_dim = n_layers * n_channels * N_T_MAX;
    gm_output_data.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output_data_ptr), gm_output_data_dim);

    uint32_t calc_buf_sz_aligned = 0x18000;
    pipe.InitBuffer(calcBuf, calc_buf_sz_aligned);

    calc_buf_offset_init = 0;
    uint32_t duplicating_gather_idxs_sz = ceil_32(N_C_PER_BLOCK * N_B_MAX * sizeof(int32_t));
    duplicating_gather_idxs = calcBuf.GetWithOffset<int32_t>(N_C_PER_BLOCK * N_B_MAX, calc_buf_offset_init);
    calc_buf_offset_init += duplicating_gather_idxs_sz;
    for (auto c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        Duplicate(duplicating_gather_idxs[c_ii * N_B_MAX], static_cast<int32_t>(c_ii * sizeof(half)), N_B_MAX);
    }

    uint32_t pows_2_sz = ceil_32(32 * sizeof(int32_t));
    pows_2 = calcBuf.GetWithOffset<int32_t>(32, calc_buf_offset_init);
    calc_buf_offset_init += pows_2_sz;
    for (auto ii = 0; ii < 32; ++ii) {
        pows_2.SetValue(ii, (1 << ii));
    }
}

__aicore__ inline void PacDecoder::decode(int layer_id, int channel_start_id) {
    // --------
    // Phase: Copy IN
    // --------
    uint32_t offset_idx = layer_id * n_channels + channel_start_id;
    auto lens = lensInQ.AllocTensor<uint32_t>();
    if (offset_idx == 0) {
        Duplicate(lens.ReinterpretCast<int32_t>(), 0, N_C_PER_BLOCK);
    } else {
        DataCopy(lens, gm_cum_lens[offset_idx - N_C_PER_BLOCK], N_C_PER_BLOCK);
    }
    DataCopy(lens[N_C_PER_BLOCK], gm_cum_lens[offset_idx], N_C_PER_BLOCK);

    lensInQ.EnQue(lens);
    lens = lensInQ.DeQue<uint32_t>();
    uint32_t calc_buf_offset = calc_buf_offset_init;

    auto bytestream = byteStreamInQ.AllocTensor<uint8_t>();
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::ALL>();
    uint32_t offset_start = offset_idx == 0 ? 0 : lens.GetValue(N_C_PER_BLOCK - 1);
    int32_t max_len = 0;
    for (auto c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        uint32_t offset_end = lens.GetValue(N_C_PER_BLOCK + c_ii);
        max_len = max_len < offset_end - offset_start ? offset_end - offset_start : max_len;
        uint32_t copy_len = ceil_32(offset_end - offset_start);
        DataCopy(bytestream[c_ii * N_T_MAX], gm_bytestream[offset_start], copy_len);
        offset_start = offset_end;
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
    }
    byteStreamInQ.EnQue(bytestream);
    bytestream = byteStreamInQ.DeQue<uint8_t>();
    lensInQ.FreeTensor(lens);

    auto meta_info = MetaInQ.AllocTensor<uint16_t>();
    DataCopy(meta_info, gm_meta_data[offset_idx * N_B_MAX], N_C_PER_BLOCK * N_B_MAX);
    MetaInQ.EnQue(meta_info);
    meta_info = MetaInQ.DeQue<uint16_t>();

    uint32_t bins_per_block = N_C_PER_BLOCK * N_B_MAX;
    uint32_t encs_sz = ceil_32(bins_per_block * sizeof(int16_t));
    LocalTensor<int16_t> encs = calcBuf.GetWithOffset<int16_t>(bins_per_block, calc_buf_offset);
    calc_buf_offset += encs_sz;

    LocalTensor<int16_t> enc_lens = calcBuf.GetWithOffset<int16_t>(bins_per_block, calc_buf_offset);
    calc_buf_offset += encs_sz;

    uint32_t enc_lens_32_sz = ceil_32(bins_per_block * sizeof(int32_t));
    LocalTensor<int32_t> enc_lens_32 = calcBuf.GetWithOffset<int32_t>(bins_per_block, calc_buf_offset);
    calc_buf_offset += enc_lens_32_sz;

    // --------
    // Phase: Derive Decode Table
    // --------

    //  Meta info in the form [enc pattern (sym 0), enc len (sym 0) | enc pattern (sym 1), enc len (sym 1) | ... ]
    uint16_t shift_byte = 8;
    ShiftRight(encs.ReinterpretCast<uint16_t>(), meta_info, shift_byte, bins_per_block);
    ShiftLeft(enc_lens.ReinterpretCast<uint16_t>(), meta_info, shift_byte, bins_per_block);
    ShiftRight(enc_lens.ReinterpretCast<uint16_t>(), enc_lens.ReinterpretCast<uint16_t>(), shift_byte, bins_per_block);

    Cast(enc_lens.ReinterpretCast<half>(), enc_lens, RoundMode::CAST_NONE, bins_per_block);
    Cast(enc_lens_32, enc_lens.ReinterpretCast<half>(), RoundMode::CAST_RINT, bins_per_block);

    MetaInQ.FreeTensor(meta_info);

    LocalTensor<half> encs_h = encs.ReinterpretCast<half>();
    Cast(encs_h, encs, RoundMode::CAST_NONE, bins_per_block);

    uint32_t cmp_sz = ceil_32(bins_per_block / 8);
    LocalTensor<int8_t> cmp_mask = calcBuf.GetWithOffset<int8_t>(bins_per_block / 8, calc_buf_offset);

    // Detect un-encodeable syms - some symbols are un-encodable, to ensure they don't interfere with decode
    // set their bin boundaries beyond the max supported value of 256
    CompareScalar(cmp_mask, enc_lens_32, 9, CMPMODE::EQ, bins_per_block);
    Not(cmp_mask.ReinterpretCast<int16_t>(), cmp_mask.ReinterpretCast<int16_t>(), cmp_sz / sizeof(int16_t));
    half inval = 512.;
    Select(encs_h, cmp_mask, encs_h, inval, SELMODE::VSEL_TENSOR_SCALAR_MODE, bins_per_block);

    uint32_t sorted_count_sz = ceil_32(bins_per_block * 8);
    LocalTensor<half> sorted = calcBuf.GetWithOffset<half>(sorted_count_sz / sizeof(half), calc_buf_offset);
    calc_buf_offset += sorted_count_sz;

    // Sort the encode bins - Quick symbol lookup requires the lookup table is ordered
    uint32_t idxs_sz = ceil_32(N_B_MAX * sizeof(int32_t));
    LocalTensor<int32_t> idxs = calcBuf.GetWithOffset<int32_t>(N_B_MAX, calc_buf_offset);
    CreateVecIndex(idxs, 0, N_B_MAX);
    calc_buf_offset += idxs_sz;

    for (uint32_t c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        Sort32(sorted[c_ii * N_B_MAX * (8 / sizeof(half))], encs_h[c_ii * N_B_MAX], idxs.ReinterpretCast<uint32_t>(), N_B_MAX / 32);
    }

    uint64_t _rsvd = 0;
    auto elems_per_256 = 256 / 8;
    auto repeats = bins_per_block / elems_per_256;
    GatherMaskParams gmp = {
        1, // src0BlockStride. 1 - Continuous data
        static_cast<uint8_t>(repeats),
        8, // src0RepeatStride - Continuous Data
        0 // src1RepeatStride - not used
    };

    uint32_t enc_syms_sz = ceil_32(bins_per_block * sizeof(int32_t));
    LocalTensor<int32_t> enc_syms = calcBuf.GetWithOffset<int32_t>(bins_per_block, calc_buf_offset);
    calc_buf_offset += enc_syms_sz;

    uint32_t enc_bins_sz = ceil_32(bins_per_block * sizeof(half));
    LocalTensor<half> enc_bins = calcBuf.GetWithOffset<half>(bins_per_block, calc_buf_offset);
    calc_buf_offset += enc_bins_sz;

    GatherMask(enc_syms, sorted.ReinterpretCast<int32_t>(), 2, false, 0, gmp, _rsvd);
    GatherMask(enc_bins, sorted.ReinterpretCast<half>(), 3, false, 0, gmp, _rsvd);

    // Shift encode values back one to get bin boundaries
    for (auto c_ii = 0; c_ii < N_C_PER_BLOCK; ++c_ii) {
        int16_t next = 0;
        auto last = enc_bins.GetValue(c_ii * N_B_MAX + 0);;
        for (auto b_ii = 0; b_ii < N_B_MAX; ++b_ii) {
            next = last;
            last = enc_bins.GetValue(c_ii * N_B_MAX + b_ii);
            enc_bins.SetValue(c_ii * N_B_MAX + b_ii, next);
        }
    }

    // --------
    // Phase: Prepare temporaries for decode
    // --------
    uint32_t curr_dec_val_sz = ceil_32(N_C_PER_BLOCK * sizeof(half));
    LocalTensor<half> curr_dec_val = calcBuf.GetWithOffset<half>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += curr_dec_val_sz;

    uint32_t curr_dec_val_bcast_sz = ceil_32(bins_per_block * sizeof(half));
    LocalTensor<half> curr_dec_val_bcast = calcBuf.GetWithOffset<half>(bins_per_block, calc_buf_offset);
    calc_buf_offset += curr_dec_val_bcast_sz;

    byteStreamInQ.FreeTensor(bytestream);

    uint32_t i32_NC_sz = ceil_32(N_C_PER_BLOCK * sizeof(int32_t));
    LocalTensor<int32_t> bits_used = calcBuf.GetWithOffset<int32_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += i32_NC_sz;
    Duplicate(bits_used, 0, N_C_PER_BLOCK);

    LocalTensor<int32_t> bits_used_bcast = calcBuf.GetWithOffset<int32_t>(2 * N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += 2 * i32_NC_sz;
    Duplicate(bits_used_bcast, 0, 2 * N_C_PER_BLOCK);

    LocalTensor<int32_t> channel_offsets = calcBuf.GetWithOffset<int32_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += i32_NC_sz;
    CreateVecIndex(channel_offsets, 0, N_C_PER_BLOCK);
    Muls(channel_offsets, channel_offsets, N_C_PER_BLOCK, N_C_PER_BLOCK);

    LocalTensor<int32_t> tmp_sym = calcBuf.GetWithOffset<int32_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += i32_NC_sz;

    LocalTensor<uint32_t> tmp_1 = calcBuf.GetWithOffset<uint32_t>(2 * N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += 2 * i32_NC_sz;

    LocalTensor<uint32_t> tmp_2 = calcBuf.GetWithOffset<uint32_t>(N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += i32_NC_sz;

    LocalTensor<uint32_t> bit_offsets = calcBuf.GetWithOffset<uint32_t>(2 * N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += 2 * i32_NC_sz;
    CreateVecIndex(bit_offsets.ReinterpretCast<int32_t>(), 0, N_C_PER_BLOCK * 2);
    ShiftRight(bit_offsets, bit_offsets, static_cast<uint32_t>(1), N_C_PER_BLOCK * 2);
    Muls(bit_offsets.ReinterpretCast<int32_t>(), bit_offsets.ReinterpretCast<int32_t>(), static_cast<int32_t>(N_T_MAX * 8), N_C_PER_BLOCK * 2);

    LocalTensor<uint32_t> duplicate_2_gather = calcBuf.GetWithOffset<uint32_t>(2 * N_C_PER_BLOCK, calc_buf_offset);
    calc_buf_offset += 2 * i32_NC_sz;
    CreateVecIndex(duplicate_2_gather.ReinterpretCast<int32_t>(), 0, N_C_PER_BLOCK * 2);
    ShiftRight(duplicate_2_gather, duplicate_2_gather, static_cast<uint32_t>(1), N_C_PER_BLOCK * 2); // in idxs 0, 0, 1, 1, 2, 2, ...
    ShiftLeft(duplicate_2_gather, duplicate_2_gather, static_cast<uint32_t>(2), N_C_PER_BLOCK * 2); // in bytes (x4) 0, 0, 4, 4, 8, 8, ...

    LocalTensor<uint8_t> syms_out = symOutQ.AllocTensor<uint8_t>();

    // --------
    // Phase: Iteratively decode
    // --------
    for (int t_ii = 0; t_ii < n_tokens; ++t_ii) {
        // Read the next byte
        Gather(bits_used_bcast, bits_used, duplicate_2_gather.ReinterpretCast<uint32_t>(), 0, 2 * N_C_PER_BLOCK);
        Add(bit_offsets.ReinterpretCast<int32_t>(), bit_offsets.ReinterpretCast<int32_t>(), bits_used_bcast, 2 * N_C_PER_BLOCK);
        read_unaligned_u8_2_half(
            bytestream,
            bit_offsets,
            curr_dec_val,
            tmp_1,
            tmp_2,
            pows_2,
            N_C_PER_BLOCK
        );

        // Identify the last bin that is less than that byte thus identifying the encoded symbol
        Gather(curr_dec_val_bcast, curr_dec_val, duplicating_gather_idxs.ReinterpretCast<uint32_t>(), 0, bins_per_block);
        Compare(cmp_mask, curr_dec_val_bcast, enc_bins, CMPMODE::LT, bins_per_block);
        ShiftRight(cmp_mask.ReinterpretCast<uint32_t>(), cmp_mask.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(1), bins_per_block / 32);
        Adds(cmp_mask.ReinterpretCast<int32_t>(), cmp_mask.ReinterpretCast<int32_t>(), 1, bins_per_block / 32);
        repeats = N_C_PER_BLOCK / (256 / 32);
        gmp = {
            1,
            static_cast<uint8_t>(1),
            8,
            1
        };

        for(auto repeat_ii = 0; repeat_ii < repeats; ++repeat_ii) {
            GatherMask(tmp_sym[8 * repeat_ii], enc_syms[N_B_MAX * repeat_ii * 8], cmp_mask.ReinterpretCast<uint32_t>()[8 * repeat_ii], true, 256, gmp, _rsvd);
        }

        // Convert to byte (desired output type)
        Cast(tmp_2.ReinterpretCast<half>(), tmp_sym, RoundMode::CAST_NONE, N_C_PER_BLOCK);
        Cast(syms_out[(t_ii * N_C_PER_BLOCK)], tmp_2.ReinterpretCast<half>(), RoundMode::CAST_RINT, N_C_PER_BLOCK);

        // Pick out bits consumed from encode lens
        Add(tmp_sym, tmp_sym, channel_offsets, N_C_PER_BLOCK);
        Muls(tmp_sym, tmp_sym, 4, N_C_PER_BLOCK);
        Gather(bits_used, enc_lens_32, tmp_sym.ReinterpretCast<uint32_t>(), 0, N_C_PER_BLOCK);
    }

    // --------
    // Phase: Copy Out
    // --------
    symOutQ.EnQue(syms_out);
    syms_out = symOutQ.DeQue<uint8_t>();

    uint16_t dst_stride = N_DBs_PER_BLOCK * ((n_channels / N_C_PER_BLOCK) - 1);
    DataCopyParams repeatParams = {static_cast<uint16_t>(n_tokens), N_DBs_PER_BLOCK, 0, dst_stride};
    DataCopy(gm_output_data[(layer_id * n_channels * n_tokens) + channel_start_id], syms_out, repeatParams);

    symOutQ.FreeTensor(syms_out);
}
} // namespace impl
} // namespace pac_coder
} // namespace kvcache_ops

extern "C" __global__ __aicore__ void pac_decode_kernel (
    GM_ADDR meta_data_ptr,
    GM_ADDR cum_lens_ptr,
    GM_ADDR bytestream_ptr,
    GM_ADDR output_data_ptr,
    const uint32_t n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe{};

    kvcache_ops::pac_coder::impl::PacDecoder decoder {
        meta_data_ptr,
        cum_lens_ptr,
        bytestream_ptr,
        output_data_ptr,
        pipe,
        n_tokens,
        n_layers,
        n_channels,
        n_bins};

    int max_work_idx = n_layers * n_channels / kvcache_ops::pac_coder::N_C_PER_BLOCK;

    int32_t coreIdx = AscendC::GetBlockIdx();
    int32_t launchedCores = AscendC::GetBlockNum();

    for (int work_idx = coreIdx; work_idx < max_work_idx; work_idx += launchedCores) {
        int layer_id = work_idx % n_layers;
        int channel_id = kvcache_ops::pac_coder::N_C_PER_BLOCK * (work_idx / n_layers);
        decoder.decode(layer_id, channel_id);
    }
}

namespace kvcache_ops {
namespace pac_coder {

void pac_decode(
    uint8_t* meta_data_ptr,
    uint8_t* cum_lens_ptr,
    uint8_t* bytestream_ptr,
    uint8_t* output_data_ptr,
    void* stream,
    const int n_aiv,
    const int n_bins,
    const int n_tokens,
    const int n_layers,
    const int n_channels) {

    int blockDim = n_layers * (n_channels / N_C_PER_BLOCK) < n_aiv ? n_layers * (n_channels / N_C_PER_BLOCK) : n_aiv;

    pac_decode_kernel<<<blockDim, nullptr, stream>>>(
        meta_data_ptr,
        cum_lens_ptr,
        bytestream_ptr,
        output_data_ptr,
        n_bins,
        n_tokens,
        n_layers,
        n_channels);
}
} // namespace pac_coder
} // namespace kvcache_ops