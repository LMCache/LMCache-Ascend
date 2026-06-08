/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "multi_layer_mem_kernels.h"

__aicore__ inline uint32_t AlignUp32Bytes(uint32_t nbytes) {
    return (nbytes + 31u) & ~31u;
}

__aicore__ inline uint32_t ScratchTokenStrideBytes(uint32_t hdBytes) {
    return (hdBytes % 32u == 0u) ? hdBytes : AlignUp32Bytes(hdBytes);
}

// MP_TRACE_LEVEL: 0=off, 1=path, 2=+stripe, 3=+inner. Call MP_TRACE(level, fmt, ...).
#ifndef MP_TRACE_LEVEL
#define MP_TRACE_LEVEL 0
#endif
#define MP_TRACE(level, fmt, ...)
// #define MP_TRACE(level, fmt, ...) \
//    do { \
//        if constexpr ((level) <= (MP_TRACE_LEVEL)) { \
//            AscendC::PRINTF("[MP_TRACE_%d] " fmt "\n", (level), ##__VA_ARGS__); \
//        } \
//    } while (0)

// KVCacheFormat::MULTI_PLANE_KV (6): fused multi-plane copy in one kernel launch.
// Host entry: csrc/mem_kernels.cpp::multi_layer_kv_transfer_multi_plane.

// Device operator: loops layers (per AI core) × planes × tokens for one chunk transfer.
// Memory layouts (init parameters and stored fields):
//
//  pagedKVCaches: ptrs[layerIdx * numPlanes + planeIdx] -> that plane's buffer.
//  Planes are separate GM allocations (one pointer each), NOT packed in one array.
//  Host shape per plane: (num_blocks, block_size, hidden_dim_bytes).
//
//  block 0 (one page in GM)
//  +-------+-------+-----+-------+
//  | slot0 | slot1 | ... | slotB |
//  +-------+-------+-----+-------+
//              ...
//  block 1 (next page)
//  +-------+-------+-----+-------+
//  | slot0 | slot1 | ... | slotB |
//  +-------+-------+-----+-------+
//  (B = block_size - 1; each slot is hdBytes)
//
//  cacheTensor  (GM_ADDR, LMCache chunk) — planes are contiguous blocks per layer.
//  Logical shape [numLayers, numTokensLmcChunk, lmcChunkLastDimBytes] — last dim is bytes.
//
//  Per-plane contiguous layout (one layer block):
//       +------------------+------------------+-----+------------------+
//       | plane 0          | plane 1          | ... | plane P-1        |
//       | tok0|tok1|...|N  | tok0|tok1|...|N  |     | tok0|tok1|...|N  |
//       | dense hd0/token | dense hd1/token | ... | dense hdP-1 + tail pad each plane |
//       +------------------+------------------+-----+------------------+
//       ^ planeByteOffsets_[0]  [1]                    [P-1]
//
//  Token t of plane p: planeByteOffsets_[p] + t * hd (planeRowStrideBytes_ == hdBytes).
//  Plane payload is contiguous (nTok*hd); tail align_up32(nTok*hd) is unused padding.
//
//  layerBlockBytes_ = sum_p align_up32(hd[p]*numTokensLmcChunk) per layer.
//  lmcChunkLastDimBytes_ from host: ceil(layerBlockBytes_ / numTokensLmcChunk).
//
//  Host / launcher: csrc/mem_kernels.cpp::multi_layer_kv_transfer_multi_plane enqueues once;
//    each AI core loops layerIdx and planeIdx via processPlane().
//
//  perPlaneSlotPtrs[p] + perPlaneSlotStarts[p]: dense int64 slots for plane p
//  perPlaneSlotCounts[p] = numTokensChunk_ for that plane
//
//  Init also sets: page2L_ (true = paged->chunk store), perLoopBuffSize_ / maxTokensPerLoop_
//  (UB chunking), pipe_ + pagedTokenQue_ (local scratch, size perLoopBuffSize_).
//
// Depth-2 UB pipeline (TQueBind VECIN/VECOUT, InitBuffer depth=2):
//   Store: Alloc -> CopyPagedToUb (MTE2) -> EnQue; next iter DeQue prev -> CopyUbToLmcChunk (MTE3) -> Free.
//   Load:  Alloc -> CopyLmcChunkToUb (MTE2) -> EnQue; next iter DeQue prev -> CopyUbToPaged (MTE3) -> Free.
// EnQue/DeQue issue the cross-pipe sync so GM->UB of part i+1 can overlap UB->GM of part i.

class MultiLayerPagedKVCopyV2MultiPlane {
    using scalar_t = int8_t;
    using slot_t = int64_t;
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

    static constexpr int32_t kMaxPlanes = 32;

public:
    __aicore__ inline MultiLayerPagedKVCopyV2MultiPlane() {}

    // Cache host GM pointers, compute per-plane byte offsets in the packed Lmc layer block, and
    // allocate pagedTokenQue_ with depth 2 and slot size perLoopBuffSize_ (one part per slot).
    __aicore__ inline void init(
        GM_ADDR pagedKVCaches, GM_ADDR cacheTensor,
        __gm__ int64_t *perPlaneSlotPtrs, __gm__ int32_t *perPlaneSlotStarts,
        __gm__ int32_t *perPlaneSlotCounts, __gm__ int32_t *perPlaneHdBytes,
        __gm__ int32_t *perPlaneBlockSizes, __gm__ int32_t *perPlanePageBuffSizes,
        __gm__ int32_t *perPlaneLmcRowOffset, const int32_t numPlanes,
        const int32_t numLayers, const int64_t lmcChunkLastDimBytes,
        const int32_t numTokensLmcChunk, const int64_t perLoopBuffSize,
        const int32_t maxTokensPerLoop, const bool page2L, AscendC::TPipe *pipe) {
        this->pipe_ = pipe;
        this->numPlanes_ = numPlanes;
        this->numLayers_ = numLayers;
        this->lmcChunkLastDimBytes_ = lmcChunkLastDimBytes;
        this->numTokensLmcChunk_ = numTokensLmcChunk;
        this->maxTokensPerLoop_ = maxTokensPerLoop;
        this->perLoopBuffSize_ = perLoopBuffSize;
        this->page2L_ = page2L;
        this->kvs_ = numPlanes;
        this->perPlaneHdBytesGm_ = perPlaneHdBytes;
        this->perPlaneBlockSizesGm_ = perPlaneBlockSizes;
        this->perPlanePageBuffSizesGm_ = perPlanePageBuffSizes;
        this->perPlaneSlotPtrsGm_ = perPlaneSlotPtrs;
        this->perPlaneSlotStartsGm_ = perPlaneSlotStarts;
        this->perPlaneSlotCountsGm_ = perPlaneSlotCounts;
        this->perPlaneLmcRowOffsetGm_ = perPlaneLmcRowOffset;

        int64_t prefix = 0;
        for (int32_t p = 0; p < numPlanes && p < kMaxPlanes; ++p) {
            this->planeByteOffsets_[p] = prefix;
            const uint32_t planePayload =
                static_cast<uint32_t>(perPlaneHdBytes[p]) *
                static_cast<uint32_t>(numTokensLmcChunk);
            prefix += static_cast<int64_t>(AlignUp32Bytes(planePayload));
        }
        this->layerBlockBytes_ = prefix;
        this->pipe_->InitBuffer(pagedTokenQue_, 2, this->perLoopBuffSize_);
    }

    // Bind this plane's hd, block size, page buffer capacity, concat slot slice, and Lmc row stride.
    // No branches; must run before processLayerCache / windowed transfers for this plane.
    __aicore__ inline void setPlane(const int32_t planeIdx) {
        this->planeIdx_ = planeIdx;
        this->hiddenDims_ = static_cast<int64_t>(this->perPlaneHdBytesGm_[planeIdx]);
        this->blockSize_ = this->perPlaneBlockSizesGm_[planeIdx];
        this->pageBuffSize_ = static_cast<int64_t>(this->perPlanePageBuffSizesGm_[planeIdx]);
        this->numTokensChunk_ = this->perPlaneSlotCountsGm_[planeIdx];
        this->planeBaseOffsetInLayer_ = this->planeByteOffsets_[planeIdx];
        this->planeLmcRowOffset_ =
            this->perPlaneLmcRowOffsetGm_ != nullptr
                ? static_cast<int32_t>(this->perPlaneLmcRowOffsetGm_[planeIdx])
                : 0;
        this->planeRowStrideBytes_ = this->hiddenDims_;
        this->slotmappingsPlane_ = reinterpret_cast<__gm__ uint8_t *>(
            this->perPlaneSlotPtrsGm_[planeIdx]) +
            static_cast<int64_t>(this->perPlaneSlotStartsGm_[planeIdx]) * sizeof(slot_t);
    }

    // Copy one contiguous byte span from Paged GM into UB at dstByteOff via DataCopyPad.
    // No branches; used as the MTE2 producer step inside the depth-2 pipeline.
    __aicore__ inline void CopyPagedToUb(AscendC::LocalTensor<uint8_t> &dst,
        const int64_t dstByteOff, AscendC::GlobalTensor<uint8_t> &src,
        const int64_t srcByteOff, const uint32_t copyBytes) const {
        const AscendC::DataCopyExtParams params{1u, copyBytes, 0u, 0u, 0u};
        const AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0u, 0u, 0u};
        AscendC::DataCopyPad(dst[dstByteOff], src[srcByteOff], params, pad);
    }

    // Copy one contiguous byte span from UB at srcByteOff into Paged GM via DataCopyPad.
    // No branches; used as the MTE3 consumer step on the load path inside the depth-2 pipeline.
    __aicore__ inline void CopyUbToPaged(AscendC::GlobalTensor<uint8_t> &dst,
        const int64_t dstByteOff, AscendC::LocalTensor<uint8_t> &src,
        const int64_t srcByteOff, const uint32_t copyBytes) const {
        const AscendC::DataCopyExtParams params{1u, copyBytes, 0u, 0u, 0u};
        AscendC::DataCopyPad(dst[dstByteOff], src[srcByteOff], params);
    }

    // Store dense UB scratch into the Lmc chunk row at lmcByteOff for numTokens.
    // Branch: bulk (dense UB + dense Lmc); else bulk_strided (padded UB); else per-token CopyUbToPaged.
    __aicore__ inline void CopyUbToLmcChunk(
        AscendC::GlobalTensor<uint8_t> &lmcGm, const int64_t lmcByteOff,
        AscendC::LocalTensor<uint8_t> &scratchU8, const int32_t numTokens,
        const int64_t hdBytes, const bool denseUbScratch = false,
        const int64_t ubSrcByteOff = 0) {
        const uint32_t blockLen = static_cast<uint32_t>(hdBytes);
        const uint32_t scratchStride =
            denseUbScratch ? blockLen : ScratchTokenStrideBytes(blockLen);
        const int64_t totalBytes = static_cast<int64_t>(numTokens) * hdBytes;
        const int64_t lmcTokenStride = this->planeRowStrideBytes_;

        // Dense UB (scratchStride==hd) and dense GM (planeRowStride==hd): one DataCopyPad
        // block (blockCount=1, blockLen=N*hd, strides 0). See DataCopyExtParams 07_0265.
        if (scratchStride == blockLen && lmcTokenStride == hdBytes) {
            MP_TRACE(1, "CopyUbToLmcChunk bulk N=%d hd=%u", numTokens, blockLen);
            const AscendC::DataCopyExtParams params{
                1u, static_cast<uint32_t>(totalBytes), 0u, 0u, 0u};
            AscendC::DataCopyPad(lmcGm[lmcByteOff], scratchU8[ubSrcByteOff], params);
            return;
        }

        // Padded UB (hd%32!=0, AlignUp32 per token) but dense GM: multi-block VECOUT->GM with
        // srcStride=1 datablock (skip UB pad), dstStride=0 Byte; blockCount<=4095 (07_0265).
        if (numTokens > 1 && scratchStride > blockLen && lmcTokenStride == hdBytes &&
            numTokens <= 4095) {
            MP_TRACE(1, "CopyUbToLmcChunk bulk_strided N=%d hd=%u", numTokens, blockLen);
            const AscendC::DataCopyExtParams params{
                static_cast<uint16_t>(numTokens), blockLen, 1u, 0u, 0u};
            AscendC::DataCopyPad(lmcGm[lmcByteOff], scratchU8[ubSrcByteOff], params);
            return;
        }

        MP_TRACE(1, "CopyUbToLmcChunk per-token copy N=%d", numTokens);
        for (int32_t t = 0; t < numTokens; ++t) {
            CopyUbToPaged(
                lmcGm, lmcByteOff + static_cast<int64_t>(t) * lmcTokenStride, scratchU8,
                ubSrcByteOff + static_cast<int64_t>(t) * static_cast<int64_t>(scratchStride),
                blockLen);
        }
    }

    // Load numTokens from the Lmc chunk row at lmcByteOff into dense UB scratch.
    // Branch: bulk (dense Lmc + dense UB); else bulk_strided (padded UB); else per-token CopyPagedToUb.
    __aicore__ inline void CopyLmcChunkToUb(
        AscendC::LocalTensor<uint8_t> &scratchU8, AscendC::GlobalTensor<uint8_t> &lmcGm,
        const int64_t lmcByteOff, const int32_t numTokens, const int64_t hdBytes,
        const bool denseUbScratch = false) {
        const uint32_t blockLen = static_cast<uint32_t>(hdBytes);
        const uint32_t scratchStride =
            denseUbScratch ? blockLen : ScratchTokenStrideBytes(blockLen);
        const int64_t totalBytes = static_cast<int64_t>(numTokens) * hdBytes;
        const int64_t lmcTokenStride = this->planeRowStrideBytes_;

        // Dense GM and dense UB: one GM->VECIN block (blockCount=1, blockLen=N*hd). 07_0265.
        if (scratchStride == blockLen && lmcTokenStride == hdBytes) {
            MP_TRACE(1, "CopyLmcChunkToUb bulk N=%d hd=%u", numTokens, blockLen);
            const AscendC::DataCopyExtParams params{
                1u, static_cast<uint32_t>(totalBytes), 0u, 0u, 0u};
            const AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0u, 0u, 0u};
            AscendC::DataCopyPad(scratchU8, lmcGm[lmcByteOff], params, pad);
            return;
        }

        // Dense GM, padded UB: multi-block GM->VECIN with srcStride=0 Byte, dstStride=1
        // datablock; blockCount<=4095. Multi-block dense UB with blockLen%32!=0 is invalid.
        if (numTokens > 1 && scratchStride > blockLen && lmcTokenStride == hdBytes &&
            numTokens <= 4095) {
            MP_TRACE(1, "CopyLmcChunkToUb bulk_strided N=%d hd=%u", numTokens, blockLen);
            const AscendC::DataCopyExtParams params{
                static_cast<uint16_t>(numTokens), blockLen, 0u, 1u, 0u};
            const AscendC::DataCopyPadExtParams<uint8_t> pad{
                false, 0u, 0u, static_cast<uint8_t>(0)};
            AscendC::DataCopyPad(scratchU8, lmcGm[lmcByteOff], params, pad);
            return;
        }

        MP_TRACE(1, "CopyLmcChunkToUb per-token copy N=%d", numTokens);
        for (int32_t t = 0; t < numTokens; ++t) {
            CopyPagedToUb(
                scratchU8, static_cast<int64_t>(t) * static_cast<int64_t>(scratchStride),
                lmcGm, lmcByteOff + static_cast<int64_t>(t) * lmcTokenStride, blockLen);
        }
    }

    // Split nTokens at paged-block boundaries using only slotStart and blockSize (no slot scan).
    // Outputs nFirst (tail of first block), nFullBlocks full blocks, nLast tail, and part count.
    __aicore__ inline void splitPagedBlockParts(const int64_t slotStart, const int32_t nTokens,
        const int32_t bs, int32_t &nFirst, int32_t &nFullBlocks, int32_t &nLast,
        int32_t &nPagedBlockParts) const {
        const int32_t intra0 = static_cast<int32_t>(slotStart % static_cast<int64_t>(bs));
        nFirst = min(bs - intra0, nTokens);
        const int32_t remain = nTokens - nFirst;
        nFullBlocks = remain / bs;
        nLast = remain % bs;
        nPagedBlockParts = 1 + nFullBlocks + (nLast > 0 ? 1 : 0);
    }

    // Return token count for block-part index blockPartIdx given split outputs from splitPagedBlockParts.
    // No branches beyond part-index selection (first / full / last).
    __aicore__ inline int32_t pagedBlockPartNumTokens(const int32_t blockPartIdx,
        const int32_t nFirst, const int32_t nFullBlocks, const int32_t nLast,
        const int32_t bs) const {
        if (blockPartIdx == 0) {
            return nFirst;
        }
        if (blockPartIdx <= nFullBlocks) {
            return bs;
        }
        return nLast;
    }

    // Scalar byte copy for one block-part when bulk cannot run (invalid lead slot or part > UB).
    // Skips slots < 0 or >= pageBuffSize_; page2L selects Paged->Lmc vs Lmc->Paged per byte.
    // Only used as safety net. In practice is never taken. I hope. (marco)
    __aicore__ inline void copyBlockSetValue(
        __gm__ slot_t *slotmappingPtr, const int32_t tokenOff, const int32_t nTokens,
        const int32_t layerIdx, const bool page2L, AscendC::GlobalTensor<uint8_t> &pagedGm,
        AscendC::GlobalTensor<uint8_t> &lmcGm, const int64_t hdBytes,
        const int64_t blockBytesI64) {
        const int64_t bsz64 = static_cast<int64_t>(this->blockSize_);
        MP_TRACE(1, "copyBlock SetValue nTok=%d off=%d", nTokens, tokenOff);
        for (int32_t t = 0; t < nTokens; ++t) {
            const int64_t slot = static_cast<int64_t>(slotmappingPtr[tokenOff + t]);
            if (slot < 0 || slot >= this->pageBuffSize_) {
                continue;
            }
            const int64_t pagedByteStart =
                (slot / bsz64) * blockBytesI64 + (slot % bsz64) * hdBytes;
            const int64_t lmcByteOff =
                static_cast<int64_t>(layerIdx) * this->layerBlockBytes_ +
                this->planeBaseOffsetInLayer_ +
                static_cast<int64_t>(this->planeLmcRowOffset_ + tokenOff + t) *
                    this->planeRowStrideBytes_;
            for (int64_t b = 0; b < hdBytes; ++b) {
                if (page2L) {
                    lmcGm.SetValue(lmcByteOff + b, pagedGm.GetValue(pagedByteStart + b));
                } else {
                    pagedGm.SetValue(pagedByteStart + b, lmcGm.GetValue(lmcByteOff + b));
                }
            }
        }
    }

    // Depth-2 consumer on store path: DeQue prior EnQue'd UB, copy dense part into Lmc, Free UB slot.
    // No branches; caller must pass tokenOff matching the EnQue'd producer part.
    __aicore__ inline void flushPendingUbPartToLmc(
        AscendC::GlobalTensor<uint8_t> &lmcGm, const int32_t layerIdx,
        const int32_t tokenOff, const int32_t partNumTokens, const int64_t hdBytes) {
        local_scalar_t buf = this->pagedTokenQue_.template DeQue<scalar_t>();
        AscendC::LocalTensor<uint8_t> scratchU8 = buf.template ReinterpretCast<uint8_t>();
        const int64_t lmcByteOff =
            static_cast<int64_t>(layerIdx) * this->layerBlockBytes_ +
            this->planeBaseOffsetInLayer_ +
            static_cast<int64_t>(this->planeLmcRowOffset_ + tokenOff) *
                this->planeRowStrideBytes_;
        CopyUbToLmcChunk(lmcGm, lmcByteOff, scratchU8, partNumTokens, hdBytes, true, 0);
        this->pagedTokenQue_.FreeTensor(buf);
    }

    // Depth-2 consumer on load path: DeQue prior EnQue'd UB, copy dense part into Paged, Free UB slot.
    // No branches; pagedByteStart must match the lead slot used when the part was loaded into UB.
    __aicore__ inline void flushPendingUbPartToPaged(
        AscendC::GlobalTensor<uint8_t> &pagedGm, const int64_t pagedByteStart,
        const int32_t partNumTokens, const int64_t hdBytes) {
        local_scalar_t buf = this->pagedTokenQue_.template DeQue<scalar_t>();
        AscendC::LocalTensor<uint8_t> scratchU8 = buf.template ReinterpretCast<uint8_t>();
        const uint32_t partBytesU32 =
            static_cast<uint32_t>(static_cast<int64_t>(partNumTokens) * hdBytes);
        CopyUbToPaged(pagedGm, pagedByteStart, scratchU8, 0, partBytesU32);
        this->pagedTokenQue_.FreeTensor(buf);
    }

    // Copy one paged-block part on the blockwise (hd<32) path using the depth-2 queue when bulk fits UB.
    // Branch bulk: store EnQue Paged->UB then flush prev to Lmc, or load EnQue Lmc->UB then flush prev to
    // Paged; branch !bulk: drain pending then copyBlockSetValue (no UB pipeline for this part).
    __aicore__ inline void copyBlock(
        const int32_t layerIdx, const int32_t tokenOff, const int32_t nTokens,
        const bool page2L, __gm__ slot_t *slotmappingPtr,
        AscendC::GlobalTensor<uint8_t> &pagedGm, AscendC::GlobalTensor<uint8_t> &lmcGm,
        const int64_t hdBytes, const int64_t blockBytesI64, const bool ubPending,
        const int32_t pendingTokenOff, const int64_t pendingPagedByteStart,
        const int32_t pendingNumTokens, bool &outUbPending, int32_t &outPendingTokenOff,
        int64_t &outPendingPagedByteStart, int32_t &outPendingNumTokens) {
        outUbPending = ubPending;
        outPendingTokenOff = pendingTokenOff;
        outPendingPagedByteStart = pendingPagedByteStart;
        outPendingNumTokens = pendingNumTokens;
        if (nTokens <= 0) {
            return;
        }
        const int64_t currentSlot = static_cast<int64_t>(slotmappingPtr[tokenOff]);
        const int64_t copyBytesI64 = static_cast<int64_t>(nTokens) * hdBytes;
        // Pipeline bulk: valid lead slot, non-empty part, nTok*hd fits one UB scratch (perLoopBuffSize_).
        const bool useBulk = currentSlot >= 0 && currentSlot < this->pageBuffSize_ &&
            copyBytesI64 > 0LL && copyBytesI64 <= this->perLoopBuffSize_;
        // Fallback: drain depth-2 queue then scalar SetValue (invalid slot or part larger than UB).
        if (!useBulk) {
            if (outUbPending) {
                if (page2L) {
                    flushPendingUbPartToLmc(
                        lmcGm, layerIdx, outPendingTokenOff, outPendingNumTokens, hdBytes);
                } else {
                    flushPendingUbPartToPaged(
                        pagedGm, outPendingPagedByteStart, outPendingNumTokens, hdBytes);
                }
                outUbPending = false;
            }
            this->copyBlockSetValue(
                slotmappingPtr, tokenOff, nTokens, layerIdx, page2L, pagedGm, lmcGm, hdBytes,
                blockBytesI64);
            return;
        }
        const uint32_t copyBytesU32 = static_cast<uint32_t>(copyBytesI64);
        const int64_t bsz64 = static_cast<int64_t>(this->blockSize_);
        const int64_t pagedByteStart =
            (currentSlot / bsz64) * blockBytesI64 + (currentSlot % bsz64) * hdBytes;

        MP_TRACE(1, "copyBlock bulk nTok=%d off=%d slot=%lld copyBytes=%u", nTokens, tokenOff,
            currentSlot, copyBytesU32);
        local_scalar_t scratch = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        AscendC::LocalTensor<uint8_t> scratchU8 = scratch.template ReinterpretCast<uint8_t>();
        if (page2L) {
            CopyPagedToUb(scratchU8, 0, pagedGm, pagedByteStart, copyBytesU32);
            this->pagedTokenQue_.EnQue(scratch);
            if (outUbPending) {
                flushPendingUbPartToLmc(
                    lmcGm, layerIdx, outPendingTokenOff, outPendingNumTokens, hdBytes);
            }
            outUbPending = true;
            outPendingTokenOff = tokenOff;
            outPendingNumTokens = nTokens;
            return;
        }
        const int64_t lmcByteOff =
            static_cast<int64_t>(layerIdx) * this->layerBlockBytes_ +
            this->planeBaseOffsetInLayer_ +
            static_cast<int64_t>(this->planeLmcRowOffset_ + tokenOff) *
                this->planeRowStrideBytes_;
        CopyLmcChunkToUb(scratchU8, lmcGm, lmcByteOff, nTokens, hdBytes, true);
        this->pagedTokenQue_.EnQue(scratch);
        if (outUbPending) {
            flushPendingUbPartToPaged(
                pagedGm, outPendingPagedByteStart, outPendingNumTokens, hdBytes);
        }
        outUbPending = true;
        outPendingTokenOff = tokenOff;
        outPendingPagedByteStart = pagedByteStart;
        outPendingNumTokens = nTokens;
    }

    // Entire-chunk path when hd<32 and one physical block fits in UB: split by block geometry, then
    // pipelined copyBlock per part (store or load per page2L). Early exit if lead slot invalid.
    __aicore__ inline void blockwiseCopy(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t *cacheTensor,
        __gm__ uint8_t *slotmappings, const int32_t layerIdx, const bool page2L) {
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t *>(slotmappings);
        const int64_t hdBytes = this->hiddenDims_;
        const int32_t numTokens = this->numTokensChunk_;
        const int32_t bs = this->blockSize_;
        const int64_t blockBytesI64 = static_cast<int64_t>(bs) * hdBytes;

        if (numTokens <= 0) {
            return;
        }

        __gm__ uint8_t *__gm__ *ptrs =
            reinterpret_cast<__gm__ uint8_t *__gm__ *>(pagedKVCaches);
        __gm__ uint8_t *pagedLayer = ptrs[layerIdx * this->kvs_ + this->planeIdx_];
        const int64_t pagedPlaneByteElems = this->pageBuffSize_ * hdBytes;
        this->pagedTokenGlobalU8_.SetGlobalBuffer(pagedLayer, pagedPlaneByteElems);

        const int64_t slot0 = static_cast<int64_t>(slotmappingPtr[0]);
        if (slot0 < 0 || slot0 >= this->pageBuffSize_) {
            MP_TRACE(2, "blockwiseCopy skip invalid slot0=%lld pageBuff=%lld", slot0,
                this->pageBuffSize_);
            return;
        }

        AscendC::GlobalTensor<uint8_t> pagedGm;
        pagedGm.SetGlobalBuffer(pagedLayer, pagedPlaneByteElems);

        const int64_t lmcChunkElems =
            static_cast<int64_t>(this->numLayers_) * static_cast<int64_t>(this->numTokensLmcChunk_) *
            this->lmcChunkLastDimBytes_;
        AscendC::GlobalTensor<uint8_t> lmcGm;
        lmcGm.SetGlobalBuffer(cacheTensor, lmcChunkElems);

        int32_t nFirst = 0;
        int32_t nFullBlocks = 0;
        int32_t nLast = 0;
        int32_t nPagedBlockParts = 0;
        this->splitPagedBlockParts(slot0, numTokens, bs, nFirst, nFullBlocks, nLast, nPagedBlockParts);

        MP_TRACE(2, "blockwiseCopy n=%d nFirst=%d nFull=%d nLast=%d slot0=%lld", numTokens, nFirst,
            nFullBlocks, nLast, slot0);

        bool ubPending = false;
        int32_t pendingTokenOff = 0;
        int64_t pendingPagedByteStart = 0;
        int32_t pendingNumTokens = 0;
        int32_t tokenOff = 0;
        for (int32_t blockPartIdx = 0; blockPartIdx < nPagedBlockParts; ++blockPartIdx) {
            const int32_t blockPartNumTokens =
                this->pagedBlockPartNumTokens(blockPartIdx, nFirst, nFullBlocks, nLast, bs);
            this->copyBlock(layerIdx, tokenOff, blockPartNumTokens, page2L, slotmappingPtr, pagedGm,
                lmcGm, hdBytes, blockBytesI64, ubPending, pendingTokenOff, pendingPagedByteStart,
                pendingNumTokens, ubPending, pendingTokenOff, pendingPagedByteStart, pendingNumTokens);
            tokenOff += blockPartNumTokens;
        }
        if (ubPending) {
            if (page2L) {
                flushPendingUbPartToLmc(
                    lmcGm, layerIdx, pendingTokenOff, pendingNumTokens, hdBytes);
            } else {
                flushPendingUbPartToPaged(
                    pagedGm, pendingPagedByteStart, pendingNumTokens, hdBytes);
            }
        }
    }

    // Store one UB window: per block-part, pipeline Paged->UB (EnQue) with UB->Lmc (DeQue prev).
    // Branch bulk: EnQue after CopyPagedToUb; branch invalid lead: drain pending then SetValue only.
    __aicore__ inline void _page2LTransfer(__gm__ uint8_t *cacheTensor, __gm__ uint8_t *slotmappings,
        const int32_t layerIdx, const int32_t startTokensIdx, const int32_t endTokensIdx) {
        const int32_t numTokens = endTokensIdx - startTokensIdx;
        if (numTokens <= 0) {
            return;
        }
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t *>(slotmappings);
        const int64_t hdBytes = this->hiddenDims_;
        const int32_t bs = this->blockSize_;
        const int64_t slot0 = static_cast<int64_t>(slotmappingPtr[startTokensIdx]);
        int32_t nFirst = 0;
        int32_t nFullBlocks = 0;
        int32_t nLast = 0;
        int32_t nPagedBlockParts = 0;
        this->splitPagedBlockParts(slot0, numTokens, bs, nFirst, nFullBlocks, nLast, nPagedBlockParts);

        const int64_t lmcChunkElems =
            static_cast<int64_t>(this->numLayers_) * static_cast<int64_t>(this->numTokensLmcChunk_) *
            this->lmcChunkLastDimBytes_;
        AscendC::GlobalTensor<uint8_t> lmcU8;
        lmcU8.SetGlobalBuffer(cacheTensor, lmcChunkElems);

        const int64_t blockBytesI64 = static_cast<int64_t>(bs) * hdBytes;
        bool ubPending = false;
        int32_t pendingTokenOff = 0;
        int32_t pendingNumTokens = 0;
        int32_t tokenOffInWindow = 0;
        for (int32_t blockPartIdx = 0; blockPartIdx < nPagedBlockParts; ++blockPartIdx) {
            const int32_t partNumTokens =
                this->pagedBlockPartNumTokens(blockPartIdx, nFirst, nFullBlocks, nLast, bs);
            if (partNumTokens <= 0) {
                continue;
            }
            const int32_t tokenOff = startTokensIdx + tokenOffInWindow;
            const int64_t partSlot = static_cast<int64_t>(slotmappingPtr[tokenOff]);
            const int64_t partBytesI64 = static_cast<int64_t>(partNumTokens) * hdBytes;
            const bool slotOk = partSlot >= 0 && partSlot < this->pageBuffSize_;
            // EnQue/DeQue bulk: valid part lead slot and part payload fits UB scratch buffer.
            const bool bulkPagedToUb =
                slotOk && partBytesI64 > 0LL && partBytesI64 <= this->perLoopBuffSize_;

            if (bulkPagedToUb) {
                const int64_t pagedByteStart = partSlot * hdBytes;
                const uint32_t partBytesU32 = static_cast<uint32_t>(partBytesI64);
                local_scalar_t scratch = this->pagedTokenQue_.template AllocTensor<scalar_t>();
                AscendC::LocalTensor<uint8_t> scratchU8 = scratch.template ReinterpretCast<uint8_t>();
                CopyPagedToUb(scratchU8, 0, this->pagedTokenGlobalU8_, pagedByteStart, partBytesU32);
                this->pagedTokenQue_.EnQue(scratch);
                MP_TRACE(1, "_page2LTransfer bulk part nTok=%d off=%d slot=%lld", partNumTokens,
                    tokenOff, partSlot);
                if (ubPending) {
                    flushPendingUbPartToLmc(
                        lmcU8, layerIdx, pendingTokenOff, pendingNumTokens, hdBytes);
                }
                ubPending = true;
                pendingTokenOff = tokenOff;
                pendingNumTokens = partNumTokens;
            } else {
                if (ubPending) {
                    flushPendingUbPartToLmc(
                        lmcU8, layerIdx, pendingTokenOff, pendingNumTokens, hdBytes);
                    ubPending = false;
                }
                MP_TRACE(1, "_page2LTransfer SetValue part nTok=%d off=%d", partNumTokens, tokenOff);
                this->copyBlockSetValue(
                    slotmappingPtr, tokenOff, partNumTokens, layerIdx, true, this->pagedTokenGlobalU8_,
                    lmcU8, hdBytes, blockBytesI64);
            }
            tokenOffInWindow += partNumTokens;
        }
        if (ubPending) {
            flushPendingUbPartToLmc(
                lmcU8, layerIdx, pendingTokenOff, pendingNumTokens, hdBytes);
        }
    }

    // Load one UB window: per block-part, pipeline Lmc->UB (EnQue) with UB->Paged (DeQue prev).
    // Branch bulk: EnQue after CopyLmcChunkToUb; branch invalid lead: drain pending then SetValue only.
    __aicore__ inline void _L2PageTransfer(__gm__ uint8_t *cacheTensor, __gm__ uint8_t *slotmappings,
        const int32_t layerIdx, const int32_t startTokensIdx, const int32_t endTokensIdx) {
        const int32_t numTokens = endTokensIdx - startTokensIdx;
        if (numTokens <= 0) {
            return;
        }
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t *>(slotmappings);
        const int64_t hdBytes = this->hiddenDims_;
        const int32_t bs = this->blockSize_;
        const int64_t slot0 = static_cast<int64_t>(slotmappingPtr[startTokensIdx]);
        int32_t nFirst = 0;
        int32_t nFullBlocks = 0;
        int32_t nLast = 0;
        int32_t nPagedBlockParts = 0;
        this->splitPagedBlockParts(slot0, numTokens, bs, nFirst, nFullBlocks, nLast, nPagedBlockParts);

        const int64_t lmcChunkElems =
            static_cast<int64_t>(this->numLayers_) * static_cast<int64_t>(this->numTokensLmcChunk_) *
            this->lmcChunkLastDimBytes_;
        AscendC::GlobalTensor<uint8_t> lmcU8;
        lmcU8.SetGlobalBuffer(cacheTensor, lmcChunkElems);

        const int64_t blockBytesI64 = static_cast<int64_t>(bs) * hdBytes;
        bool ubPending = false;
        int32_t pendingTokenOff = 0;
        int64_t pendingPagedByteStart = 0;
        int32_t pendingNumTokens = 0;
        int32_t tokenOffInWindow = 0;
        for (int32_t blockPartIdx = 0; blockPartIdx < nPagedBlockParts; ++blockPartIdx) {
            const int32_t partNumTokens =
                this->pagedBlockPartNumTokens(blockPartIdx, nFirst, nFullBlocks, nLast, bs);
            if (partNumTokens <= 0) {
                continue;
            }
            const int32_t tokenOff = startTokensIdx + tokenOffInWindow;
            const int64_t partSlot = static_cast<int64_t>(slotmappingPtr[tokenOff]);
            const int64_t partBytesI64 = static_cast<int64_t>(partNumTokens) * hdBytes;
            const bool slotOk = partSlot >= 0 && partSlot < this->pageBuffSize_;
            // Same UB-size gate as store; load pipelines Lmc->UB then flushes prior UB->Paged.
            const bool bulkLmcToUb =
                slotOk && partBytesI64 > 0LL && partBytesI64 <= this->perLoopBuffSize_;

            if (bulkLmcToUb) {
                const int64_t pagedByteStart = partSlot * hdBytes;
                const int64_t lmcByteOff =
                    static_cast<int64_t>(layerIdx) * this->layerBlockBytes_ +
                    this->planeBaseOffsetInLayer_ +
                    static_cast<int64_t>(this->planeLmcRowOffset_ + tokenOff) *
                        this->planeRowStrideBytes_;
                local_scalar_t scratch = this->pagedTokenQue_.template AllocTensor<scalar_t>();
                AscendC::LocalTensor<uint8_t> scratchU8 = scratch.template ReinterpretCast<uint8_t>();
                CopyLmcChunkToUb(scratchU8, lmcU8, lmcByteOff, partNumTokens, hdBytes, true);
                this->pagedTokenQue_.EnQue(scratch);
                MP_TRACE(1, "_L2PageTransfer bulk part nTok=%d off=%d slot=%lld", partNumTokens,
                    tokenOff, partSlot);
                if (ubPending) {
                    flushPendingUbPartToPaged(
                        this->pagedTokenGlobalU8_, pendingPagedByteStart, pendingNumTokens, hdBytes);
                }
                ubPending = true;
                pendingTokenOff = tokenOff;
                pendingPagedByteStart = pagedByteStart;
                pendingNumTokens = partNumTokens;
            } else {
                if (ubPending) {
                    flushPendingUbPartToPaged(
                        this->pagedTokenGlobalU8_, pendingPagedByteStart, pendingNumTokens, hdBytes);
                    ubPending = false;
                }
                MP_TRACE(1, "_L2PageTransfer SetValue part nTok=%d off=%d", partNumTokens, tokenOff);
                this->copyBlockSetValue(
                    slotmappingPtr, tokenOff, partNumTokens, layerIdx, false, this->pagedTokenGlobalU8_,
                    lmcU8, hdBytes, blockBytesI64);
            }
            tokenOffInWindow += partNumTokens;
        }
        if (ubPending) {
            flushPendingUbPartToPaged(
                this->pagedTokenGlobalU8_, pendingPagedByteStart, pendingNumTokens, hdBytes);
        }
    }

    // Dispatch one (layer, plane): blockwiseCopy when hd<32 and block fits UB; else windowed loops.
    // Windowed branch calls _page2LTransfer (store) or _L2PageTransfer (load) per maxTokensPerLoop_ slice.
    __aicore__ inline void processLayerCache(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t *cacheTensor,
        __gm__ uint8_t *slotmappings, const int32_t layerIdx, const bool page2L) {
        const int64_t hdBytes = this->hiddenDims_;
        const int64_t blockBytes = static_cast<int64_t>(this->blockSize_) * hdBytes;
        MP_TRACE(1, "layer hd=%lld bs=%d bb=%lld ub=%lld", hdBytes, this->blockSize_, blockBytes,
            this->perLoopBuffSize_);
        // blockwiseCopy: hd<32, non-zero block, one paged block 32B-aligned and fits UB (else windowed).
        if (hdBytes < 32 && this->blockSize_ > 0 && blockBytes > 0 && (blockBytes % 32LL) == 0 &&
            blockBytes <= this->perLoopBuffSize_) {
            MP_TRACE(1, "blockwiseCopy for this plane");
            this->blockwiseCopy(pagedKVCaches, cacheTensor, slotmappings, layerIdx, page2L);
            return;
        }
        MP_TRACE(1, "Taking windowed path for this plane.");
        __gm__ uint8_t *__gm__ *pagedKVCachesPtr =
            reinterpret_cast<__gm__ uint8_t *__gm__ *>(pagedKVCaches);
        __gm__ uint8_t *pagedLayerKVCaches =
            pagedKVCachesPtr[layerIdx * this->kvs_ + this->planeIdx_];
        const int64_t pagedByteElems = this->pageBuffSize_ * hdBytes;
        this->pagedTokenGlobalU8_.SetGlobalBuffer(pagedLayerKVCaches, pagedByteElems);

        for (int32_t startTokensIdx = 0; startTokensIdx < this->numTokensChunk_;
            startTokensIdx += this->maxTokensPerLoop_) {
            int32_t endTokensIdx = startTokensIdx + this->maxTokensPerLoop_;
            endTokensIdx = min(endTokensIdx, this->numTokensChunk_);
            if (page2L) {
                this->_page2LTransfer(cacheTensor, slotmappings, layerIdx, startTokensIdx, endTokensIdx);
            } else {
                this->_L2PageTransfer(cacheTensor, slotmappings, layerIdx, startTokensIdx, endTokensIdx);
            }
        }
    }

    // setPlane(planeIdx) then processLayerCache for that plane within the current layer loop.
    __aicore__ inline void processPlane(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t *cacheTensor,
        const int32_t planeIdx, const int32_t layerIdx, const bool page2L) {
        this->setPlane(planeIdx);
        MP_TRACE(1, "plane=%d nTok=%d hd=%lld", planeIdx, this->numTokensChunk_, this->hiddenDims_);
        this->processLayerCache(
            pagedKVCaches, cacheTensor, this->slotmappingsPlane_, layerIdx, page2L);
    }

private:
    AscendC::TPipe *pipe_;
    // Depth-2 scratch queue: VECIN producer (GM->UB), VECOUT consumer (UB->GM); see class header.
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> pagedTokenQue_;
    AscendC::GlobalTensor<uint8_t> pagedTokenGlobalU8_;

    __gm__ int32_t *perPlaneHdBytesGm_{nullptr};
    __gm__ int32_t *perPlaneBlockSizesGm_{nullptr};
    __gm__ int32_t *perPlanePageBuffSizesGm_{nullptr};
    __gm__ int64_t *perPlaneSlotPtrsGm_{nullptr};
    __gm__ int32_t *perPlaneSlotStartsGm_{nullptr};
    __gm__ int32_t *perPlaneSlotCountsGm_{nullptr};
    __gm__ int32_t *perPlaneLmcRowOffsetGm_{nullptr};
    __gm__ uint8_t *slotmappingsPlane_{nullptr};

    int64_t planeByteOffsets_[kMaxPlanes];
    int32_t numPlanes_{0};
    int32_t numLayers_{0};
    int32_t planeIdx_{0};
    int64_t pageBuffSize_{0};
    int64_t hiddenDims_{0};
    int32_t numTokensChunk_{0};
    int32_t numTokensLmcChunk_{0};
    int32_t maxTokensPerLoop_{0};
    int64_t perLoopBuffSize_{0};
    int32_t kvs_{0};
    bool page2L_{false};
    int64_t lmcChunkLastDimBytes_{0};
    int64_t layerBlockBytes_{0};
    int64_t planeBaseOffsetInLayer_{0};
    int32_t planeLmcRowOffset_{0};
    int64_t planeRowStrideBytes_{0};
    int32_t blockSize_{0};
};

extern "C" __global__ __aicore__ void multi_layer_paged_kv_copy_v2_multi_plane_int8_t_int64_t(
    GM_ADDR pagedKVCaches, GM_ADDR dstCacheTensor, GM_ADDR perPlaneSlotPtrs,
    GM_ADDR perPlaneSlotStarts, GM_ADDR perPlaneSlotCounts, GM_ADDR perPlaneHdBytes,
    GM_ADDR perPlaneBlockSizes, GM_ADDR perPlanePageBuffSizes, GM_ADDR perPlaneLmcRowOffset,
    const int32_t numPlanes, const int32_t numLayers, const int64_t lmcChunkLastDimBytes,
    const int32_t numTokensLmcChunk, const int64_t perLoopBuffer, const int32_t maxTokensPerLoop,
    const bool page2L) {
    AscendC::TPipe pipe;
    MultiLayerPagedKVCopyV2MultiPlane op{};
    const int32_t bIdx = AscendC::GetBlockIdx();
    const int32_t launchedCores = AscendC::GetBlockNum();
    const int32_t layersPerCore = (numLayers + launchedCores - 1) / launchedCores;
    const int32_t startLayersIdx = bIdx * layersPerCore;
    const int32_t endLayersIdx = min(numLayers, startLayersIdx + layersPerCore);
    op.init(pagedKVCaches, dstCacheTensor,
        reinterpret_cast<__gm__ int64_t *>(perPlaneSlotPtrs),
        reinterpret_cast<__gm__ int32_t *>(perPlaneSlotStarts),
        reinterpret_cast<__gm__ int32_t *>(perPlaneSlotCounts),
        reinterpret_cast<__gm__ int32_t *>(perPlaneHdBytes),
        reinterpret_cast<__gm__ int32_t *>(perPlaneBlockSizes),
        reinterpret_cast<__gm__ int32_t *>(perPlanePageBuffSizes),
        reinterpret_cast<__gm__ int32_t *>(perPlaneLmcRowOffset), numPlanes, numLayers,
        lmcChunkLastDimBytes, numTokensLmcChunk, perLoopBuffer, maxTokensPerLoop, page2L, &pipe);
    for (int32_t layerIdx = startLayersIdx; layerIdx < endLayersIdx; layerIdx++) {
        for (int32_t planeIdx = 0; planeIdx < numPlanes; planeIdx++) {
            op.processPlane(pagedKVCaches, dstCacheTensor, planeIdx, layerIdx, page2L);
        }
    }
}

namespace kvcache_ops {

extern void multi_layer_kv_transfer_multi_plane_kernel_v2(
    uint32_t blockDim, void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor,
    int64_t *perPlaneSlotPtrs, int32_t *perPlaneSlotStarts, int32_t *perPlaneSlotCounts,
    int32_t *perPlaneHdBytes, int32_t *perPlaneBlockSizes, int32_t *perPlanePageBuffSizes,
    int32_t *perPlaneLmcRowOffset, int32_t numPlanes, int32_t numLayers,
    int64_t lmcChunkLastDimBytes, int32_t numTokensLmcChunk, int64_t perLoopBuffer,
    int32_t maxTokensPerLoop, bool page2L) {
    multi_layer_paged_kv_copy_v2_multi_plane_int8_t_int64_t<<<blockDim, nullptr, stream>>>(
        pagedKVCaches, dstCacheTensor, perPlaneSlotPtrs, perPlaneSlotStarts,
        perPlaneSlotCounts, perPlaneHdBytes, perPlaneBlockSizes, perPlanePageBuffSizes,
        perPlaneLmcRowOffset, numPlanes, numLayers, lmcChunkLastDimBytes, numTokensLmcChunk,
        perLoopBuffer, maxTokensPerLoop, page2L);
}

} // namespace kvcache_ops
