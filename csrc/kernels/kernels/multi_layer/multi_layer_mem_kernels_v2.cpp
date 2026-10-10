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
#include <stdexcept>
#include <string>

template <typename scalar_t, typename slot_t, kvcache_ops::KVCacheFormat kvcache_fmt> 
class MultiLayerPagedKVCopyV2 {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline MultiLayerPagedKVCopyV2() {}

    __aicore__ inline void init(GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, GM_ADDR slotmappings,
                                const int64_t hiddenDims, const int32_t numLayers, const int64_t pageBuffSize,
                                const int32_t numTokensChunk, const int64_t perLoopBuffSize,
                                const int32_t maxTokensPerLoop, const bool page2L, AscendC::TPipe *pipe,
                                const int64_t kHiddenDims = 0, const int64_t vHiddenDims = 0,
                                const int64_t dsaHiddenDims = 0)
    {
        this->pipe_ = pipe;
        this->numLayers_ = numLayers;
        this->hiddenDims_ = hiddenDims;
        this->pageBuffSize_ = pageBuffSize;
        this->numTokensChunk_ = numTokensChunk;
        this->maxTokensPerLoop_ = maxTokensPerLoop;
        this->perLoopBuffSize_ = perLoopBuffSize;
        this->page2L_ = page2L;
        this->valid_ = true;
        
        // For MLA_KV and DSA_KV, store different hidden_dims
        if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::MLA_KV || 
                      kvcache_fmt == kvcache_ops::KVCacheFormat::DSA_KV) {
            this->kHiddenDims_ = kHiddenDims;
            this->vHiddenDims_ = vHiddenDims;
            this->dsaHiddenDims_ = dsaHiddenDims;
        }
        
        // we assume this is taken care of in the kernel launch.
        this->pipe_->InitBuffer(pagedTokenQue_, 2, this->perLoopBuffSize_);
    }

    __aicore__ inline int64_t GetHiddenDims(const int cacheIdx) {
        if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::MLA_KV) {
            return (cacheIdx == 0) ? this->kHiddenDims_ : this->vHiddenDims_;
        } else if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::DSA_KV) {
            if (cacheIdx == 0) return this->kHiddenDims_;
            else if (cacheIdx == 1) return this->vHiddenDims_;
            else return this->dsaHiddenDims_;
        } else {
            return this->hiddenDims_;
        }
    }

    __aicore__ inline int64_t GetLMCBaseOffset(const int cacheIdx) {
        if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::MLA_KV) {
            if (cacheIdx == 0) return 0;
            else return this->numLayers_ * this->numTokensChunk_ * this->kHiddenDims_;
        } else if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::DSA_KV) {
            if (cacheIdx == 0) return 0;
            else if (cacheIdx == 1) return this->numLayers_ * this->numTokensChunk_ * this->kHiddenDims_;
            else return this->numLayers_ * this->numTokensChunk_ * (this->kHiddenDims_ + this->vHiddenDims_);
        } else {
            return static_cast<int64_t>(cacheIdx) * this->numLayers_ * this->numTokensChunk_ * this->hiddenDims_;
        }
    }

    __aicore__ inline void _page2LTransfer(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor, 
                                           __gm__ uint8_t *slotmappings, const int cacheIdx, 
                                           const int layerIdx, const int32_t startTokensIdx, 
                                           const int32_t endTokensIdx, 
                                           const int32_t actualTokensPerInnerLoop,
                                           const int64_t pagedOffset) {
        // get the slotmappings 
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        
        // Get the correct hidden_dims for this cacheIdx
        int64_t hiddenDims = GetHiddenDims(cacheIdx);
        
        // 1. alloc per layer per loop cache buffer
        local_scalar_t perLayerSingleCacheBuffer = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        int64_t slot;      
        int64_t tmpPagedOffset;
        int64_t localTensorTokenOffset;
        // 2. copy num tokens
        for (int64_t tokenIdx = startTokensIdx; tokenIdx < endTokensIdx; tokenIdx++) {
            slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);
            tmpPagedOffset = pagedOffset + slot * hiddenDims;
            localTensorTokenOffset = (tokenIdx - startTokensIdx) * hiddenDims;
            AscendC::DataCopy(perLayerSingleCacheBuffer[localTensorTokenOffset], this->pagedTokenGlobal_[tmpPagedOffset],
                              hiddenDims);
        }

        // 3. enque & deque
        pagedTokenQue_.EnQue(perLayerSingleCacheBuffer);
        perLayerSingleCacheBuffer = pagedTokenQue_.DeQue<scalar_t>();
        
        // 4. copy singleCache buffer to the right global idx
        // For MLA_KV and DSA_KV, use the correct base offset
        int64_t cacheTensorLayerOffset = GetLMCBaseOffset(cacheIdx) + 
                                         static_cast<int64_t>(layerIdx) * this->numTokensChunk_ * hiddenDims +
                                         static_cast<int64_t>(startTokensIdx) * hiddenDims;
        AscendC::DataCopy(this->lmcBufferGlobal_[cacheTensorLayerOffset], perLayerSingleCacheBuffer, 
                          actualTokensPerInnerLoop * hiddenDims);

        // 5. Free
        pagedTokenQue_.FreeTensor(perLayerSingleCacheBuffer);
    }

    __aicore__ inline void _L2PageTransfer(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor, 
                                           __gm__ uint8_t *slotmappings, const int cacheIdx, 
                                           const int layerIdx, const int32_t startTokensIdx, 
                                           const int32_t endTokensIdx, 
                                           const int32_t actualTokensPerInnerLoop,
                                           const int64_t pagedOffset) {
        // get the slotmappings 
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);
        
        // Get the correct hidden_dims for this cacheIdx
        int64_t hiddenDims = GetHiddenDims(cacheIdx);
        
        // 1. alloc per layer per cache buffer
        local_scalar_t perLayerSingleCacheBuffer = this->pagedTokenQue_.template AllocTensor<scalar_t>();
        
        // 2. copy the L buffer to local
        // For MLA_KV and DSA_KV, use the correct base offset
        int64_t cacheTensorLayerOffset = GetLMCBaseOffset(cacheIdx) + 
                                         static_cast<int64_t>(layerIdx) * this->numTokensChunk_ * hiddenDims +
                                         static_cast<int64_t>(startTokensIdx) * hiddenDims;
        
        AscendC::DataCopy(perLayerSingleCacheBuffer, this->lmcBufferGlobal_[cacheTensorLayerOffset], 
                          actualTokensPerInnerLoop * hiddenDims);
        
        // 3. enque & deque
        pagedTokenQue_.EnQue(perLayerSingleCacheBuffer);
        perLayerSingleCacheBuffer = pagedTokenQue_.DeQue<scalar_t>();

        // 4. now this is in ub
        int64_t slot;
        int64_t tmpPagedOffset;
        int64_t localTensorTokenOffset;
        // copy per token into paged
        for (int64_t tokenIdx = startTokensIdx; tokenIdx < endTokensIdx; tokenIdx++) {
            slot = static_cast<int64_t>(slotmappingPtr[tokenIdx]);
            tmpPagedOffset = pagedOffset + slot * hiddenDims;
            localTensorTokenOffset = (tokenIdx - startTokensIdx) * hiddenDims;
            AscendC::DataCopy(this->pagedTokenGlobal_[tmpPagedOffset], perLayerSingleCacheBuffer[localTensorTokenOffset], 
                              hiddenDims);
        }

        // 5. free
        pagedTokenQue_.FreeTensor(perLayerSingleCacheBuffer);
    }

    __aicore__ inline void processLayerCache(__gm__ uint8_t *pagedKVCaches, __gm__ uint8_t* cacheTensor, 
                                             __gm__ uint8_t *slotmappings, const int cacheIdx, 
                                             const int layerIdx, const bool page2L) 
    {
        // Get the correct hidden_dims for this cacheIdx
        int64_t hiddenDims = GetHiddenDims(cacheIdx);

        // vllm 0.9.2：One pointer per layer, pointing to [2, pages, page_size, ...]
        // vllm 0.11.0：Two pointers per layer (Key and Value independent)
        // Pointer array layout:[Layer0.Key, Layer0.Value, Layer1.Key, Layer1.Value, ...]
        __gm__ uint8_t *pagedLayerKVCaches = 
            kvcache_ops::GetLayerBasePtr<kvcache_fmt>(pagedKVCaches, layerIdx, cacheIdx);
        
        int64_t pagedOffset = 0;
        if constexpr (kvcache_fmt == kvcache_ops::KVCacheFormat::MERGED_KV) {
            pagedOffset = cacheIdx * this->pageBuffSize_ * hiddenDims;
        }
        
        // For both page2L and L2Page, we copy per token via and to the pagedcache.
        this->pagedTokenGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(pagedLayerKVCaches),
                                                hiddenDims);
        
        // For the cache tensor, since per layer is contiguous, we do contiguous copy.
        this->lmcBufferGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(cacheTensor),
                                               this->numTokensChunk_ * hiddenDims);

        // loop over tokens per loop
        int32_t startTokensIdx;
        int32_t endTokensIdx;
        int32_t actualTokensPerInnerLoop;

        for (startTokensIdx = 0; startTokensIdx < this->numTokensChunk_; startTokensIdx += this->maxTokensPerLoop_) {
            endTokensIdx = startTokensIdx + this->maxTokensPerLoop_;
            endTokensIdx = min(endTokensIdx, this->numTokensChunk_);
            actualTokensPerInnerLoop = endTokensIdx - startTokensIdx;

            if (page2L) {
                this->_page2LTransfer(pagedKVCaches, cacheTensor, slotmappings, cacheIdx, layerIdx, 
                                     startTokensIdx, endTokensIdx, actualTokensPerInnerLoop, pagedOffset);
            } else {
                this->_L2PageTransfer(pagedKVCaches, cacheTensor, slotmappings, cacheIdx, layerIdx, 
                                     startTokensIdx, endTokensIdx, actualTokensPerInnerLoop, pagedOffset);
            }
        }
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> pagedTokenQue_;

    // [layers * [kvs, numPages * pagedSize, heads*headsize]]
    AscendC::GlobalTensor<scalar_t> pagedTokenGlobal_;
    // [kvs, layers, numTokensChunk, heads*headsize]
    AscendC::GlobalTensor<scalar_t> lmcBufferGlobal_;
    int32_t numLayers_; // num layers
    int64_t pageBuffSize_; // pages * pageSize
    int64_t hiddenDims_; // heads * headSize (for MERGED_KV and SEPARATE_KV)
    int32_t numTokensChunk_; // num tokens in the cache tensor chunk
    int32_t maxTokensPerLoop_; // num tokens per inner loop for transferring
    int64_t perLoopBuffSize_; // buffer size in innerloop within UB
    bool valid_;
    bool page2L_; // true, from pagedTensor to LMC, false otherwise
    
    // For MLA_KV and DSA_KV: different hidden_dims for K/V/DSA_K
    int64_t kHiddenDims_;
    int64_t vHiddenDims_;
    int64_t dsaHiddenDims_;
};

#define MULTI_LAYER_PAGED_KV_COPY_V2_KERNEL_NAME(TYPE, SLOTTYPE, FMT) \
    multi_layer_paged_kv_copy_v2_##TYPE##_##SLOTTYPE##_##FMT

#define MULTI_LAYER_PAGED_KV_COPY_V2_DECLARE(TYPE, SLOTTYPE, FMT)                                     \
    extern "C" __global__ __aicore__ void MULTI_LAYER_PAGED_KV_COPY_V2_KERNEL_NAME(TYPE, SLOTTYPE, FMT)( \
        __gm__ uint8_t* pagedKVCaches, __gm__ uint8_t* dstCacheTensor, __gm__ uint8_t* slotmappings,    \
        const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,                           \
        const int64_t pageBuffSize, const int32_t numTokensChunk,                                       \
        const int64_t perLoopBuffer, const int32_t maxTokensPerLoop, const bool page2L,                 \
        const int64_t kHiddenDims, const int64_t vHiddenDims, const int64_t dsaHiddenDims)              \
    {                                                                                                   \
        AscendC::TPipe pipe;                                                                            \
        MultiLayerPagedKVCopyV2<TYPE, SLOTTYPE, kvcache_ops::KVCacheFormat::FMT> op{};                  \
        int32_t bIdx = AscendC::GetBlockIdx();                                                          \
        int32_t launchedCores = AscendC::GetBlockNum();                                                 \
        int32_t layersPerCore = (numLayers + launchedCores - 1) / launchedCores;                        \
        int32_t startLayersIdx = bIdx * layersPerCore;                                                  \
        int32_t endLayersIdx = min(numLayers, startLayersIdx + layersPerCore);                          \
        op.init(pagedKVCaches, dstCacheTensor, slotmappings, hiddenDims,                                \
                numLayers, pageBuffSize, numTokensChunk, perLoopBuffer, maxTokensPerLoop, page2L, &pipe, \
                kHiddenDims, vHiddenDims, dsaHiddenDims);                                               \
        for (int32_t layerIdx = startLayersIdx; layerIdx < endLayersIdx; layerIdx++) {                  \
            for (int32_t cacheIdx = 0; cacheIdx < kvs; cacheIdx++) {                                    \
                op.processLayerCache(pagedKVCaches, dstCacheTensor, slotmappings, cacheIdx, layerIdx, page2L); \
            }                                                                                           \
        }                                                                                               \
    }

#define EXPAND_FMT_V2(TYPE, SLOTTYPE) \
    MULTI_LAYER_PAGED_KV_COPY_V2_DECLARE(TYPE, SLOTTYPE, MERGED_KV) \
    MULTI_LAYER_PAGED_KV_COPY_V2_DECLARE(TYPE, SLOTTYPE, SEPARATE_KV) \
    MULTI_LAYER_PAGED_KV_COPY_V2_DECLARE(TYPE, SLOTTYPE, MLA_KV) \
    MULTI_LAYER_PAGED_KV_COPY_V2_DECLARE(TYPE, SLOTTYPE, DSA_KV)

#define EXPAND_SLOT_V2(TYPE) \
    EXPAND_FMT_V2(TYPE, int32_t) \
    EXPAND_FMT_V2(TYPE, int64_t)

// Declare support kernel entry in the device side
EXPAND_SLOT_V2(half)
EXPAND_SLOT_V2(int8_t)
#if (__CCE_AICORE__ >= 220)
EXPAND_SLOT_V2(bfloat16_t)
#endif

namespace kvcache_ops {

#define SPECIALIZE_V2_LAUNCHER(TYPE, SLOTTYPE, FMT)                                                    \
template<>                                                                                             \
struct V2Launcher<TYPE, SLOTTYPE, KVCacheFormat::FMT> {                                                \
    static void Launch(uint32_t blockDim, void *stream,                                                \
                      uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings,          \
                      const V2Config& config,                                                          \
                      int64_t kHiddenDims = 0, int64_t vHiddenDims = 0, int64_t dsaHiddenDims = 0)     \
    {                                                                                                  \
        MULTI_LAYER_PAGED_KV_COPY_V2_KERNEL_NAME(TYPE, SLOTTYPE, FMT)<<<blockDim, nullptr, stream>>>( \
            pagedKVCaches, dstCacheTensor, slotmappings,                                               \
            config.common.hiddenDims, config.common.kvs, config.common.numLayers,                      \
            config.common.pageBuffSize, config.common.numTokensChunk,                                  \
            config.perLoopBuffSize, config.maxTokensPerLoop, config.common.page2L,                     \
            kHiddenDims, vHiddenDims, dsaHiddenDims);                                                  \
    }                                                                                                  \
};

#define EXPAND_V2_LAUNCHER_FMT(TYPE, SLOTTYPE) \
    SPECIALIZE_V2_LAUNCHER(TYPE, SLOTTYPE, MERGED_KV) \
    SPECIALIZE_V2_LAUNCHER(TYPE, SLOTTYPE, SEPARATE_KV) \
    SPECIALIZE_V2_LAUNCHER(TYPE, SLOTTYPE, MLA_KV) \
    SPECIALIZE_V2_LAUNCHER(TYPE, SLOTTYPE, DSA_KV)

#define EXPAND_V2_LAUNCHER_SLOT(TYPE) \
    EXPAND_V2_LAUNCHER_FMT(TYPE, int32_t) \
    EXPAND_V2_LAUNCHER_FMT(TYPE, int64_t)

EXPAND_V2_LAUNCHER_SLOT(half)
EXPAND_V2_LAUNCHER_SLOT(int8_t)
// this compile definition is for the host side.
#if (ASCEND_AICORE_ARCH >= 220)
EXPAND_V2_LAUNCHER_SLOT(bfloat16_t)
#endif

extern void multi_layer_kv_transfer_kernel_v2(kvcache_ops::AscendType type, kvcache_ops::AscendType slotType, 
                                              const kvcache_ops::KVCacheFormat kvcacheFormat,uint32_t blockDim, void *stream,
                                              uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings, 
                                              const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers, 
                                              const int64_t pageBuffSize, const int32_t numTokensChunk, 
                                              const int64_t perLoopBuffer, const int32_t maxTokensPerLoop,
                                              const bool page2L,
                                              const int64_t kHiddenDims = 0, const int64_t vHiddenDims = 0, 
                                              const int64_t dsaHiddenDims = 0)
{
    auto config = kvcache_ops::MakeV2Config(
        hiddenDims, numLayers, pageBuffSize, numTokensChunk, page2L, kvs,
        perLoopBuffer, maxTokensPerLoop
    );

    switch(type) {
        case kvcache_ops::AscendType::FP16:
            kvcache_ops::dispatch_paged_kernel_on_slot_type<kvcache_ops::V2Launcher, half>(
                slotType, kvcacheFormat, blockDim, stream, 
                pagedKVCaches, dstCacheTensor, slotmappings, config, kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;    
#if (ASCEND_AICORE_ARCH >= 220)
        case kvcache_ops::AscendType::BF16:
            kvcache_ops::dispatch_paged_kernel_on_slot_type<kvcache_ops::V2Launcher, bfloat16_t>(
                slotType, kvcacheFormat, blockDim, stream, 
                pagedKVCaches, dstCacheTensor, slotmappings, config, kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
#endif
        case kvcache_ops::AscendType::INT8:
            kvcache_ops::dispatch_paged_kernel_on_slot_type<kvcache_ops::V2Launcher, int8_t>(
                slotType, kvcacheFormat, blockDim, stream, 
                pagedKVCaches, dstCacheTensor, slotmappings, config, kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
        default:
            ASCENDC_REPORT_NOT_SUPPORT(false, std::to_string(static_cast<int>(type)) + " is not supported.")
            throw std::runtime_error("Scalar type: " + std::to_string(static_cast<int>(type)) + " not supported. This should not have happened.");
    }
}

} // namespace kvcache_ops
