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
 
#ifndef MULTI_LAYER_MEM_KERNELS_H
#define MULTI_LAYER_MEM_KERNELS_H

#include "kernel_operator.h"
#include "../types.h"
#include <stdexcept>
#include <string>

namespace kvcache_ops {

struct StandardConfig {
    int64_t hiddenDims; // heads * headSize
    int32_t numLayers; // num layers
    int64_t pageBuffSize; // pages * pageSize
    int32_t numTokensChunk; // num tokens in the cache tensor chunk
    bool page2L; // true, from pagedTensor to LMC, false otherwise
    int32_t kvs;
};

struct Chunk310PConfig {
    StandardConfig common;
    // 310p
    int32_t numKVHead;
    int32_t headSize;
    int32_t blockSize;
    int32_t chunkSize;
};

struct V2Config {
    StandardConfig common;
    int64_t perLoopBuffSize;  // buffer size in innerloop within UB
    int32_t maxTokensPerLoop; // num tokens per inner loop for transferring
};

inline StandardConfig MakeStandardConfig(
    int64_t hiddenDims, int32_t numLayers, int64_t pageBuffSize,
    int32_t numTokensChunk, int32_t kvs, bool page2L)
{
    StandardConfig cfg;
    cfg.hiddenDims = hiddenDims;
    cfg.numLayers = numLayers;
    cfg.pageBuffSize = pageBuffSize;
    cfg.numTokensChunk = numTokensChunk;
    cfg.page2L = page2L;
    cfg.kvs = kvs;
    return cfg;
}

inline Chunk310PConfig Make310PConfig(
    int64_t hiddenDims, int32_t numLayers, int64_t pageBuffSize,
    int32_t numTokensChunk, bool page2L, int32_t kvs,
    int32_t numKVHead, int32_t headSize, int32_t blockSize, int32_t chunkSize)
{
    Chunk310PConfig cfg;
    cfg.common = {hiddenDims, numLayers, pageBuffSize, numTokensChunk, page2L, kvs};
    cfg.numKVHead = numKVHead;
    cfg.headSize = headSize;
    cfg.blockSize = blockSize;
    cfg.chunkSize = chunkSize;
    return cfg;
}

inline V2Config MakeV2Config(
    int64_t hiddenDims, int32_t numLayers, int64_t pageBuffSize,
    int32_t numTokensChunk, bool page2L, int32_t kvs,
    int64_t perLoopBuffSize, int32_t maxTokensPerLoop)
{
    V2Config cfg;
    cfg.common = {hiddenDims, numLayers, pageBuffSize, numTokensChunk, page2L, kvs};
    cfg.perLoopBuffSize = perLoopBuffSize;
    cfg.maxTokensPerLoop = maxTokensPerLoop;
    return cfg;
}

// Helper function to get layer base pointer based on KVCache format
template <KVCacheFormat fmt>
__aicore__ inline __gm__ uint8_t* GetLayerBasePtr(
    GM_ADDR pagedKVCaches, 
    int32_t layerIdx, 
    int32_t kvIdx) 
{   
    // its a pointer within the GM addr space, that point to another GM addr space
    __gm__ uint8_t * __gm__ *pagedKVCachesPtr = 
        reinterpret_cast<__gm__ uint8_t* __gm__ *>(pagedKVCaches);
    
    // getting the right ptr to the paged kvcache layer
    if constexpr (fmt == KVCacheFormat::MERGED_KV) {
        return pagedKVCachesPtr[layerIdx];
    } else if constexpr (fmt == KVCacheFormat::SEPARATE_KV || fmt == KVCacheFormat::MLA_KV) {
        return pagedKVCachesPtr[layerIdx * 2 + kvIdx];
    } else if constexpr (fmt == KVCacheFormat::DSA_KV) {
        return pagedKVCachesPtr[layerIdx * 3 + kvIdx];
    }
}

template <typename scalar_t, typename slot_t, KVCacheFormat fmt>
struct StandardPolicy {

    int64_t hiddenDims_;
    int32_t numLayers_;
    int64_t pageBuffSize_;
    int32_t numTokensChunk_;
    bool page2L_;

    __aicore__ inline void Init(
        int64_t hiddenDims, int32_t numLayers, int64_t pageBuffSize,
        int32_t numTokensChunk, bool page2L, int32_t kvs)
    {
        hiddenDims_ = hiddenDims;
        numLayers_ = numLayers;
        pageBuffSize_ = pageBuffSize;
        numTokensChunk_ = numTokensChunk;
        page2L_ = page2L;
    }

    __aicore__ inline void InitBuffer(
        AscendC::TPipe* pipe, 
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4>& tokenQue) 
    {
        pipe->InitBuffer(tokenQue, 4, hiddenDims_ * sizeof(scalar_t));
    }

    __aicore__ inline void ProcessToken(
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4>& tokenQue,
        GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, int64_t slot, int32_t tokenIdx, int32_t kvIdx) 
    {
        for (int32_t layerIdx = 0; layerIdx < numLayers_; layerIdx++) {
            __gm__ uint8_t* layerBase = GetLayerBasePtr<fmt>(pagedKVCaches, layerIdx, kvIdx);
            
            int64_t pagedOffset = GetPagedOffset(slot, kvIdx);
            int64_t lmcOffset = GetLMCOffset(kvIdx, layerIdx, tokenIdx);

            AscendC::GlobalTensor<scalar_t> pagedGlobal;
            AscendC::GlobalTensor<scalar_t> lmcGlobal;
            
            pagedGlobal.SetGlobalBuffer(
                reinterpret_cast<__gm__ scalar_t*>(layerBase) + pagedOffset, hiddenDims_);
            lmcGlobal.SetGlobalBuffer(
                reinterpret_cast<__gm__ scalar_t*>(cacheTensor) + lmcOffset, hiddenDims_);

            // Alloc Tensor for local page
            AscendC::LocalTensor<scalar_t> localTensor = tokenQue.template AllocTensor<scalar_t>();
            
            if (page2L_) {
                // copy from global tensor into local
                AscendC::DataCopy(localTensor, pagedGlobal, hiddenDims_);
                tokenQue.EnQue(localTensor);
                localTensor = tokenQue.template DeQue<scalar_t>();
                //datacopy into GM
                AscendC::DataCopy(lmcGlobal, localTensor, hiddenDims_);
            } else {
                AscendC::DataCopy(localTensor, lmcGlobal, hiddenDims_);
                tokenQue.EnQue(localTensor);
                localTensor = tokenQue.template DeQue<scalar_t>();
                AscendC::DataCopy(pagedGlobal, localTensor, hiddenDims_);
            }
            
            // free alloced Tensor
            tokenQue.FreeTensor(localTensor);
        }
    }
private:
    __aicore__ inline int64_t GetPagedOffset(int64_t slot, int32_t kvIdx) 
    {
        if constexpr (fmt == KVCacheFormat::MERGED_KV) {
            return kvIdx * pageBuffSize_ * hiddenDims_ + slot * hiddenDims_;
        } else {
            return slot * hiddenDims_;
        }
    }

    __aicore__ inline int64_t GetLMCOffset(int32_t kvIdx, int32_t layerIdx, int32_t tokenIdx) 
    {
        return static_cast<int64_t>(kvIdx) * numLayers_ * numTokensChunk_ * hiddenDims_ +
               static_cast<int64_t>(layerIdx) * numTokensChunk_ * hiddenDims_ +
               static_cast<int64_t>(tokenIdx) * hiddenDims_;
    }
};

template <typename scalar_t, typename slot_t, KVCacheFormat fmt>
struct MLAPolicy {
    int64_t k_hidden_dims_;
    int64_t v_hidden_dims_;
    int32_t numLayers_;
    int64_t pageBuffSize_;
    int32_t numTokensChunk_;
    bool page2L_;

    __aicore__ inline void Init(
        int64_t k_hidden_dims, int64_t v_hidden_dims, int32_t numLayers,
        int64_t pageBuffSize, int32_t numTokensChunk, bool page2L, int32_t kvs)
    {
        k_hidden_dims_ = k_hidden_dims;
        v_hidden_dims_ = v_hidden_dims;
        numLayers_ = numLayers;
        pageBuffSize_ = pageBuffSize;
        numTokensChunk_ = numTokensChunk;
        page2L_ = page2L;
    }

    __aicore__ inline int64_t GetHiddenDims(int32_t kvIdx)
    {
        return (kvIdx == 0) ? k_hidden_dims_ : v_hidden_dims_;
    }

    __aicore__ inline void InitBuffer(
        AscendC::TPipe* pipe, 
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4>& tokenQue)
    {
        pipe->InitBuffer(tokenQue, 4, k_hidden_dims_ * sizeof(scalar_t));
    }

    __aicore__ inline void ProcessToken(
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4>& tokenQue,
        GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, int64_t slot, int32_t tokenIdx, int32_t kvIdx)
    {
        int64_t hiddenDims = GetHiddenDims(kvIdx);
        
        for (int32_t layerIdx = 0; layerIdx < numLayers_; layerIdx++) {
            __gm__ uint8_t* layerBase = GetLayerBasePtr<fmt>(pagedKVCaches, layerIdx, kvIdx);
            
            int64_t pagedOffset = GetPagedOffset(slot, kvIdx);
            int64_t lmcOffset = GetLMCOffset(kvIdx, layerIdx, tokenIdx);

            AscendC::GlobalTensor<scalar_t> pagedGlobal;
            AscendC::GlobalTensor<scalar_t> lmcGlobal;
            
            pagedGlobal.SetGlobalBuffer(
                reinterpret_cast<__gm__ scalar_t*>(layerBase) + pagedOffset, hiddenDims);
            lmcGlobal.SetGlobalBuffer(
                reinterpret_cast<__gm__ scalar_t*>(cacheTensor) + lmcOffset, hiddenDims);

            AscendC::LocalTensor<scalar_t> localTensor = tokenQue.template AllocTensor<scalar_t>();
            
            if (page2L_) {
                AscendC::DataCopy(localTensor, pagedGlobal, hiddenDims);
                tokenQue.EnQue(localTensor);
                localTensor = tokenQue.template DeQue<scalar_t>();
                AscendC::DataCopy(lmcGlobal, localTensor, hiddenDims);
            } else {
                AscendC::DataCopy(localTensor, lmcGlobal, hiddenDims);
                tokenQue.EnQue(localTensor);
                localTensor = tokenQue.template DeQue<scalar_t>();
                AscendC::DataCopy(pagedGlobal, localTensor, hiddenDims);
            }
            
            tokenQue.FreeTensor(localTensor);
        }
    }

private:
    __aicore__ inline int64_t GetPagedOffset(int64_t slot, int32_t kvIdx)
    {
        return slot * GetHiddenDims(kvIdx);
    }

    __aicore__ inline int64_t GetLMCOffset(int32_t kvIdx, int32_t layerIdx, int32_t tokenIdx)
    {
        int64_t hiddenDims = GetHiddenDims(kvIdx);
        int64_t base_offset = 0;
        
        if (kvIdx == 1) {
            base_offset = numLayers_ * numTokensChunk_ * k_hidden_dims_;
        }
        
        return base_offset + 
               static_cast<int64_t>(layerIdx) * numTokensChunk_ * hiddenDims +
               static_cast<int64_t>(tokenIdx) * hiddenDims;
    }
};

template <typename scalar_t, typename slot_t, KVCacheFormat fmt>
struct DSAPolicy {
    int64_t k_hidden_dims_;
    int64_t v_hidden_dims_;
    int64_t dsa_hidden_dims_;
    int32_t numLayers_;
    int64_t pageBuffSize_;
    int32_t numTokensChunk_;
    bool page2L_;

    __aicore__ inline void Init(
        int64_t k_hidden_dims, int64_t v_hidden_dims, int64_t dsa_hidden_dims,
        int32_t numLayers, int64_t pageBuffSize, int32_t numTokensChunk,
        bool page2L, int32_t kvs)
    {
        k_hidden_dims_ = k_hidden_dims;
        v_hidden_dims_ = v_hidden_dims;
        dsa_hidden_dims_ = dsa_hidden_dims;
        numLayers_ = numLayers;
        pageBuffSize_ = pageBuffSize;
        numTokensChunk_ = numTokensChunk;
        page2L_ = page2L;
    }

    __aicore__ inline int64_t GetHiddenDims(int32_t kvIdx)
    {
        if (kvIdx == 0) return k_hidden_dims_;
        else if (kvIdx == 1) return v_hidden_dims_;
        else return dsa_hidden_dims_;
    }

    __aicore__ inline void InitBuffer(
        AscendC::TPipe* pipe, 
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4>& tokenQue)
    {
        int64_t max_hidden_dims = k_hidden_dims_;
        if (v_hidden_dims_ > max_hidden_dims) max_hidden_dims = v_hidden_dims_;
        if (dsa_hidden_dims_ > max_hidden_dims) max_hidden_dims = dsa_hidden_dims_;
        
        pipe->InitBuffer(tokenQue, 4, max_hidden_dims * sizeof(scalar_t));
    }

    __aicore__ inline void ProcessToken(
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4>& tokenQue,
        GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, int64_t slot, int32_t tokenIdx, int32_t kvIdx)
    {
        int64_t hiddenDims = GetHiddenDims(kvIdx);
        
        for (int32_t layerIdx = 0; layerIdx < numLayers_; layerIdx++) {
            __gm__ uint8_t* layerBase = GetLayerBasePtr<fmt>(pagedKVCaches, layerIdx, kvIdx);
            
            int64_t pagedOffset = GetPagedOffset(slot, kvIdx);
            int64_t lmcOffset = GetLMCOffset(kvIdx, layerIdx, tokenIdx);

            AscendC::GlobalTensor<scalar_t> pagedGlobal;
            AscendC::GlobalTensor<scalar_t> lmcGlobal;
            
            pagedGlobal.SetGlobalBuffer(
                reinterpret_cast<__gm__ scalar_t*>(layerBase) + pagedOffset, hiddenDims);
            lmcGlobal.SetGlobalBuffer(
                reinterpret_cast<__gm__ scalar_t*>(cacheTensor) + lmcOffset, hiddenDims);

            AscendC::LocalTensor<scalar_t> localTensor = tokenQue.template AllocTensor<scalar_t>();
            
            if (page2L_) {
                AscendC::DataCopy(localTensor, pagedGlobal, hiddenDims);
                tokenQue.EnQue(localTensor);
                localTensor = tokenQue.template DeQue<scalar_t>();
                AscendC::DataCopy(lmcGlobal, localTensor, hiddenDims);
            } else {
                AscendC::DataCopy(localTensor, lmcGlobal, hiddenDims);
                tokenQue.EnQue(localTensor);
                localTensor = tokenQue.template DeQue<scalar_t>();
                AscendC::DataCopy(pagedGlobal, localTensor, hiddenDims);
            }
            
            tokenQue.FreeTensor(localTensor);
        }
    }

private:
    __aicore__ inline int64_t GetPagedOffset(int64_t slot, int32_t kvIdx) 
    {
        return slot * GetHiddenDims(kvIdx);
    }

    __aicore__ inline int64_t GetLMCOffset(int32_t kvIdx, int32_t layerIdx, int32_t tokenIdx) 
    {
        int64_t hiddenDims = GetHiddenDims(kvIdx);
        int64_t base_offset = 0;
        
        if (kvIdx == 1) {
            base_offset = numLayers_ * numTokensChunk_ * k_hidden_dims_;
        } else if (kvIdx == 2) {
            base_offset = numLayers_ * numTokensChunk_ * (k_hidden_dims_ + v_hidden_dims_);
        }
        
        return base_offset +
               static_cast<int64_t>(layerIdx) * numTokensChunk_ * hiddenDims +
               static_cast<int64_t>(tokenIdx) * hiddenDims;
    }
};

template <typename scalar_t, typename slot_t, KVCacheFormat fmt>
struct Chunk310PPolicy {
    int64_t hiddenDims_;
    int32_t numLayers_;
    int64_t pageBuffSize_;
    int32_t numTokensChunk_;
    bool page2L_;
    int32_t numKVHead_;
    int32_t headSize_;
    int32_t blockSize_;
    int32_t chunkSize_;
    int32_t chunksPerHead_;
    int32_t totalChunks_;
    int32_t numBlocks_;

    __aicore__ inline void Init(
        int64_t hiddenDims, int32_t numLayers, int64_t pageBuffSize,
        int32_t numTokensChunk, bool page2L, int32_t kvs,
        int32_t numKVHead, int32_t headSize, int32_t blockSize, int32_t chunkSize) 
    {
        hiddenDims_ = hiddenDims;
        numLayers_ = numLayers;
        pageBuffSize_ = pageBuffSize;
        numTokensChunk_ = numTokensChunk;
        page2L_ = page2L;
        numKVHead_ = numKVHead;
        headSize_ = headSize;
        blockSize_ = blockSize;
        chunkSize_ = chunkSize;
        
        chunksPerHead_ = headSize_ / chunkSize_;
        totalChunks_ = numKVHead_ * chunksPerHead_;
        numBlocks_ = pageBuffSize_ / blockSize_;
    }

    __aicore__ inline void InitBuffer(
        AscendC::TPipe* pipe, 
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4>& tokenQue) 
    {
        pipe->InitBuffer(tokenQue, 4, chunkSize_ * sizeof(scalar_t));
    }

    __aicore__ inline void ProcessToken(
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4>& tokenQue,
        GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, int64_t slot, int32_t tokenIdx, int32_t kvIdx) 
    {   
        if (slot == -1) {
            return;
        }
        int64_t blockId = slot / blockSize_;
        int64_t tokenInBlock = slot % blockSize_;

        for (int32_t layerIdx = 0; layerIdx < numLayers_; layerIdx++) {
            __gm__ uint8_t* layerBase = GetLayerBasePtr<fmt>(pagedKVCaches, layerIdx, kvIdx);

            for (int32_t headIdx = 0; headIdx < numKVHead_; headIdx++) {
                for (int32_t chunkIdx = 0; chunkIdx < chunksPerHead_; chunkIdx++) {
                    
                    int64_t globalChunkIdx = headIdx * chunksPerHead_ + chunkIdx;
                    int64_t pagedOffset = GetPagedChunkOffset(kvIdx, blockId, globalChunkIdx, tokenInBlock);
                    int64_t lmcOffset = GetLMCChunkOffset(kvIdx, layerIdx, tokenIdx, headIdx, chunkIdx);

                    AscendC::GlobalTensor<scalar_t> pagedGlobal;
                    AscendC::GlobalTensor<scalar_t> lmcGlobal;
                    
                    pagedGlobal.SetGlobalBuffer(
                        reinterpret_cast<__gm__ scalar_t*>(layerBase) + pagedOffset, chunkSize_);
                    lmcGlobal.SetGlobalBuffer(
                        reinterpret_cast<__gm__ scalar_t*>(cacheTensor) + lmcOffset, chunkSize_);

                    AscendC::LocalTensor<scalar_t> chunkTensor = tokenQue.template AllocTensor<scalar_t>();
                    
                    if (page2L_) {
                        AscendC::DataCopy(chunkTensor, pagedGlobal, chunkSize_);
                        tokenQue.EnQue(chunkTensor);
                        chunkTensor = tokenQue.template DeQue<scalar_t>();
                        AscendC::DataCopy(lmcGlobal, chunkTensor, chunkSize_);
                    } else {
                        AscendC::DataCopy(chunkTensor, lmcGlobal, chunkSize_);
                        tokenQue.EnQue(chunkTensor);
                        chunkTensor = tokenQue.template DeQue<scalar_t>();
                        AscendC::DataCopy(pagedGlobal, chunkTensor, chunkSize_);
                    }
                    
                    tokenQue.FreeTensor(chunkTensor);
                }
            }
        }
    }

private:
    __aicore__ inline int64_t GetPagedChunkOffset(
        int32_t kvIdx, int64_t blockId, int64_t globalChunkIdx, int64_t tokenInBlock) 
    {
        if constexpr (fmt == KVCacheFormat::MERGED_KV) {
            return kvIdx * (numBlocks_ * totalChunks_ * blockSize_ * chunkSize_) +
                   blockId * (totalChunks_ * blockSize_ * chunkSize_) +
                   globalChunkIdx * (blockSize_ * chunkSize_) +
                   tokenInBlock * chunkSize_;
        } else {
            return blockId * (totalChunks_ * blockSize_ * chunkSize_) +
                   globalChunkIdx * (blockSize_ * chunkSize_) +
                   tokenInBlock * chunkSize_;
        }
    }

    __aicore__ inline int64_t GetLMCChunkOffset(
        int32_t kvIdx, int32_t layerIdx, int32_t tokenIdx, int32_t headIdx, int32_t chunkIdx) 
    {
        return static_cast<int64_t>(kvIdx) * numLayers_ * numTokensChunk_ * hiddenDims_ +
               static_cast<int64_t>(layerIdx) * numTokensChunk_ * hiddenDims_ +
               static_cast<int64_t>(tokenIdx) * hiddenDims_ +
               static_cast<int64_t>(headIdx) * headSize_ +
               static_cast<int64_t>(chunkIdx) * chunkSize_;
    }
};

template <typename scalar_t, typename slot_t, typename PolicyT>
class MultiLayerPagedKVCopyProcessor {
public:
    __aicore__ inline MultiLayerPagedKVCopyProcessor() {}

    __aicore__ inline PolicyT& GetPolicy() {
        return policy_;
    }

    template<typename... Args>
    __aicore__ inline void InitCommon(
        GM_ADDR pagedKVCaches, GM_ADDR cacheTensor, GM_ADDR slotmappings,AscendC::TPipe* pipe,
        int64_t hiddenDims, int32_t numLayers, int64_t pageBuffSize,int32_t numTokensChunk, 
        bool page2L, int32_t kvs, Args... args)
    {
        pipe_ = pipe;
        slotmappings_ = slotmappings;
        pagedKVCaches_ = pagedKVCaches;
        cacheTensor_ = cacheTensor;
        numTokensChunk_ = numTokensChunk;
        kvs_ = kvs;
        
        policy_.Init(hiddenDims, numLayers, pageBuffSize, numTokensChunk, page2L, kvs, args...);
        
        policy_.InitBuffer(pipe_, tokenQue_);
    }

    __aicore__ inline void process(int32_t coreNum) {
        int64_t blockIdx = AscendC::GetBlockIdx();
        __gm__ slot_t* slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings_);

        for (int64_t tokenIdx = blockIdx; tokenIdx < numTokensChunk_; tokenIdx += coreNum) {
            slot_t slotRaw = slotmappingPtr[tokenIdx];
            int64_t slot = static_cast<int64_t>(slotRaw);
            
            for (int32_t kvIdx = 0; kvIdx < kvs_; kvIdx++) {
                policy_.ProcessToken(tokenQue_, pagedKVCaches_, cacheTensor_, 
                                    slot, tokenIdx, kvIdx);
            }
        }
    }

private:
    AscendC::TPipe* pipe_;
    PolicyT policy_;
    
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 4> tokenQue_;
    
    GM_ADDR slotmappings_;
    GM_ADDR pagedKVCaches_;
    GM_ADDR cacheTensor_;
    int32_t numTokensChunk_;
    int32_t kvs_;
};

template<typename scalar_t, typename slot_t, KVCacheFormat fmt>
struct StandardLauncher {
    static void Launch(
        uint32_t blockDim, 
        void* stream, 
        uint8_t* pagedKVCaches, 
        uint8_t* dstCacheTensor, 
        uint8_t* slotmappings,
        const StandardConfig& config,
        int64_t kHiddenDims = 0,
        int64_t vHiddenDims = 0,
        int64_t dsaHiddenDims = 0); 
};

template<typename scalar_t, typename slot_t, KVCacheFormat fmt>
struct Chunk310PLauncher {
    static void Launch(
        uint32_t blockDim, 
        void* stream, 
        uint8_t* pagedKVCaches, 
        uint8_t* dstCacheTensor, 
        uint8_t* slotmappings,
        const Chunk310PConfig& config,
        int64_t kHiddenDims = 0,
        int64_t vHiddenDims = 0,
        int64_t dsaHiddenDims = 0);
};

template<typename scalar_t, typename slot_t, KVCacheFormat fmt>
struct V2Launcher {
    static void Launch(
        uint32_t blockDim, 
        void* stream, 
        uint8_t* pagedKVCaches, 
        uint8_t* dstCacheTensor, 
        uint8_t* slotmappings,
        const V2Config& config,
        int64_t kHiddenDims = 0,
        int64_t vHiddenDims = 0,
        int64_t dsaHiddenDims = 0);
};

template<template<typename, typename, KVCacheFormat> class LauncherT, typename scalar_t, typename slot_t, typename ConfigT>
void dispatch_paged_kernel_on_format(
    KVCacheFormat kvcacheFormat, 
    uint32_t blockDim, 
    void* stream, 
    uint8_t* pagedKVCaches, 
    uint8_t* dstCacheTensor, 
    uint8_t* slotmappings,
    const ConfigT& config,
    int64_t kHiddenDims = 0,
    int64_t vHiddenDims = 0,
    int64_t dsaHiddenDims = 0)
{
    switch (kvcacheFormat) {
        case KVCacheFormat::MERGED_KV:
            LauncherT<scalar_t, slot_t, KVCacheFormat::MERGED_KV>::Launch(
                blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings, config,
                kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
        case KVCacheFormat::SEPARATE_KV:
            LauncherT<scalar_t, slot_t, KVCacheFormat::SEPARATE_KV>::Launch(
                blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings, config,
                kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
        case KVCacheFormat::MLA_KV:
            LauncherT<scalar_t, slot_t, KVCacheFormat::MLA_KV>::Launch(
                blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings, config,
                kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
        case KVCacheFormat::DSA_KV:
            LauncherT<scalar_t, slot_t, KVCacheFormat::DSA_KV>::Launch(
                blockDim, stream, pagedKVCaches, dstCacheTensor, slotmappings, config,
                kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
        default:
            ASCENDC_REPORT_NOT_SUPPORT(false, "Unsupported KVCacheFormat.");
            break;
    }
}

template<template<typename, typename, KVCacheFormat> class LauncherT, typename scalar_t, typename ConfigT>
void dispatch_paged_kernel_on_slot_type(
    AscendType slotType, 
    KVCacheFormat kvcacheFormat,
    uint32_t blockDim, 
    void* stream, 
    uint8_t* pagedKVCaches, 
    uint8_t* dstCacheTensor, 
    uint8_t* slotmappings,
    const ConfigT& config,
    int64_t kHiddenDims = 0,
    int64_t vHiddenDims = 0,
    int64_t dsaHiddenDims = 0) 
{
    switch (slotType) {
        case AscendType::INT32:
            dispatch_paged_kernel_on_format<LauncherT, scalar_t, int32_t>(
                kvcacheFormat, blockDim, stream, 
                pagedKVCaches, dstCacheTensor, slotmappings, config,
                kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
        case AscendType::INT64:
            dispatch_paged_kernel_on_format<LauncherT, scalar_t, int64_t>(
                kvcacheFormat, blockDim, stream, 
                pagedKVCaches, dstCacheTensor, slotmappings, config,
                kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
        default:
            ASCENDC_REPORT_NOT_SUPPORT(false, std::to_string(static_cast<int>(slotType)) + " is not supported.")
            throw std::runtime_error("Slot type: " + std::to_string(static_cast<int>(slotType)) + " not supported. This should not have happened.");
    }
}

} // namespace kvcache_ops

#endif // MULTI_LAYER_MEM_KERNELS_H
