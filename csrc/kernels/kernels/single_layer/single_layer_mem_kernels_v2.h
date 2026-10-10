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

#ifndef SINGLE_LAYER_MEM_KERNELS_V2_H
#define SINGLE_LAYER_MEM_KERNELS_V2_H

#include "kernel_operator.h"

constexpr int32_t ASCEND_BLOCK_LEN = 32;

template <typename scalar_t>
struct MergedPolicy {
    AscendC::GlobalTensor<scalar_t> vllmKVGlobal;
    int64_t blockStride;
    int64_t valueOffset;
    int32_t headDims;
    int32_t numHeads;
    int32_t blockSize;

    __aicore__ inline void Init(GM_ADDR kvPtr, int64_t stride, int64_t vOffset, int64_t bufSize,
                                int32_t nHeads, int32_t hDims, int32_t bSize) {
        vllmKVGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(kvPtr), bufSize);
        blockStride = stride;
        valueOffset = vOffset;
        numHeads = nHeads;
        headDims = hDims;
        blockSize = bSize;
    }

    __aicore__ inline void Copy2Local(const AscendC::LocalTensor<scalar_t>& localTensor, 
                                      int64_t blockIdx, int64_t blockOffset, 
                                      int32_t localKIdx, int32_t localVIdx) {
        int64_t kIdx = blockIdx * blockStride + blockOffset * numHeads * headDims;
        int64_t vIdx = kIdx + valueOffset;
        int32_t len = numHeads * headDims;

        AscendC::DataCopy(localTensor[localKIdx], vllmKVGlobal[kIdx], len);
        AscendC::DataCopy(localTensor[localVIdx], vllmKVGlobal[vIdx], len);
    }

    __aicore__ inline void Copy2Global(const AscendC::LocalTensor<scalar_t>& localTensor, 
                                       int64_t blockIdx, int64_t blockOffset, 
                                       int32_t localKIdx, int32_t localVIdx) {
        int64_t kIdx = blockIdx * blockStride + blockOffset * numHeads * headDims;
        int64_t vIdx = kIdx + valueOffset;
        int32_t len = numHeads * headDims;

        AscendC::DataCopy(vllmKVGlobal[kIdx], localTensor[localKIdx], len);
        AscendC::DataCopy(vllmKVGlobal[vIdx], localTensor[localVIdx], len);
    }
};

template <typename scalar_t>
struct SeparatePolicy {
    AscendC::GlobalTensor<scalar_t> vllmKeyGlobal;
    AscendC::GlobalTensor<scalar_t> vllmValueGlobal;
    int64_t keyBlockStride;
    int64_t valueBlockStride;
    int32_t headDims;
    int32_t numHeads;
    int32_t blockSize;

    __aicore__ inline void Init(GM_ADDR kPtr, GM_ADDR vPtr, int64_t kStride, int64_t vStride, 
                                int64_t kSize, int64_t vSize, 
                                int32_t nHeads, int32_t hDims, int32_t bSize) {
        vllmKeyGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(kPtr), kSize);
        vllmValueGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(vPtr), vSize);
        keyBlockStride = kStride;
        valueBlockStride = vStride;
        numHeads = nHeads;
        headDims = hDims;
        blockSize = bSize;
    }

    __aicore__ inline void Copy2Local(const AscendC::LocalTensor<scalar_t>& localTensor, 
                                      int64_t blockIdx, int64_t blockOffset, 
                                      int32_t localKIdx, int32_t localVIdx) {
        int64_t kIdx = blockIdx * keyBlockStride + blockOffset * numHeads * headDims;
        int64_t vIdx = blockIdx * valueBlockStride + blockOffset * numHeads * headDims;
        int32_t len = numHeads * headDims;

        AscendC::DataCopy(localTensor[localKIdx], vllmKeyGlobal[kIdx], len);
        AscendC::DataCopy(localTensor[localVIdx], vllmValueGlobal[vIdx], len);
    }

    __aicore__ inline void Copy2Global(const AscendC::LocalTensor<scalar_t>& localTensor, 
                                       int64_t blockIdx, int64_t blockOffset, 
                                       int32_t localKIdx, int32_t localVIdx) {
        int64_t kIdx = blockIdx * keyBlockStride + blockOffset * numHeads * headDims;
        int64_t vIdx = blockIdx * valueBlockStride + blockOffset * numHeads * headDims;
        int32_t len = numHeads * headDims;

        AscendC::DataCopy(vllmKeyGlobal[kIdx], localTensor[localKIdx], len);
        AscendC::DataCopy(vllmValueGlobal[vIdx], localTensor[localVIdx], len);
    }
};

template <typename scalar_t, typename slot_t, typename PolicyT> 
class SingleLayerPagedKVCopyProcessor {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;

public:
    __aicore__ inline SingleLayerPagedKVCopyProcessor() {}

    // Accessor to initialize the specific policy
    __aicore__ inline PolicyT& GetPolicy() {
        return policy_;
    }

    // Common Initialization for shared resources (LMC, Pipe, Queue)
    __aicore__ inline void InitCommon(GM_ADDR lmcKeyValueCachePtr, 
                                      GM_ADDR slotMappingPtr, 
                                      const int64_t lmcTokenStride, const int64_t lmcValueOffset, const int64_t lmcBufferSize,
                                      const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims, 
                                      const int32_t numTokens, const int32_t blockSize, const bool page2L, const bool lmcTokensMajor, 
                                      AscendC::TPipe *pipe)
    {
        this->pipe_ = pipe;
        this->numHeads_ = numHeads;
        this->numTokens_ = numTokens;
        this->blockSize_ = blockSize;
        this->page2L_ = page2L;
        this->headDims_ = headDims;
        this->lmcTokenStride_ = lmcTokenStride;
        this->lmcValueOffset_ = lmcValueOffset;
        this->lmcTokensMajor_ = lmcTokensMajor;
        
        // Fixed constant as per original implementation
        this->numKvs_ = 2; 

        this->lmcBufferGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(lmcKeyValueCachePtr), lmcBufferSize);

        uint64_t localTokenBufferSize = maxTokensPerLoop * this->numKvs_ * this->numHeads_ * this->headDims_ * sizeof(scalar_t);
        this->pipe_->InitBuffer(this->tokenQue_, 2, localTokenBufferSize);
    }

    __aicore__ inline void process(__gm__ uint8_t *slotmappings, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        if (this->page2L_) {
            this->runCopyPage2L(slotmappings, tokenIdx, actualTokensPerLoop);
        } else {
            this->runCopyL2Page(slotmappings, tokenIdx, actualTokensPerLoop);
        }
    }

private:
    // VLLM (Global) -> LMC (Global)
    __aicore__ inline void runCopyPage2L(__gm__ uint8_t *slotmappings, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        // alloc local buffer per tokens
        local_scalar_t tokensBufferTensor = this->tokenQue_.template AllocTensor<scalar_t>();
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);

        // [Produce]: Read from VLLM to Local UB
        int64_t slot, blockIdx, blockOffset;
        int64_t localTokenBuffKIdx, localTokenBuffVIdx;
        int64_t realTokenIdx;

        // per slot mapping we copy the tokens
        // we are moving number of actualTokensPerLoop tokens from paged memory to local UB memory
        // if we have a standard merged KV tensor, we stride for the value offset; for separate KV tensors, we directly stride.
        // otherwise, for MLA, we skip the value offset, and the local buffer would already have it taken into account.
        for (int32_t innerTokenIdx = 0; innerTokenIdx < actualTokensPerLoop; innerTokenIdx++) {
            realTokenIdx = tokenIdx + innerTokenIdx;
            slot = static_cast<int64_t>(slotmappingPtr[realTokenIdx]);
            
            // NOTE(niming): Skip tokens that have already been matched in the prefix cache.
            // A slot value of -1 indicates a prefix-match hit, meaning the KV data 
            // for this token already exists in the cache and does not need processing.
            if (slot == -1) {
                continue;
            }

            // work out where is it in the vllm page buff
            blockIdx = slot / this->blockSize_;
            blockOffset = slot % this->blockSize_;

            localTokenBuffKIdx = innerTokenIdx * this->numKvs_ * this->numHeads_ * this->headDims_;
            localTokenBuffVIdx = localTokenBuffKIdx + this->numHeads_ * this->headDims_; // stride for heads*headDims
            
            policy_.Copy2Local(tokensBufferTensor, blockIdx, blockOffset, localTokenBuffKIdx, localTokenBuffVIdx);
        }

        // [Sync]: Wait for Data to arrive in UB
        this->tokenQue_.EnQue(tokensBufferTensor);
        tokensBufferTensor = this->tokenQue_.template DeQue<scalar_t>();

        // [Consume]: Write from Local UB to LMC
        CopyLocalToLmc(tokensBufferTensor, tokenIdx, actualTokensPerLoop);
        
        this->tokenQue_.FreeTensor(tokensBufferTensor);
    }

    // LMC (Global) -> VLLM (Global)
    __aicore__ inline void runCopyL2Page(__gm__ uint8_t *slotmappings, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        // alloc local buffer per tokens
        local_scalar_t tokensBufferTensor = this->tokenQue_.template AllocTensor<scalar_t>();
        __gm__ slot_t *slotmappingPtr = reinterpret_cast<__gm__ slot_t*>(slotmappings);

        // [Produce]: Read from LMC to Local UB
        CopyLmcToLocal(tokensBufferTensor, tokenIdx, actualTokensPerLoop);

        // [Sync]: Wait for Data to arrive in UB (CRITICAL FIX)
        this->tokenQue_.EnQue(tokensBufferTensor);
        tokensBufferTensor = this->tokenQue_.template DeQue<scalar_t>();

        // [Consume]: Write from Local UB to VLLM
        int64_t slot, blockIdx, blockOffset;
        int64_t localTokenBuffKIdx, localTokenBuffVIdx;
        int64_t realTokenIdx;

        for (int32_t innerTokenIdx = 0; innerTokenIdx < actualTokensPerLoop; innerTokenIdx++) {
            realTokenIdx = tokenIdx + innerTokenIdx;
            slot = static_cast<int64_t>(slotmappingPtr[realTokenIdx]);
            
            if (slot == -1) {
                continue;
            }

            blockIdx = slot / this->blockSize_;
            blockOffset = slot % this->blockSize_;

            localTokenBuffKIdx = static_cast<int64_t>(innerTokenIdx) * this->numKvs_ * this->numHeads_ * this->headDims_;
            localTokenBuffVIdx = localTokenBuffKIdx + this->numHeads_ * this->headDims_;

            policy_.Copy2Global(tokensBufferTensor, blockIdx, blockOffset, localTokenBuffKIdx, localTokenBuffVIdx);
        }

        this->tokenQue_.FreeTensor(tokensBufferTensor);
    }

    __aicore__ inline void CopyLocalToLmc(local_scalar_t& tokensBufferTensor, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        int64_t perCacheBlockLen = (this->numHeads_ * this->headDims_ * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        int64_t lmcTokenKOffset = tokenIdx * this->lmcTokenStride_;
        
        // copy from lmcbuffer to local tokens
        // we always do tokens major in the ub buffer
        AscendC::DataCopyParams tokenCopyParams;
        tokenCopyParams.blockCount = actualTokensPerLoop;

        if (this->lmcTokensMajor_ || this->numKvs_ == 1) {
            tokenCopyParams.blockLen = perCacheBlockLen * this->numKvs_;
            tokenCopyParams.srcStride = 0;
            tokenCopyParams.dstStride = 0;
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenKOffset], tokensBufferTensor, tokenCopyParams);
        } else {
            // tokensMajor local -> twoMajor global
            tokenCopyParams.blockLen = perCacheBlockLen;
            tokenCopyParams.srcStride = perCacheBlockLen;
            tokenCopyParams.dstStride = 0;
            
            // Copy K
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenKOffset], tokensBufferTensor, tokenCopyParams);
            
            // Copy V
            int64_t lmcTokenVOffset = lmcTokenKOffset + this->lmcValueOffset_;
            int64_t localVOffset = this->numHeads_ * this->headDims_;
            AscendC::DataCopy(this->lmcBufferGlobal_[lmcTokenVOffset], tokensBufferTensor[localVOffset], tokenCopyParams);
        }
    }

    __aicore__ inline void CopyLmcToLocal(local_scalar_t& tokensBufferTensor, int32_t tokenIdx, int32_t actualTokensPerLoop) {
        int64_t perCacheBlockLen = (this->numHeads_ * this->headDims_ * sizeof(scalar_t)) / ASCEND_BLOCK_LEN;
        
        // copy from lmcbuffer to local tokens
        // we always do tokens major in the ub buffer
        AscendC::DataCopyParams tokensCopyParams;
        tokensCopyParams.blockCount = actualTokensPerLoop;
        int64_t lmcTokenIdx = tokenIdx * this->lmcTokenStride_;
        
        if (this->lmcTokensMajor_ || this->numKvs_ == 1) {
            tokensCopyParams.blockLen = perCacheBlockLen * this->numKvs_;
            tokensCopyParams.srcStride = 0;
            tokensCopyParams.dstStride = 0;
            AscendC::DataCopy(tokensBufferTensor, this->lmcBufferGlobal_[lmcTokenIdx], tokensCopyParams);
        } else {
            // twoMajor global -> tokensMajor local
            tokensCopyParams.blockLen = perCacheBlockLen;
            tokensCopyParams.srcStride = 0;
            tokensCopyParams.dstStride = perCacheBlockLen;
            
            // Copy K
            AscendC::DataCopy(tokensBufferTensor, this->lmcBufferGlobal_[lmcTokenIdx], tokensCopyParams);
            
            // Copy V
            int64_t localVOffset = this->numHeads_ * this->headDims_;
            int64_t lmcTokenVIdx = lmcTokenIdx + this->lmcValueOffset_;
            AscendC::DataCopy(tokensBufferTensor[localVOffset], this->lmcBufferGlobal_[lmcTokenVIdx], tokensCopyParams);
        }
    }

private:
    AscendC::TPipe *pipe_;
    // Instance of the specific policy
    PolicyT policy_; 
    // a depth of 2
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> tokenQue_;

    // Depends on LMC setting whether we store in tokensMajor or not.
    // the layout would be the followings:
    // [tokens, kvs, heads*headsize] or [kvs, tokens, heads*headsize]
    // TODO: check whether should combine the two and use a loop
    AscendC::GlobalTensor<scalar_t> lmcBufferGlobal_;

    int64_t lmcTokenStride_;
    int64_t lmcValueOffset_;
    int32_t blockSize_; // the size of the paged attention tokens block
    int32_t headDims_;
    int32_t numHeads_;
    int32_t numTokens_; // num tokens in the cache tensor chunk
    int16_t numKvs_; // 1 if MLA else 2
    bool page2L_; // whether the direction of copy is from page to lmc
    bool lmcTokensMajor_; // whether the lmc buffer is in tokens major i.e. [tokens, kvs, ...]
};

#endif // SINGLE_LAYER_MEM_KERNELS_V2_H
