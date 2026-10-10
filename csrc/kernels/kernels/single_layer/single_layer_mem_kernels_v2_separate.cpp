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

#include "single_layer_mem_kernels_v2.h"
#include <stdio.h>
#include "../types.h"
#include <string>
#include <stdexcept>

#define PAGED_KV_COPY_KERNEL_EXECUTE(PROCESSOR_TYPE)                                                        \
    int32_t coreIdx = AscendC::GetBlockIdx();                                                               \
    int32_t launchedCores = AscendC::GetBlockNum();                                                         \
    int32_t tokensPerCore = (numTokens + launchedCores - 1) / launchedCores;                                \
    int32_t startTokenIdx = coreIdx * tokensPerCore;                                                        \
    int32_t endTokenIdx = min(numTokens, startTokenIdx + tokensPerCore);                                    \
    for (int32_t tokenIdx = startTokenIdx; tokenIdx < endTokenIdx; tokenIdx += maxTokensPerLoop) {          \
        int32_t actualTokensPerLoop = min(maxTokensPerLoop, endTokenIdx - tokenIdx);                        \
        op.process(slotMappingPtr, tokenIdx, actualTokensPerLoop);                                          \
    }

// we splits tokens per core
// and loop over tokensPerCore with each loop having maxTokensPerLoop 
#define SINGLE_LAYER_PAGED_KV_COPY_V2_SEPARATE_TYPE_DECLARE(TYPE, SLOTTYPE)                                     \
    extern "C" __global__ __aicore__ void single_layer_paged_kv_copy_v2_separate_##TYPE##_##SLOTTYPE(           \
        __gm__ uint8_t* lmcKeyValueCachePtr, __gm__ uint8_t* vllmKeyPtr, __gm__ uint8_t* vllmValuePtr,          \
        __gm__ uint8_t* slotMappingPtr, const int64_t keyBlockStride, const int64_t valueBlockStride,           \
        const int64_t vllmKeyBufferSize, const int64_t vllmValueBufferSize,                                     \
        const int64_t lmcTokenStride, const int64_t lmcValueOffset, const int64_t lmcBufferSize,                \
        const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims,                         \
        const int32_t numTokens, const int32_t blockSize, const bool page2L,                                    \
        const bool lmcTokensMajor)                                                                              \
    {                                                                                                           \
        AscendC::TPipe pipe;                                                                                    \
        SingleLayerPagedKVCopyProcessor<TYPE, SLOTTYPE, SeparatePolicy<TYPE>> op;                               \
        /* 1. Initialize Policy-specific parameters */                                                          \
        op.GetPolicy().Init(vllmKeyPtr, vllmValuePtr, keyBlockStride, valueBlockStride,                         \
                            vllmKeyBufferSize, vllmValueBufferSize, numHeads, headDims, blockSize);             \
        /* 2. Initialize Common parameters */                                                                   \
        op.InitCommon(lmcKeyValueCachePtr, slotMappingPtr, lmcTokenStride, lmcValueOffset, lmcBufferSize,       \
                      maxTokensPerLoop, numHeads, headDims, numTokens, blockSize, page2L, lmcTokensMajor, &pipe);\
        /* 3. Execute */                                                                                        \
        PAGED_KV_COPY_KERNEL_EXECUTE();                                                                         \
    }

// Declare support kernel entry at the device side
#define SINGLE_LAYER_PAGED_KV_COPY_V2_SEPARATE_TYPE_SLOTTYPE_DECLARE_DEVICE(TYPE)    \
    SINGLE_LAYER_PAGED_KV_COPY_V2_SEPARATE_TYPE_DECLARE(TYPE, int32_t);              \
    SINGLE_LAYER_PAGED_KV_COPY_V2_SEPARATE_TYPE_DECLARE(TYPE, int64_t);

// Supported Types instantiation
SINGLE_LAYER_PAGED_KV_COPY_V2_SEPARATE_TYPE_SLOTTYPE_DECLARE_DEVICE(half);
SINGLE_LAYER_PAGED_KV_COPY_V2_SEPARATE_TYPE_SLOTTYPE_DECLARE_DEVICE(int8_t);

#if (__CCE_AICORE__ >= 220)
SINGLE_LAYER_PAGED_KV_COPY_V2_SEPARATE_TYPE_SLOTTYPE_DECLARE_DEVICE(bfloat16_t);
#endif

namespace kvcache_ops {

// HostSide Declaration
#define SINGLE_LAYER_PAGED_KV_COPY_V2_SEPARATE_KERNEL_CALL(TYPE, SLOTTYPE)                                 \
    single_layer_paged_kv_copy_v2_separate_##TYPE##_##SLOTTYPE<<<blockDim, nullptr, stream>>>(             \
        lmcKeyValueCachePtr, vllmKeyPtr, vllmValuePtr, slotMappingPtr,                                     \
        keyBlockStride, valueBlockStride, vllmKeyBufferSize, vllmValueBufferSize,                          \
        lmcTokenStride, lmcValueOffset, lmcBufferSize, maxTokensPerLoop,                                   \
        numHeads, headDims, numTokens, blockSize, page2L, lmcTokensMajor);

template<typename T, typename SlotT>
void single_layer_paged_kernel_v2_separate(
    uint32_t blockDim, void* stream, uint8_t* lmcKeyValueCachePtr,
    uint8_t* vllmKeyPtr, uint8_t* vllmValuePtr, uint8_t* slotMappingPtr,
    const int64_t keyBlockStride, const int64_t valueBlockStride, const int64_t vllmKeyBufferSize,
    const int64_t vllmValueBufferSize, const int64_t lmcTokenStride, const int64_t lmcValueOffset,
    const int64_t lmcBufferSize, const int32_t maxTokensPerLoop, const int32_t numHeads,
    const int32_t headDims, const int32_t numTokens, const int32_t blockSize,
    const bool page2L, const bool lmcTokensMajor
);

#define SINGLE_LAYER_PAGED_KERNEL_V2_SEPARATE_CALL_TYPE_DECLARE(TYPE, SLOTTYPE)                            \
    template <>                                                                                            \
    void single_layer_paged_kernel_v2_separate<TYPE, SLOTTYPE>(                                            \
        uint32_t blockDim, void* stream, uint8_t* lmcKeyValueCachePtr,                                     \
        uint8_t* vllmKeyPtr, uint8_t* vllmValuePtr, uint8_t* slotMappingPtr,                               \
        const int64_t keyBlockStride, const int64_t valueBlockStride, const int64_t vllmKeyBufferSize,     \
        const int64_t vllmValueBufferSize, const int64_t lmcTokenStride, const int64_t lmcValueOffset,     \
        const int64_t lmcBufferSize, const int32_t maxTokensPerLoop, const int32_t numHeads,               \
        const int32_t headDims, const int32_t numTokens, const int32_t blockSize,                          \
        const bool page2L, const bool lmcTokensMajor)                                                      \
    {                                                                                                      \
        SINGLE_LAYER_PAGED_KV_COPY_V2_SEPARATE_KERNEL_CALL(TYPE, SLOTTYPE);                                \
    }

// Instantiate Host Callers
#define SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_SLOTTYPE_MLA_DECLARE_HOST(TYPE)      \
    SINGLE_LAYER_PAGED_KERNEL_V2_SEPARATE_CALL_TYPE_DECLARE(TYPE, int32_t);      \
    SINGLE_LAYER_PAGED_KERNEL_V2_SEPARATE_CALL_TYPE_DECLARE(TYPE, int64_t);

// Declare the kernel entry at the host side
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_SLOTTYPE_MLA_DECLARE_HOST(half);
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_SLOTTYPE_MLA_DECLARE_HOST(int8_t);
#if (ASCEND_AICORE_ARCH >= 220)
SINGLE_LAYER_PAGED_KERNEL_CALL_TYPE_SLOTTYPE_MLA_DECLARE_HOST(bfloat16_t);
#endif

// Dispatch Functions
template <typename T>
void dispatch_single_layer_kernel_v2_separate_on_slot_type(
        kvcache_ops::AscendType slotType, uint32_t blockDim, void* stream,
        uint8_t* lmcKeyValueCachePtr, uint8_t* vllmKeyPtr, uint8_t* vllmValuePtr,
        uint8_t* slotMappingPtr, const int64_t keyBlockStride, const int64_t valueBlockStride,
        const int64_t vllmKeyBufferSize, const int64_t vllmValueBufferSize, const int64_t lmcTokenStride,
        const int64_t lmcValueOffset, const int64_t lmcBufferSize, const int32_t maxTokensPerLoop,
        const int32_t numHeads, const int32_t headDims, const int32_t numTokens,
        const int32_t blockSize, const bool page2L, const bool lmcTokensMajor)
{
    switch(slotType) {
        case kvcache_ops::AscendType::INT32:
            single_layer_paged_kernel_v2_separate<T, int32_t>(
                blockDim, stream, lmcKeyValueCachePtr, vllmKeyPtr, vllmValuePtr, slotMappingPtr, keyBlockStride, valueBlockStride, vllmKeyBufferSize, vllmValueBufferSize,
                lmcTokenStride, lmcValueOffset, lmcBufferSize, maxTokensPerLoop, numHeads, headDims, numTokens, blockSize, page2L, lmcTokensMajor
            );
            break; 
        case kvcache_ops::AscendType::INT64:
            single_layer_paged_kernel_v2_separate<T, int64_t>(
                blockDim, stream, lmcKeyValueCachePtr, vllmKeyPtr, vllmValuePtr, slotMappingPtr, keyBlockStride, valueBlockStride, vllmKeyBufferSize, vllmValueBufferSize,
                lmcTokenStride, lmcValueOffset, lmcBufferSize, maxTokensPerLoop, numHeads, headDims, numTokens, blockSize, page2L, lmcTokensMajor
            );
            break;
        default:
            ASCENDC_REPORT_NOT_SUPPORT(false, std::to_string(static_cast<int>(slotType)) + " is not supported.")
            throw std::runtime_error("Slot type: " + std::to_string(static_cast<int>(slotType)) +" not supported.");
    }
}


// Public Entry Points (API)
extern void single_layer_kv_transfer_kernel_v2_separate(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType, uint32_t blockDim,
    void* stream, uint8_t* lmcKeyValueCachePtr, uint8_t* vllmKeyPtr,
    uint8_t* vllmValuePtr, uint8_t* slotMappingPtr, const int64_t keyBlockStride,
    const int64_t valueBlockStride, const int64_t vllmKeyBufferSize, const int64_t vllmValueBufferSize,
    const int64_t lmcTokenStride, const int64_t lmcValueOffset, const int64_t lmcBufferSize,
    const int32_t maxTokensPerLoop, const int32_t numHeads, const int32_t headDims,
    const int32_t numTokens, const int32_t blockSize, const bool page2L, const bool lmcTokensMajor)
{
    switch(type) {
        case kvcache_ops::AscendType::FP16:
            dispatch_single_layer_kernel_v2_separate_on_slot_type<half>(slotType, blockDim, stream, lmcKeyValueCachePtr, vllmKeyPtr, vllmValuePtr, slotMappingPtr,
                                                                        keyBlockStride, valueBlockStride,vllmKeyBufferSize, vllmValueBufferSize,lmcTokenStride, lmcValueOffset, 
                                                                        lmcBufferSize,maxTokensPerLoop, numHeads, headDims, numTokens, blockSize,page2L, lmcTokensMajor);
            break;
#if (ASCEND_AICORE_ARCH >= 220)
        case kvcache_ops::AscendType::BF16:
            dispatch_single_layer_kernel_v2_separate_on_slot_type<bfloat16_t>(slotType, blockDim, stream, lmcKeyValueCachePtr, vllmKeyPtr, vllmValuePtr, slotMappingPtr,
                                                                        keyBlockStride, valueBlockStride,vllmKeyBufferSize, vllmValueBufferSize,lmcTokenStride, lmcValueOffset, 
                                                                        lmcBufferSize,maxTokensPerLoop, numHeads, headDims, numTokens, blockSize,page2L, lmcTokensMajor);
            break;
#endif
        case kvcache_ops::AscendType::INT8:
            dispatch_single_layer_kernel_v2_separate_on_slot_type<int8_t>(slotType, blockDim, stream, lmcKeyValueCachePtr, vllmKeyPtr, vllmValuePtr, slotMappingPtr,
                                                                        keyBlockStride, valueBlockStride,vllmKeyBufferSize, vllmValueBufferSize,lmcTokenStride, lmcValueOffset, 
                                                                        lmcBufferSize,maxTokensPerLoop, numHeads, headDims, numTokens, blockSize,page2L, lmcTokensMajor);
            break;
        default:
            return;
    }
}

} // namespace kvcache_ops
