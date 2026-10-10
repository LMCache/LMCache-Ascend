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

#define MULTI_LAYER_PAGED_KV_COPY_KERNEL_NAME(TYPE, SLOTTYPE, FMT) \
    multi_layer_paged_kv_copy_##TYPE##_##SLOTTYPE##_##FMT


#define MULTI_LAYER_PAGED_KV_COPY_DECLARE(TYPE, SLOTTYPE, FMT)                                            \
    extern "C" __global__ __aicore__ void MULTI_LAYER_PAGED_KV_COPY_KERNEL_NAME(TYPE, SLOTTYPE, FMT)(     \
        __gm__ uint8_t* pagedKVCaches, __gm__ uint8_t* dstCacheTensor, __gm__ uint8_t* slotmappings,      \
        const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers, const int64_t pageBuffSize, \
        const int32_t numTokensChunk, const int coreNum, const bool page2L)                               \
    {                                                                                                     \
        AscendC::TPipe pipe;                                                                              \
        kvcache_ops::MultiLayerPagedKVCopyProcessor<TYPE, SLOTTYPE,                                       \
            kvcache_ops::StandardPolicy<TYPE, SLOTTYPE, kvcache_ops::KVCacheFormat::FMT>> op;             \
                                                                                                          \
        op.InitCommon(pagedKVCaches, dstCacheTensor, slotmappings, &pipe, hiddenDims, numLayers,          \
                      pageBuffSize, numTokensChunk, page2L, kvs);                                         \
        op.process(coreNum);                                                                              \
    }

#define EXPAND_FMT_STANDARD(TYPE, SLOTTYPE) \
    MULTI_LAYER_PAGED_KV_COPY_DECLARE(TYPE, SLOTTYPE, MERGED_KV) \
    MULTI_LAYER_PAGED_KV_COPY_DECLARE(TYPE, SLOTTYPE, SEPARATE_KV)

#define EXPAND_SLOT_STANDARD(TYPE) \
    EXPAND_FMT_STANDARD(TYPE, int32_t) \
    EXPAND_FMT_STANDARD(TYPE, int64_t)

EXPAND_SLOT_STANDARD(half)
EXPAND_SLOT_STANDARD(int8_t)
#if (__CCE_AICORE__ >= 220)
EXPAND_SLOT_STANDARD(bfloat16_t)
#endif

// Host Side 
namespace kvcache_ops {

#define SPECIALIZE_KERNEL_LAUNCHER_STANDARD(TYPE, SLOTTYPE, FMT)                                       \
template<>                                                                                             \
struct StandardLauncher<TYPE, SLOTTYPE, KVCacheFormat::FMT> {                                          \
    static void Launch(uint32_t blockDim, void *stream,                                                \
                      uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings,          \
                      const StandardConfig& config,                                                    \
                      int64_t kHiddenDims = 0, int64_t vHiddenDims = 0, int64_t dsaHiddenDims = 0)     \
    {                                                                                                  \
        (void)kHiddenDims; (void)vHiddenDims; (void)dsaHiddenDims;                                     \
        MULTI_LAYER_PAGED_KV_COPY_KERNEL_NAME(TYPE, SLOTTYPE, FMT)<<<blockDim, nullptr, stream>>>(     \
        pagedKVCaches, dstCacheTensor, slotmappings, config.hiddenDims, config.kvs, config.numLayers,  \
        config.pageBuffSize, config.numTokensChunk, blockDim, config.page2L);                          \
    }                                                                                                  \
};

#define SPECIALIZE_KERNEL_LAUNCHER_STANDARD_UNSUPPORTED(TYPE, SLOTTYPE, FMT)                           \
template<>                                                                                             \
struct StandardLauncher<TYPE, SLOTTYPE, KVCacheFormat::FMT> {                                          \
    static void Launch(uint32_t blockDim, void *stream,                                                \
                      uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings,          \
                      const StandardConfig& config,                                                    \
                      int64_t kHiddenDims = 0, int64_t vHiddenDims = 0, int64_t dsaHiddenDims = 0)     \
    {                                                                                                  \
        (void)blockDim; (void)stream; (void)pagedKVCaches; (void)dstCacheTensor; (void)slotmappings;   \
        (void)config; (void)kHiddenDims; (void)vHiddenDims; (void)dsaHiddenDims;                       \
        ASCENDC_REPORT_NOT_SUPPORT(false, #FMT " is not supported in StandardLauncher.");              \
        throw std::runtime_error(#FMT " format is not supported in StandardLauncher.");                \
    }                                                                                                  \
};

#define EXPAND_LAUNCHER_STANDARD_FMT(TYPE, SLOTTYPE) \
    SPECIALIZE_KERNEL_LAUNCHER_STANDARD(TYPE, SLOTTYPE, MERGED_KV) \
    SPECIALIZE_KERNEL_LAUNCHER_STANDARD(TYPE, SLOTTYPE, SEPARATE_KV) \
    SPECIALIZE_KERNEL_LAUNCHER_STANDARD_UNSUPPORTED(TYPE, SLOTTYPE, MLA_KV) \
    SPECIALIZE_KERNEL_LAUNCHER_STANDARD_UNSUPPORTED(TYPE, SLOTTYPE, DSA_KV)

#define EXPAND_LAUNCHER_STANDARD_SLOT(TYPE) \
    EXPAND_LAUNCHER_STANDARD_FMT(TYPE, int32_t) \
    EXPAND_LAUNCHER_STANDARD_FMT(TYPE, int64_t)

EXPAND_LAUNCHER_STANDARD_SLOT(half)
EXPAND_LAUNCHER_STANDARD_SLOT(int8_t)
#if (ASCEND_AICORE_ARCH >= 220)
EXPAND_LAUNCHER_STANDARD_SLOT(bfloat16_t)
#endif

extern void multi_layer_kv_transfer_kernel(kvcache_ops::AscendType type, kvcache_ops::AscendType slotType, const kvcache_ops::KVCacheFormat kvcacheFormat,
                                           uint32_t blockDim, void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, 
                                           uint8_t *slotmappings, const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers, 
                                           const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L,
                                           int64_t kHiddenDims = 0, int64_t vHiddenDims = 0, int64_t dsaHiddenDims = 0)
{
    auto config = MakeStandardConfig(
        hiddenDims, numLayers, pageBuffSize, numTokensChunk, kvs, page2L
    );

    switch(type) {
        case AscendType::FP16:
            dispatch_paged_kernel_on_slot_type<StandardLauncher, half>(
                slotType, kvcacheFormat, blockDim, stream, 
                pagedKVCaches, dstCacheTensor, slotmappings, config,
                kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
        case AscendType::INT8:
            dispatch_paged_kernel_on_slot_type<StandardLauncher, int8_t>(
                slotType, kvcacheFormat, blockDim, stream, 
                pagedKVCaches, dstCacheTensor, slotmappings, config,
                kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
#if (ASCEND_AICORE_ARCH >= 220)
        case AscendType::BF16:
            dispatch_paged_kernel_on_slot_type<StandardLauncher, bfloat16_t>(
                slotType, kvcacheFormat, blockDim, stream, 
                pagedKVCaches, dstCacheTensor, slotmappings, config,
                kHiddenDims, vHiddenDims, dsaHiddenDims);
            break;
#endif
        default:
            ASCENDC_REPORT_NOT_SUPPORT(false, 
                std::to_string(static_cast<int>(type)) + " is not supported.");
            throw std::runtime_error(
                "Scalar type: " + std::to_string(static_cast<int>(type)) + " not supported.");
    }
}

} // namespace kvcache_ops