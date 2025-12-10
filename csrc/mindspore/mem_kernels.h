#pragma once
#include "managed_mem.h"
#include "kernels/types.h"
#include "ms_extension/api.h"

using BaseTensor = mindspore::tensor::BaseTensor;
using BaseTensorPtr = mindspore::tensor::BaseTensorPtr;
using PyBoostUtils = mindspore::kernel::pyboost::PyBoostUtils;

namespace kvcache_ops {
void multi_layer_kv_transfer_kernel(kvcache_ops::AscendType type, kvcache_ops::AscendType slotType, const kvcache_ops::KVCacheFormat kvcache_format,
                                    uint32_t blockDim, void *stream, uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, 
                                    uint8_t *slotmappings,const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,
                                    const int64_t pageBuffSize, const int32_t numTokensChunk, const bool page2L);

void single_layer_kv_transfer_kernel(kvcache_ops::AscendType type, kvcache_ops::AscendType slotType, 
                                     uint32_t blockDim, void *stream, uint8_t *dstCacheTensor, 
                                     uint8_t *keyCachePtr, uint8_t *valueCachePtr,
                                     uint8_t *slotmappings, const int64_t hiddenDims, const int32_t numTokens, 
                                     const bool page2L, const bool tokenMajor, const bool isMLA);

void load_and_reshape_flash_kernel(kvcache_ops::AscendType type, kvcache_ops::AscendType slotType, 
                                  uint32_t blockDim, void *stream, uint8_t *dstCacheTensor, uint8_t *keyCachePtr, 
                                  uint8_t *valueCachePtr, uint8_t *slotmappings, const int64_t hiddenDims, 
                                  const int64_t numPages, const int32_t pagedSize, const int32_t numTokens, 
                                  const int32_t numLayers, const int32_t layerIdx, const bool page2L);
}


void multi_layer_kv_transfer(py::array& key_value, // [kv, num_layer, num_tokens, hidden]
                             ms::Tensor key_value_ptrs, // [num_layers]
                             ms::Tensor slot_mapping, // [num_tokens]
                             const int page_buffer_size, const bool direction,
                             const bool use_mla, const int kvcache_format_raw);

// void multi_layer_kv_transfer_unilateral(ms::Tensor& key_value,
//                                         const ms::Tensor& key_ptrs,
//                                         const ms::Tensor& value_ptrs,
//                                         const ms::Tensor& slot_mapping,
//                                         const ms::Device& paged_memory_device,
//                                         const int page_buffer_size,
//                                         const bool direction);

// void single_layer_kv_transfer(py::array& key_value,
//                               ms::Tensor vllm_key_cache,
//                               ms::Tensor vllm_value_cache,
//                               ms::Tensor slot_mapping,
//                               const bool direction,
//                               const bool token_major = false);

// void load_and_reshape_flash(ms::Tensor& key_value, ms::Tensor& key_cache,
//                             ms::Tensor& value_cache,
//                             ms::Tensor& slot_mapping, const int layer_idx);

// void reshape_and_cache_back_flash(ms::Tensor& key_value,
//                                   ms::Tensor& key_cache,
//                                   ms::Tensor& value_cache,
//                                   ms::Tensor& slot_mapping,
//                                   const int layer_idx);