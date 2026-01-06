/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#pragma once

#include "kernels/types.h"
#include "managed_mem.h"
#include <c10/core/ScalarType.h>
#include <string>
#include <torch/torch.h>

#include "tiling/platform/platform_ascendc.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace vllm_ascend {
kvcache_ops::AscendType get_dtype_from_torch(at::ScalarType scalarType);
} // namespace vllm_ascend

template <typename T, typename TENSOR_TYPE>
T *get_kernel_ptr(TENSOR_TYPE &tensor) {
  torch::Device device = tensor.device();
  // NPU should be using PrivateUse1
  if (device.is_privateuseone() || device.is_cuda()) {
    return static_cast<T *>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    // find device ptr based on the host pinned ptr
    // because acl does not currently support HostGetDevicePointer API
    void *devPtr = get_device_ptr(tensor.data_ptr());
    TORCH_CHECK(
        devPtr != nullptr,
        "Unable to retrieve device ptr, is this a host registered pointer ?");
    return reinterpret_cast<T *>(devPtr);
  } else {
    TORCH_CHECK(
        false,
        "Invalid device. Device must be ascend (PrivateUseOne) or pinned cpu.");
  }
}

struct MultiLayerKVConfig {
  uint8_t *page_buffer_ptrs;
  uint8_t *slot_mapping_ptr;

  int num_layers;
  int num_tokens;
  int hidden_dims;
  int kv_size;

  kvcache_ops::KVCacheFormat kvcache_format;

  aclrtStream stream;
  at::ScalarType scalar_type;
  at::ScalarType slot_type;
  const char *socName;

  uint32_t aiv_num;
  int32_t maxTokensPerLoop;
  int64_t singlePerLoopBuffer;

  int page_buffer_size;
  bool direction;
};

MultiLayerKVConfig prepare_multi_layer_kv_config(
    const torch::Tensor &key_value, const torch::Tensor &key_value_ptrs,
    const torch::Tensor &slot_mapping, const torch::Device &paged_memory_device,
    int page_buffer_size, bool direction, bool use_mla, int kvcache_format_raw);

void compute_multi_layer_ub_params(MultiLayerKVConfig &config,
                                   const torch::Tensor &key_value,
                                   const torch::Device &paged_memory_device,
                                   const torch::Tensor &key_value_ptrs);

struct SingleLayerKVConfig {

  uint8_t *lmc_cache_ptr;
  uint8_t *vllm_cache_ptr;
  uint8_t *slot_mapping_ptr;

  int32_t num_tokens;
  int32_t num_heads;
  int32_t head_dims;
  int32_t block_size;
  int16_t kv_size;

  aclrtStream stream;
  at::ScalarType scalar_type;
  at::ScalarType slot_type;
  const char *socName;

  uint32_t aiv_num;
  int32_t max_tokens_per_loop;

  int64_t lmc_token_stride;
  int64_t lmc_value_offset;
  int64_t vllm_block_stride;
  int64_t vllm_value_offset;

  int64_t lmc_buffer_size;
  int64_t vllm_buffer_size;

  bool direction;
  bool token_major;
};

SingleLayerKVConfig prepare_single_layer_kv_config(
    const torch::Tensor &lmc_cache, const torch::Tensor &vllm_cache,
    const torch::Tensor &slot_mapping, bool direction, bool token_major,
    bool vllm_two_major);

void compute_single_layer_ub_params(SingleLayerKVConfig &config,
                                    const torch::Tensor &vllm_cache);

void compute_single_layer_strides(SingleLayerKVConfig &config,
                                  const torch::Tensor &lmc_cache,
                                  const torch::Tensor &vllm_cache,
                                  bool token_major, bool vllm_two_major);