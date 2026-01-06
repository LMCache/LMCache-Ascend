#include "utils.h"
#include "dcmi_management.h"
#include <stdexcept>
#include <string>

#include <Python.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace vllm_ascend {
kvcache_ops::AscendType get_dtype_from_torch(at::ScalarType scalarType) {
  if (scalarType == at::ScalarType::Float) {
    return kvcache_ops::AscendType::FP32;
  } else if (scalarType == at::ScalarType::BFloat16) {
    return kvcache_ops::AscendType::BF16;
  } else if (scalarType == at::ScalarType::Half) {
    return kvcache_ops::AscendType::FP16;
  } else if (scalarType == at::ScalarType::Long) {
    return kvcache_ops::AscendType::INT64;
  } else if (scalarType == at::ScalarType::Int) {
    return kvcache_ops::AscendType::INT32;
  } else {
    TORCH_CHECK(false, "ScalarType not supported.");
  }
}
} // namespace vllm_ascend

MultiLayerKVConfig prepare_multi_layer_kv_config(
    const torch::Tensor &key_value, const torch::Tensor &key_value_ptrs,
    const torch::Tensor &slot_mapping, const torch::Device &paged_memory_device,
    int page_buffer_size, bool direction, bool use_mla,
    int kvcache_format_raw) {
  MultiLayerKVConfig config;

  // it is actually a uint8_t**. we will reinterpret it inside the kernel
  config.page_buffer_ptrs =
      get_kernel_ptr<uint8_t, const torch::Tensor>(key_value_ptrs);
  config.slot_mapping_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(slot_mapping);

  config.num_layers = key_value.size(1);
  config.num_tokens = slot_mapping.size(0);
  config.hidden_dims = key_value.size(-1);
  config.kv_size = use_mla ? 1 : 2;

  config.kvcache_format =
      static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw);

  config.page_buffer_size = page_buffer_size;
  config.direction = direction;

  config.scalar_type = key_value.scalar_type();
  config.slot_type = slot_mapping.scalar_type();

  config.socName = aclrtGetSocName();

  return config;
}

SingleLayerKVConfig prepare_single_layer_kv_config(
    const torch::Tensor &lmc_cache, const torch::Tensor &vllm_cache,
    const torch::Tensor &slot_mapping, bool direction, bool token_major,
    bool vllm_two_major) {
  SingleLayerKVConfig config;

  config.lmc_cache_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(lmc_cache);
  config.vllm_cache_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(vllm_cache);
  config.slot_mapping_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(slot_mapping);

  config.num_tokens = slot_mapping.size(0);
  config.num_heads = vllm_cache.size(-2);
  config.head_dims = vllm_cache.size(-1);
  config.block_size = vllm_cache.size(-3);
  config.kv_size = 2;

  bool is_mla = false;
  if (token_major) {
    is_mla = lmc_cache.size(1) == 1;
  } else {
    is_mla = lmc_cache.size(0) == 1;
  }

  if (is_mla) {
    PyErr_SetString(PyExc_RuntimeError,
                    "MLA is not supported yet. Please contact LMCache Ascend.");
    throw py::error_already_set();
  }

  config.scalar_type = vllm_cache.scalar_type();
  config.slot_type = slot_mapping.scalar_type();
  config.socName = aclrtGetSocName();

  config.direction = direction;
  config.token_major = token_major;

  config.lmc_buffer_size = static_cast<int64_t>(lmc_cache.nbytes());
  config.vllm_buffer_size = static_cast<int64_t>(vllm_cache.nbytes());

  return config;
}

void compute_multi_layer_ub_params(MultiLayerKVConfig &config,
                                   const torch::Tensor &key_value,
                                   const torch::Device &paged_memory_device,
                                   const torch::Tensor &key_value_ptrs) {
  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  // we require the kv ptr list to be on the device too
  const c10::OptionalDeviceGuard kv_device_guard(device_of(key_value_ptrs));

  config.stream = c10_npu::getCurrentNPUStream().stream();

  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(config.socName);
  uint64_t ubSize;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  // we only launched with at most 4 aiv
  config.aiv_num = static_cast<uint32_t>(std::min(config.num_layers, 4));

  constexpr int32_t numBuffsOnDev = 2;
  // step 1. use per tokens buff size to derive how many tokens can be allocated
  // per loop
  int64_t baseBuffSize =
      numBuffsOnDev * config.hidden_dims * key_value.element_size();

  if (ubSize < static_cast<uint64_t>(baseBuffSize)) {
    std::string errStr =
        "Per TokenBuffer Size: " + std::to_string(baseBuffSize) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError,
                    (errStr + " Please contact us.").c_str());
    throw py::error_already_set();
  }

  // step 2. work out how many tokens per loop
  config.maxTokensPerLoop =
      static_cast<int32_t>(ubSize / baseBuffSize) -
      1; // Subtract 1 to provide a safety margin and avoid over-allocating the
         // UB buffer, ensuring we do not exceed hardware limits due to possible
         // rounding or small additional allocations.
  config.maxTokensPerLoop = std::min(config.maxTokensPerLoop,
                                     static_cast<int32_t>(config.num_tokens));

  // step 3. double check whether the perloop buffer can accommodate everything
  int64_t totalPerLoopBuffer =
      static_cast<int64_t>(config.maxTokensPerLoop) * baseBuffSize;
  if (ubSize < static_cast<uint64_t>(totalPerLoopBuffer)) {
    std::string errStr =
        "Per Loop Buffer Size: " + std::to_string(totalPerLoopBuffer) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError,
                    (errStr + " Please contact us.").c_str());
    throw py::error_already_set();
  }

  // using double buffs mean we actually want to allocate half of this per
  // round.
  config.singlePerLoopBuffer = totalPerLoopBuffer / numBuffsOnDev;
}

void compute_single_layer_ub_params(SingleLayerKVConfig &config,
                                    const torch::Tensor &vllm_cache) {
  const c10::OptionalDeviceGuard device_guard(device_of(vllm_cache));

  config.stream = c10_npu::getCurrentNPUStream().stream();

  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(config.socName);
  uint64_t ubSize;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  config.aiv_num =
      static_cast<uint32_t>(std::min(4, static_cast<int>(config.num_tokens)));

  uint32_t numBuffsOnDev = 2;
  // each token buffer is kv * heads * headdims * size
  uint64_t baseBuffSize = numBuffsOnDev * config.kv_size * config.num_heads *
                          config.head_dims * vllm_cache.element_size();

  // first check whether one token k cache actually passes the ub
  if (ubSize < baseBuffSize) {
    std::string errStr =
        "Per Token Cache Buffer Size: " + std::to_string(baseBuffSize) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError,
                    (errStr + " Please contact LMCache Ascend.").c_str());
    throw py::error_already_set();
  }

  // we are going to work out how many tokens to copy maximally per innerloop
  config.max_tokens_per_loop = static_cast<int32_t>(ubSize / baseBuffSize);
  config.max_tokens_per_loop =
      std::min(config.max_tokens_per_loop, config.num_tokens);
}

void compute_single_layer_strides(SingleLayerKVConfig &config,
                                  const torch::Tensor &lmc_cache,
                                  const torch::Tensor &vllm_cache,
                                  bool token_major, bool vllm_two_major) {
  // LMC buffer strides
  if (token_major) {
    // [tokens, 2, heads*headdim]
    config.lmc_token_stride = lmc_cache.stride(0);
    config.lmc_value_offset = lmc_cache.stride(1);
  } else {
    // [2, tokens, heads*headdim]
    config.lmc_token_stride = lmc_cache.stride(1);
    config.lmc_value_offset = lmc_cache.stride(0);
  }

  // vLLM buffer strides
  if (vllm_two_major) {
    // [2, num_blocks, block_size, num_heads, head_size]
    config.vllm_block_stride = vllm_cache.stride(1);
    config.vllm_value_offset = vllm_cache.stride(0);
  } else {
    // [num_blocks, 2, block_size, num_heads, head_size]
    config.vllm_block_stride = vllm_cache.stride(0);
    config.vllm_value_offset = vllm_cache.stride(1);
  }
}