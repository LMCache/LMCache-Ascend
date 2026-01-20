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

void compute_single_layer_ub_params(const KVTransferDims &dims,
                                    KVTransferUBParams &ub_params,
                                    const torch::Tensor &vllm_cache) {

  const c10::OptionalDeviceGuard device_guard(device_of(vllm_cache));

  ub_params.stream = c10_npu::getCurrentNPUStream().stream();
  const char *socName = aclrtGetSocName();

  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  uint64_t ubSize;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  ub_params.aiv_num = static_cast<uint32_t>(std::min(4, dims.num_tokens));

  uint32_t numBuffsOnDev = 2;
  // each token buffer is kv * heads * headdims * size
  uint64_t baseBuffSize = numBuffsOnDev * dims.kv_size * dims.num_heads *
                          dims.head_dims * vllm_cache.element_size();

  if (ubSize < baseBuffSize) {
    std::string errStr =
        "Per Token Cache Buffer Size: " + std::to_string(baseBuffSize) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError,
                    (errStr + " Please contact LMCache Ascend.").c_str());
    throw py::error_already_set();
  }

  // we are going to work out how many tokens to copy maximally per innerloop
  ub_params.max_tokens_per_loop = static_cast<int32_t>(ubSize / baseBuffSize);
  ub_params.max_tokens_per_loop =
      std::min(ub_params.max_tokens_per_loop, dims.num_tokens);
}

void compute_single_layer_strides(
    const KVTransferDims &dims, KVTransferStrides &strides,
    const torch::Tensor &lmc_cache,
    const torch::Tensor &vllm_k_cache, // for merged
    bool token_major, bool vllm_two_major, bool is_separate,
    const torch::Tensor *vllm_v_cache) { // for separate

  // LMC strides
  if (token_major) {
    // Shape: [tokens, 2, heads*headdim]
    strides.lmc_token_stride = lmc_cache.stride(0);
    strides.lmc_val_offset = lmc_cache.stride(1);
  } else {
    // Shape: [2, tokens, heads*headdim]
    strides.lmc_token_stride = lmc_cache.stride(1);
    strides.lmc_val_offset = lmc_cache.stride(0);
  }
  strides.lmc_bytes = static_cast<int64_t>(lmc_cache.nbytes());

  // vLLM buffer strides
  if (!is_separate) {
    if (vllm_two_major) {
      // Shape: [2, num_blocks, block_size, heads, head_dims]
      strides.vllm_k_stride = vllm_k_cache.stride(1);   // Block stride
      strides.vllm_val_offset = vllm_k_cache.stride(0); // K to V offset
    } else {
      // Shape: [num_blocks, 2, block_size, heads, head_dims]
      strides.vllm_k_stride = vllm_k_cache.stride(0);   // Block stride
      strides.vllm_val_offset = vllm_k_cache.stride(1); // K to V offset
    }
    strides.vllm_k_bytes = static_cast<int64_t>(vllm_k_cache.nbytes());

    // only for separate
    strides.vllm_v_stride = 0;
    strides.vllm_v_bytes = 0;

  } else {
    // SEPARATE
    TORCH_CHECK(vllm_v_cache != nullptr,
                "vllm_v_cache required for SEPARATE format");

    // Shape: [num_blocks, block_size, heads, head_dims]
    strides.vllm_k_stride = vllm_k_cache.stride(0);
    strides.vllm_v_stride = vllm_v_cache->stride(0);

    strides.vllm_k_bytes = static_cast<int64_t>(vllm_k_cache.nbytes());
    strides.vllm_v_bytes = static_cast<int64_t>(vllm_v_cache->nbytes());

    // only for merged
    strides.vllm_val_offset = 0;
  }
}

SingleLayerKVConfig prepare_single_layer_kv_config(
    torch::Tensor &lmc_dst_cache, std::vector<torch::Tensor> &vllm_kv_caches,
    torch::Tensor &slot_mapping, bool direction, bool token_major,
    bool vllm_two_major, bool is_separate) {

  SingleLayerKVConfig config;

  torch::Tensor &vllm_k_cache = vllm_kv_caches[0];
  torch::Tensor *vllm_v_cache = is_separate ? &vllm_kv_caches[1] : nullptr;

  // Dims
  config.dims.num_tokens = slot_mapping.size(0);
  config.dims.num_heads = vllm_k_cache.size(-2);
  config.dims.head_dims = vllm_k_cache.size(-1);
  config.dims.block_size = vllm_k_cache.size(-3);
  config.dims.kv_size = 2;

  // ptrs
  config.ptrs.lmc_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(lmc_dst_cache);
  config.ptrs.vllm_k_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(vllm_k_cache);
  config.ptrs.slot_mapping_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(slot_mapping);

  if (vllm_v_cache != nullptr) {
    config.ptrs.vllm_v_ptr =
        get_kernel_ptr<uint8_t, const torch::Tensor>(*vllm_v_cache);
  } else {
    config.ptrs.vllm_v_ptr = nullptr;
  }

  config.ub_params.scalar_type_num =
      vllm_ascend::get_dtype_from_torch(vllm_k_cache.scalar_type());
  config.ub_params.slot_type_num =
      vllm_ascend::get_dtype_from_torch(slot_mapping.scalar_type());

  config.direction = direction;
  config.token_major = token_major;

  // MLA
  bool is_mla = false;
  if (token_major) {
    is_mla = lmc_dst_cache.size(1) == 1; // [tokens, 1, hidden]
  } else {
    is_mla = lmc_dst_cache.size(0) == 1; // [1, tokens, hidden]
  }
  if (is_mla) {
    PyErr_SetString(PyExc_RuntimeError, "MLA is not supported yet.");
    throw py::error_already_set();
  }

  // Compute UB Params
  compute_single_layer_ub_params(config.dims, config.ub_params, vllm_k_cache);

  // Compute Strides
  compute_single_layer_strides(config.dims, config.strides, lmc_dst_cache,
                               vllm_k_cache, token_major, vllm_two_major,
                               is_separate, vllm_v_cache);

  return config;
}

HostChunkMetadata
prepare_host_chunk_metadata(const std::vector<torch::Tensor> &lmc_tensors,
                            const std::vector<int64_t> &chunk_sizes,
                            const KVTransferStrides &strides,
                            int64_t element_size, bool token_major) {

  size_t num_chunks = lmc_tensors.size();
  HostChunkMetadata meta;
  meta.ptrs.resize(num_chunks);
  meta.copy_sizes.resize(num_chunks);
  meta.v_offsets.resize(num_chunks);
  meta.element_size = element_size;
  meta.bytes_per_token = strides.lmc_token_stride * element_size;

  for (size_t i = 0; i < num_chunks; ++i) {
    meta.ptrs[i] = static_cast<uint8_t *>(lmc_tensors[i].data_ptr());
    meta.copy_sizes[i] = chunk_sizes[i] * meta.bytes_per_token;

    if (!token_major) {
      meta.v_offsets[i] = lmc_tensors[i].stride(0) * element_size;
    } else {
      meta.v_offsets[i] = 0;
    }
  }
  return meta;
}

void execute_batched_memcpy(
    const SingleLayerKVConfig &config, const HostChunkMetadata &meta,
    const std::vector<int64_t> &chunk_offsets,
    bool is_d2h // true: Device to Host, false: Host to Device
) {
  aclError ret;
  size_t num_chunks = meta.ptrs.size();
  aclrtMemcpyKind kind =
      is_d2h ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;

  // only for !token_major
  int64_t staging_v_plane_offset =
      config.strides.lmc_val_offset * meta.element_size;
  aclrtStream stream = config.ub_params.stream;

  for (size_t i = 0; i < num_chunks; ++i) {
    uint8_t *staging_ptr =
        config.ptrs.lmc_ptr + chunk_offsets[i] * meta.bytes_per_token;
    uint8_t *host_ptr = meta.ptrs[i];
    int64_t size = meta.copy_sizes[i];

    if (config.token_major) {
      if (!is_d2h) // H2D
        ret = aclrtMemcpyAsync(staging_ptr, size, host_ptr, size, kind, stream);
      else // D2H
        ret = aclrtMemcpyAsync(host_ptr, size, staging_ptr, size, kind, stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy failed (TokenMajor) chunk ", i,
                  " ret=", ret);
    } else {
      // K Plane
      if (!is_d2h)
        ret = aclrtMemcpyAsync(staging_ptr, size, host_ptr, size, kind, stream);
      else
        ret = aclrtMemcpyAsync(host_ptr, size, staging_ptr, size, kind, stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (K) failed chunk ", i,
                  " ret=", ret);

      // V Plane
      uint8_t *staging_v = staging_ptr + staging_v_plane_offset;
      uint8_t *host_v = host_ptr + meta.v_offsets[i];

      if (!is_d2h)
        ret = aclrtMemcpyAsync(staging_v, size, host_v, size, kind, stream);
      else
        ret = aclrtMemcpyAsync(host_v, size, staging_v, size, kind, stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE, "Memcpy (V) failed chunk ", i,
                  " ret=", ret);
    }
  }
}

bool validate_vllm_caches(const std::vector<torch::Tensor> &vllm_kv_caches,
                          int kvcache_format_raw) {
  kvcache_ops::KVCacheFormat format =
      static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw);
  bool is_separate = (format == kvcache_ops::KVCacheFormat::SEPARATE_KV);

  if (!is_separate && format != kvcache_ops::KVCacheFormat::MERGED_KV) {
    std::string err =
        "Invalid KV cache format: " + std::to_string(kvcache_format_raw) +
        ". Expected 1 (MERGED_KV) or 2 (SEPARATE_KV)";
    PyErr_SetString(PyExc_ValueError, err.c_str());
    throw py::error_already_set();
  }

  if (vllm_kv_caches.empty()) {
    PyErr_SetString(PyExc_ValueError, "vllm_kv_caches cannot be empty");
    throw py::error_already_set();
  }

  if (is_separate) {
    if (vllm_kv_caches.size() != 2) {
      PyErr_SetString(PyExc_ValueError,
                      "SEPARATE_KV expects 2 tensors (K and V).");
      throw py::error_already_set();
    }
    if (vllm_kv_caches[0].sizes() != vllm_kv_caches[1].sizes()) {
      throw py::value_error("K and V caches must have the same shape.");
    }
  } else {
    if (vllm_kv_caches.size() != 1) {
      PyErr_SetString(PyExc_ValueError, "MERGED_KV expects 1 tensor.");
      throw py::error_already_set();
    }
  }

  return is_separate;
}