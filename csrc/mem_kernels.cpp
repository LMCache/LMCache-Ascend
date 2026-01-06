#include "mem_kernels.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>

namespace py = pybind11;

/**
 * Quickly offload KV cache from vLLM paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in vLLM's KV buffer has a shape of
 * [2, PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each AIV Core processes the copy for a token
 *
 * Therefore:
 *  AIV Core - token
 *
 * The function does:
 * slot_id = slot_mapping[tokenId]
 * ptrs[mem_offset(kv, layer, tokenId, hiddenDims)] = key_value[mem_offset(kv,
 * layer, pages, pageSize, slot_id, hiddenDims)]
 *
 * Param:
 *  - direction: false  means LMCache to PagedBuffer, true  means PagedBuffer to
 * LMCache
 */
void multi_layer_kv_transfer(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla, const int kvcache_format_raw) {
  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw);

  // Calculate UB buffer parameters
  compute_multi_layer_ub_params(config, key_value, paged_memory_device,
                                key_value_ptrs);

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([config, key_value_ptr]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);

    kvcache_ops::multi_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, config.kvcache_format, config.aiv_num,
        config.stream, config.page_buffer_ptrs, key_value_ptr,
        config.slot_mapping_ptr, config.hidden_dims, config.kv_size,
        config.num_layers, config.page_buffer_size, config.num_tokens,
        config.singlePerLoopBuffer, config.maxTokensPerLoop, config.direction);
    return 0;
  });
  cmd.Run();
  return;
};

void fused_multi_layer_kv_transfer(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    torch::Tensor &staging_cache,        // staging buffer
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, // true: from_gpu, false: to_gpu
    const bool use_mla, const int kvcache_format_raw) {
  // get host cpu buffer pointer for aclrtMemcpyAsync
  uint8_t *key_value_ptr = static_cast<uint8_t *>(key_value.data_ptr());
  uint8_t *staging_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(staging_cache);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw);

  compute_multi_layer_ub_params(config, key_value, paged_memory_device,
                                key_value_ptrs);

  // Calculate and verify the CPU buffer size
  size_t cpu_buffer_size = static_cast<size_t>(config.kv_size) *
                           config.num_layers * config.num_tokens *
                           config.hidden_dims * key_value.element_size();

  TORCH_CHECK(
      staging_cache.numel() * staging_cache.element_size() >= cpu_buffer_size,
      "staging_cache size insufficient: need ", cpu_buffer_size, " bytes, got ",
      staging_cache.numel() * staging_cache.element_size());

  at_npu::native::OpCommand cmd;
  cmd.Name("fused_multi_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([config, staging_cache_ptr, key_value_ptr,
                        cpu_buffer_size]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);

    aclError ret;
    // direction: false = to_gpu (H2D), true = from_gpu (D2H)
    bool isH2D = !config.direction;

    // Step 1: H2D memcpy (to_gpu) currently not used
    if (isH2D) {
      ret = aclrtMemcpyAsync(staging_cache_ptr, cpu_buffer_size, key_value_ptr,
                             cpu_buffer_size, ACL_MEMCPY_HOST_TO_DEVICE,
                             config.stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE,
                  "H2D memcpy failed: cpu_buffer -> staging_cache, ret=", ret);
    }

    // Step 2: Kernel (Gather or Scatter)
    kvcache_ops::multi_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, config.kvcache_format, config.aiv_num,
        config.stream, config.page_buffer_ptrs, staging_cache_ptr,
        config.slot_mapping_ptr, config.hidden_dims, config.kv_size,
        config.num_layers, config.page_buffer_size, config.num_tokens,
        config.singlePerLoopBuffer, config.maxTokensPerLoop, config.direction);

    // Step 3: D2H memcpy (from_gpu)
    if (!isH2D) {
      ret = aclrtMemcpyAsync(key_value_ptr, cpu_buffer_size, staging_cache_ptr,
                             cpu_buffer_size, ACL_MEMCPY_DEVICE_TO_HOST,
                             config.stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE,
                  "D2H memcpy failed: staging_cache -> cpu_buffer, ret=", ret);
    }

    return 0;
  });
  cmd.Run();
  return;
}

void multi_layer_kv_transfer_310p(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla, const int num_kv_head,
    const int head_size, const int blockSize) {
  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
  // it is actually a uint8_t**. we will reinterpret it inside the kernel
  uint8_t *page_buffer_ptrs =
      get_kernel_ptr<uint8_t, const torch::Tensor>(key_value_ptrs);
  uint8_t *slot_mapping_ptr =
      get_kernel_ptr<uint8_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = slot_mapping.size(0);
  int hidden_dims = key_value.size(-1);
  int kv_size = 2;
  if (use_mla) {
    kv_size = 1;
  }

  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  // we require the kv ptr list to be on the device too
  const c10::OptionalDeviceGuard kv_device_guard(device_of(key_value_ptrs));

  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at::ScalarType scalar_type = key_value.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();
  const char *socName = aclrtGetSocName();

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_kernel_310p");
  cmd.SetCustomHandler(
      [scalar_type, slot_type, socName, stream, page_buffer_ptrs, key_value_ptr,
       slot_mapping_ptr, hidden_dims, kv_size, num_layers, page_buffer_size,
       num_tokens, direction, num_kv_head, head_size, blockSize]() -> int {
        auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
        auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
        auto ascendcPlatform =
            platform_ascendc::PlatformAscendCManager::GetInstance(socName);
        uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
        kvcache_ops::multi_layer_kv_transfer_kernel_310p(
            dtype_num, slot_num, aiv_num, stream, page_buffer_ptrs,
            key_value_ptr, slot_mapping_ptr, hidden_dims, kv_size, num_layers,
            page_buffer_size, num_tokens, direction, num_kv_head, head_size,
            blockSize);
        return 0;
      });
  cmd.Run();
  return;
};

void multi_layer_kv_transfer_unilateral(
    torch::Tensor &key_value, const torch::Tensor &key_ptrs,
    const torch::Tensor &value_ptrs, const torch::Tensor &slot_mapping,
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction) {
  // TODO:
  PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
  throw py::error_already_set();
};

void single_layer_kv_transfer(
    torch::Tensor &lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]
    torch::Tensor &vllm_key_value_cache, // [2, num_blocks, block_size,
                                         // num_heads, head_size]
    torch::Tensor &slot_mapping,         // [num_tokens]
    const bool direction, // false: LMCache to PagedBuffer, true: PagedBuffer to
                          // LMCache
    const bool token_major,   // true: lmc_key_value_cache is [num_tokens, 2,
                              // num_heads*head_size] false: lmc_key_value_cache
                              // is [2, num_tokens, num_heads*head_size]
    const bool vllm_two_major // true: vllm_key_value_cache is [2, num_blocks,
                              // block_size, num_heads, head_size] false:
                              // vllm_key_value_cache is [num_blocks, 2,
                              // block_size, num_heads, head_size]
) {
  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      lmc_key_value_cache, vllm_key_value_cache, slot_mapping, direction,
      token_major, vllm_two_major);

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping));
  compute_single_layer_ub_params(config, vllm_key_value_cache);

  // precompute the strides for lmc_buffer & vllm_buffer
  compute_single_layer_strides(config, lmc_key_value_cache,
                               vllm_key_value_cache, token_major,
                               vllm_two_major);

  at_npu::native::OpCommand cmd;
  cmd.Name("single_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([config]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);

    kvcache_ops::single_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, config.aiv_num, config.stream,
        config.lmc_cache_ptr, config.vllm_cache_ptr, config.slot_mapping_ptr,
        config.vllm_block_stride, config.vllm_value_offset,
        config.vllm_buffer_size, config.lmc_token_stride,
        config.lmc_value_offset, config.lmc_buffer_size,
        config.max_tokens_per_loop, config.num_heads, config.head_dims,
        config.num_tokens, config.block_size, config.direction,
        config.token_major);
    return 0;
  });
  cmd.Run();
  return;
};

void batched_fused_single_layer_kv_transfer(
    std::vector<torch::Tensor>
        &lmc_tensors, // N CPU pinned memory tensors
                      // token_major=true:  [num_tokens, 2, num_heads*head_size]
                      // token_major=false: [2, num_tokens, num_heads*head_size]
    torch::Tensor &staging_cache, // NPU staging buffer
                                  // token_major=true:  [num_tokens, 2,
                                  // num_heads*head_size] token_major=false: [2,
                                  // num_tokens, num_heads*head_size]
    torch::Tensor
        &vllm_key_value_cache,        // vllm_two_major=true:  [2, num_blocks,
                                      // block_size, num_heads, head_size]
                                      // vllm_two_major=false: [num_blocks, 2,
                                      // block_size, num_heads, head_size]
    torch::Tensor &slot_mapping_full, // [num_tokens]
    std::vector<int64_t>
        &chunk_offsets,                // token offset in staging for each chunk
    std::vector<int64_t> &chunk_sizes, // token count for each chunk
    const bool direction, // false: CPU -> staging -> paged (to_gpu) true: paged
                          // -> staging -> CPU (from_gpu)
    const bool
        token_major, // true: [tokens, 2, hidden], false: [2, tokens, hidden]
    const bool vllm_two_major // true: [2, blocks, ...], false: [blocks, 2, ...]
) {
  size_t num_chunks = lmc_tensors.size();
  if (chunk_offsets.size() != num_chunks || chunk_sizes.size() != num_chunks) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "chunk_offsets and chunk_sizes must have the same size as lmc_tensors");
    throw py::error_already_set();
  }

  if (num_chunks == 0) {
    PyErr_SetString(
        PyExc_ValueError,
        "num_chunks must be greater than 0. Check 'starts' and 'ends' inputs.");
    throw py::error_already_set();
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    if (lmc_tensors[i].is_cuda() ||
        lmc_tensors[i].device().type() == c10::DeviceType::PrivateUse1) {
      std::string errStr = "lmc_tensors[" + std::to_string(i) +
                           "] must be on CPU (pinned memory)";
      PyErr_SetString(PyExc_RuntimeError, errStr.c_str());
      throw py::error_already_set();
    }
  }

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      staging_cache, vllm_key_value_cache, slot_mapping_full, direction,
      token_major, vllm_two_major);

  const c10::OptionalDeviceGuard slot_device_guard(
      device_of(slot_mapping_full));
  compute_single_layer_ub_params(config, vllm_key_value_cache);

  compute_single_layer_strides(config, staging_cache, vllm_key_value_cache,
                               token_major, vllm_two_major);

  int64_t element_size = staging_cache.element_size();
  int64_t bytes_per_token = config.lmc_token_stride * element_size;
  int64_t staging_v_plane_offset = config.lmc_value_offset * element_size;

  std::vector<uint8_t *> lmc_ptrs(num_chunks);
  std::vector<int64_t> lmc_copy_sizes(num_chunks);
  std::vector<int64_t> lmc_v_offsets(num_chunks);

  for (size_t i = 0; i < num_chunks; ++i) {
    lmc_ptrs[i] = static_cast<uint8_t *>(lmc_tensors[i].data_ptr());
    lmc_copy_sizes[i] = chunk_sizes[i] * bytes_per_token;

    if (!token_major) {
      lmc_v_offsets[i] = lmc_tensors[i].stride(0) * element_size;
    }
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("batched_fused_single_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([config, num_chunks, lmc_ptrs, lmc_copy_sizes,
                        chunk_offsets, bytes_per_token, staging_v_plane_offset,
                        lmc_v_offsets]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);
    aclError ret;

    // to_gpu (CPU -> staging -> paged)
    if (!config.direction) {
      // Step 1: Multiple H2D memcpy
      for (size_t i = 0; i < num_chunks; ++i) {
        uint8_t *staging_base =
            config.lmc_cache_ptr + chunk_offsets[i] * bytes_per_token;
        int64_t chunk_kv_bytes = lmc_copy_sizes[i];

        if (config.token_major) {
          ret = aclrtMemcpyAsync(staging_base, chunk_kv_bytes, lmc_ptrs[i],
                                 chunk_kv_bytes, ACL_MEMCPY_HOST_TO_DEVICE,
                                 config.stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE, "H2D memcpy failed for chunk ", i,
                      ", ret=", ret);
        } else {
          // K plane
          ret = aclrtMemcpyAsync(staging_base, chunk_kv_bytes, lmc_ptrs[i],
                                 chunk_kv_bytes, ACL_MEMCPY_HOST_TO_DEVICE,
                                 config.stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE, "H2D memcpy (K) failed for chunk ",
                      i, ", ret=", ret);

          // V plane
          ret = aclrtMemcpyAsync(staging_base + staging_v_plane_offset,
                                 chunk_kv_bytes, lmc_ptrs[i] + lmc_v_offsets[i],
                                 chunk_kv_bytes, ACL_MEMCPY_HOST_TO_DEVICE,
                                 config.stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE, "H2D memcpy (V) failed for chunk ",
                      i, ", ret=", ret);
        }
      }

      // Step 2: Scatter (staging -> paged KV cache)
      kvcache_ops::single_layer_kv_transfer_kernel_v2(
          dtype_num, slot_num, config.aiv_num, config.stream,
          config.lmc_cache_ptr, config.vllm_cache_ptr, config.slot_mapping_ptr,
          config.vllm_block_stride, config.vllm_value_offset,
          config.vllm_buffer_size, config.lmc_token_stride,
          config.lmc_value_offset, config.lmc_buffer_size,
          config.max_tokens_per_loop, config.num_heads, config.head_dims,
          config.num_tokens, config.block_size, false, config.token_major);
    }
    // from_gpu (paged -> staging -> CPU)
    else {
      // Step 1: Gather (paged -> staging)
      kvcache_ops::single_layer_kv_transfer_kernel_v2(
          dtype_num, slot_num, config.aiv_num, config.stream,
          config.lmc_cache_ptr, config.vllm_cache_ptr, config.slot_mapping_ptr,
          config.vllm_block_stride, config.vllm_value_offset,
          config.vllm_buffer_size, config.lmc_token_stride,
          config.lmc_value_offset, config.lmc_buffer_size,
          config.max_tokens_per_loop, config.num_heads, config.head_dims,
          config.num_tokens, config.block_size, true, config.token_major);

      // Step 2: Multiple D2H memcpy
      for (size_t i = 0; i < num_chunks; ++i) {
        uint8_t *staging_base =
            config.lmc_cache_ptr + chunk_offsets[i] * bytes_per_token;
        int64_t chunk_kv_bytes = lmc_copy_sizes[i];

        if (config.token_major) {
          ret = aclrtMemcpyAsync(lmc_ptrs[i], chunk_kv_bytes, staging_base,
                                 chunk_kv_bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                 config.stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE, "D2H memcpy failed for chunk ", i,
                      ", ret=", ret);
        } else {
          // K
          ret = aclrtMemcpyAsync(lmc_ptrs[i], chunk_kv_bytes, staging_base,
                                 chunk_kv_bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                 config.stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE, "D2H memcpy (K) failed for chunk ",
                      i, ", ret=", ret);

          // V
          ret = aclrtMemcpyAsync(lmc_ptrs[i] + lmc_v_offsets[i], chunk_kv_bytes,
                                 staging_base + staging_v_plane_offset,
                                 chunk_kv_bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                 config.stream);
          TORCH_CHECK(ret == ACL_ERROR_NONE, "D2H memcpy (V) failed for chunk ",
                      i, ", ret=", ret);
        }
      }
    }

    return 0;
  });
  cmd.Run();
  return;
}

void load_and_reshape_flash(
    torch::Tensor &key_value, // [2, num_layer, num_tokens, num_heads*head_size]
                              // must be one gpu / pinned cpu
    torch::Tensor &key_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor
        &value_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens],
    const int layer_idx) {

  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
  uint8_t *key_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_cache);
  uint8_t *value_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(value_cache);

  uint8_t *slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

  int num_tokens = slot_mapping.size(0);
  int num_layers = key_value.size(1);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);
  int hidden_dims = key_value.size(-1);
  const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at::ScalarType scalar_type = key_value.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();
  const char *socName = aclrtGetSocName();

  at_npu::native::OpCommand cmd;
  cmd.Name("load_and_reshape_flash_kernel");
  cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, key_value_ptr,
                        key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                        hidden_dims, num_blocks, block_size, num_tokens,
                        num_layers, layer_idx]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    kvcache_ops::load_and_reshape_flash_kernel(
        dtype_num, slot_num, aiv_num, stream, key_value_ptr, key_cache_ptr,
        value_cache_ptr, slot_mapping_ptr, hidden_dims, num_blocks, block_size,
        num_tokens, num_layers, layer_idx, true);
    return 0;
  });
  cmd.Run();
  return;
};

void reshape_and_cache_back_flash(
    torch::Tensor &key_value, // [2, num_layer, num_tokens, num_heads*head_size]
                              // must be one gpu / pinned cpu
    torch::Tensor &key_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor
        &value_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens],
    const int layer_idx) {

  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
  uint8_t *key_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_cache);
  uint8_t *value_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(value_cache);

  uint8_t *slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

  int num_tokens = slot_mapping.size(0);
  int num_layers = key_value.size(1);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);
  int hidden_dims = key_value.size(-1);
  const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at::ScalarType scalar_type = key_value.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();

  const char *socName = aclrtGetSocName();

  at_npu::native::OpCommand cmd;
  cmd.Name("reshape_and_cache_back_flash");
  cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, key_value_ptr,
                        key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                        hidden_dims, num_blocks, block_size, num_tokens,
                        num_layers, layer_idx]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    kvcache_ops::load_and_reshape_flash_kernel(
        dtype_num, slot_num, aiv_num, stream, key_value_ptr, key_cache_ptr,
        value_cache_ptr, slot_mapping_ptr, hidden_dims, num_blocks, block_size,
        num_tokens, num_layers, layer_idx, false);
    return 0;
  });
  cmd.Run();
  return;
};
