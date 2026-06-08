#include "mem_kernels.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils.h"
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
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
    const bool direction, const bool use_mla, const int kvcache_format_raw,
    const int64_t k_hidden_dims, const int64_t v_hidden_dims,
    const int64_t dsa_hidden_dims, const int64_t dsa_c8_scale_plane_bytes,
    const int32_t paged_kv_block_size) {
  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw, k_hidden_dims,
      v_hidden_dims, dsa_hidden_dims, dsa_c8_scale_plane_bytes,
      paged_kv_block_size);

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
        config.singlePerLoopBuffer, config.maxTokensPerLoop, config.direction,
        config.k_hidden_dims, config.v_hidden_dims,
        config.dsa_hidden_dims);
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
    const bool use_mla, const int kvcache_format_raw,
    const int64_t k_hidden_dims, const int64_t v_hidden_dims,
    const int64_t dsa_hidden_dims, const int64_t dsa_c8_scale_plane_bytes,
    const int32_t paged_kv_block_size) {
  // get host cpu buffer pointer for aclrtMemcpyAsync
  uint8_t *key_value_ptr = static_cast<uint8_t *>(key_value.data_ptr());
  uint8_t *staging_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(staging_cache);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw, k_hidden_dims,
      v_hidden_dims, dsa_hidden_dims, dsa_c8_scale_plane_bytes,
      paged_kv_block_size);

  compute_multi_layer_ub_params(config, key_value, paged_memory_device,
                                key_value_ptrs);

  // Calculate and verify the CPU buffer size
  // For MLA_KV and DSA_KV, K/V have different hidden_dims
  // Use staging_cache's actual size for verification
  size_t staging_cache_size =
      static_cast<size_t>(staging_cache.numel()) * staging_cache.element_size();

  size_t required_size = 0;
  if (config.is_dsa_c8) {
    // Multi-plane LMCache chunk: padded row layout [layers, chunk_tokens, row_bytes].
    const int64_t lmc_row_bytes =
        static_cast<int64_t>(staging_cache.size(-1)) * staging_cache.element_size();
    required_size = static_cast<size_t>(config.num_layers) *
                    static_cast<size_t>(config.num_tokens_lmc_chunk) *
                    static_cast<size_t>(lmc_row_bytes);
  } else {
    switch (config.kvcache_format) {
    case kvcache_ops::KVCacheFormat::MLA_KV:
      required_size = static_cast<size_t>(config.num_layers) * config.num_tokens *
                      (config.k_hidden_dims + config.v_hidden_dims) *
                      key_value.element_size();
      break;
    case kvcache_ops::KVCacheFormat::DSA_KV:
      required_size =
          static_cast<size_t>(config.num_layers) * config.num_tokens *
          (config.k_hidden_dims + config.v_hidden_dims + config.dsa_hidden_dims) *
          key_value.element_size();
      break;
    default:
      required_size = static_cast<size_t>(config.kv_size) * config.num_layers *
                      config.num_tokens * config.hidden_dims *
                      key_value.element_size();
      break;
    }
  }

  TORCH_CHECK(staging_cache_size >= required_size,
              "staging_cache size insufficient: need ", required_size,
              " bytes, got ", staging_cache_size);

  // DSA-C8 metadata tensors are ephemeral; recordStream keeps their storage
  // alive until the multi-plane kernel completes on config.stream.
  torch::Tensor hidden_dim_bytes;
  torch::Tensor block_sizes;
  torch::Tensor page_buffer_sizes;
  torch::Tensor slot_ptrs;
  torch::Tensor slot_starts;
  torch::Tensor slot_counts;
  if (config.is_dsa_c8) {
    const auto i32_opts = torch::TensorOptions()
                              .dtype(torch::kInt32)
                              .device(slot_mapping.device());
    const auto i64_opts = torch::TensorOptions()
                              .dtype(torch::kInt64)
                              .device(slot_mapping.device());
    hidden_dim_bytes = torch::tensor(
        {static_cast<int32_t>(config.k_hidden_dims),
         static_cast<int32_t>(config.v_hidden_dims),
         static_cast<int32_t>(config.dsa_hidden_dims),
         static_cast<int32_t>(config.dsa_c8_scale_plane_bytes)},
        i32_opts);
    block_sizes = torch::full(
        {4}, static_cast<int32_t>(config.paged_kv_block_size), i32_opts);
    page_buffer_sizes = torch::full(
        {4}, static_cast<int32_t>(config.page_buffer_size), i32_opts);
    const int32_t ntok = static_cast<int32_t>(config.num_tokens);
    const int64_t slot_data_ptr =
        reinterpret_cast<int64_t>(slot_mapping.data_ptr());
    slot_ptrs = torch::full({4}, slot_data_ptr, i64_opts);
    slot_starts = torch::zeros({4}, i32_opts);
    slot_counts = torch::full({4}, ntok, i32_opts);
  }

  const c10_npu::NPUStream npu_stream = c10_npu::getCurrentNPUStream();
  at_npu::native::OpCommand cmd;
  cmd.Name("fused_multi_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([config, staging_cache_ptr, key_value_ptr, required_size,
                        hidden_dim_bytes, block_sizes, page_buffer_sizes,
                        slot_ptrs, slot_starts, slot_counts, staging_cache,
                        npu_stream]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);

    aclError ret;
    // direction: false = to_gpu (H2D), true = from_gpu (D2H)
    bool isH2D = !config.direction;

    // Step 1: H2D memcpy (to_gpu) currently not used
    if (isH2D) {
      ret = aclrtMemcpyAsync(staging_cache_ptr, required_size, key_value_ptr,
                             required_size, ACL_MEMCPY_HOST_TO_DEVICE,
                             config.stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE,
                  "H2D memcpy failed: cpu_buffer -> staging_cache, ret=", ret);
    }

    // Step 2: Kernel (Gather or Scatter)
    if (config.is_dsa_c8) {
      kvcache_ops::multi_layer_kv_transfer_multi_plane_kernel_v2(
          config.aiv_num, config.stream, config.page_buffer_ptrs, staging_cache_ptr,
          slot_ptrs.data_ptr<int64_t>(), slot_starts.data_ptr<int32_t>(),
          slot_counts.data_ptr<int32_t>(), hidden_dim_bytes.data_ptr<int32_t>(),
          block_sizes.data_ptr<int32_t>(), page_buffer_sizes.data_ptr<int32_t>(),
          nullptr, 4, config.num_layers,
          static_cast<int64_t>(staging_cache.size(-1)) * staging_cache.element_size(),
          config.num_tokens_lmc_chunk, config.singlePerLoopBuffer,
          config.maxTokensPerLoop, config.direction);
      c10_npu::NPUCachingAllocator::recordStream(
          hidden_dim_bytes.storage().data_ptr(), npu_stream);
      c10_npu::NPUCachingAllocator::recordStream(
          block_sizes.storage().data_ptr(), npu_stream);
      c10_npu::NPUCachingAllocator::recordStream(
          page_buffer_sizes.storage().data_ptr(), npu_stream);
      c10_npu::NPUCachingAllocator::recordStream(
          slot_ptrs.storage().data_ptr(), npu_stream);
      c10_npu::NPUCachingAllocator::recordStream(
          slot_starts.storage().data_ptr(), npu_stream);
      c10_npu::NPUCachingAllocator::recordStream(
          slot_counts.storage().data_ptr(), npu_stream);
    } else {
      kvcache_ops::multi_layer_kv_transfer_kernel_v2(
          dtype_num, slot_num, config.kvcache_format, config.aiv_num,
          config.stream, config.page_buffer_ptrs, staging_cache_ptr,
          config.slot_mapping_ptr, config.hidden_dims, config.kv_size,
          config.num_layers, config.page_buffer_size, config.num_tokens,
          config.singlePerLoopBuffer, config.maxTokensPerLoop,
          config.direction, config.k_hidden_dims, config.v_hidden_dims,
          config.dsa_hidden_dims);
    }

    // Step 3: D2H memcpy (from_gpu)
    if (!isH2D) {
      ret = aclrtMemcpyAsync(key_value_ptr, required_size, staging_cache_ptr,
                             required_size, ACL_MEMCPY_DEVICE_TO_HOST,
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
    const int head_size, const int blockSize, const int kvcache_format_raw) {
  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw);

  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  // we require the kv ptr list to be on the device too
  const c10::OptionalDeviceGuard kv_device_guard(device_of(key_value_ptrs));

  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_kernel_310p");
  cmd.SetCustomHandler([config, stream, key_value_ptr, num_kv_head, head_size,
                        blockSize]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(config.socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    kvcache_ops::multi_layer_kv_transfer_kernel_310p(
        dtype_num, slot_num, config.kvcache_format, aiv_num, stream,
        config.page_buffer_ptrs, key_value_ptr, config.slot_mapping_ptr,
        config.hidden_dims, config.kv_size, config.num_layers,
        config.page_buffer_size, config.num_tokens, config.direction,
        num_kv_head, head_size, blockSize);
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
    torch::Tensor
        &lmc_key_value_cache, // [num_tokens, 2, num_heads*head_size]
                              // or [2, num_tokens, num_heads*head_size]
    std::vector<torch::Tensor> &vllm_kv_caches,
    // SEPARATE_KV: list[k_tensor, v_tensor]
    // k_tensor/v_tensor = [num_blocks, block_size, num_heads, head_size]
    // MERGED_KV:
    // vllm_two_major=true:  [2, num_blocks, block_size, num_heads, head_size]
    // vllm_two_major=false: [num_blocks, 2, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens]
    const bool direction, // false: LMCache -> Paged, true: Paged -> LMCache
    const int kvcache_format_raw, // 1: MERGED_KV, 2: SEPARATE_KV
    const bool
        token_major, // true: [tokens, 2, hidden], false: [2, tokens, hidden]
    const bool vllm_two_major // true: [2, blocks, ...], false: [blocks, 2, ...]
                              // (only for MERGED_KV)
) {
  bool is_separate = validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping));

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      lmc_key_value_cache, vllm_kv_caches, slot_mapping, direction, token_major,
      vllm_two_major, is_separate);

  at_npu::native::OpCommand cmd;
  cmd.Name(is_separate ? "single_layer_kv_transfer_kernel_v2_separate"
                       : "single_layer_kv_transfer_kernel_v2");

  cmd.SetCustomHandler([config, is_separate]() -> int {
    if (!is_separate) {
      // Merged KV Kernel
      kvcache_ops::single_layer_kv_transfer_kernel_v2(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_val_offset, config.strides.vllm_k_bytes,
          config.strides.lmc_token_stride, config.strides.lmc_val_offset,
          config.strides.lmc_bytes, config.ub_params.max_tokens_per_loop,
          config.dims.num_heads, config.dims.head_dims, config.dims.num_tokens,
          config.dims.block_size, config.direction, config.token_major);
    } else {
      // Separate KV Kernel
      kvcache_ops::single_layer_kv_transfer_kernel_v2_separate(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_v_stride, config.strides.vllm_k_bytes,
          config.strides.vllm_v_bytes, config.strides.lmc_token_stride,
          config.strides.lmc_val_offset, config.strides.lmc_bytes,
          config.ub_params.max_tokens_per_loop, config.dims.num_heads,
          config.dims.head_dims, config.dims.num_tokens, config.dims.block_size,
          config.direction, config.token_major);
    }
    return 0;
  });
  cmd.Run();
}

void batched_fused_single_layer_kv_transfer(
    std::vector<torch::Tensor>
        &lmc_tensors, // N CPU pinned memory tensors
                      // token_major=true:  [num_tokens, 2, num_heads*head_size]
                      // token_major=false: [2, num_tokens, num_heads*head_size]
    torch::Tensor &staging_cache, // NPU staging buffer
                                  // token_major=true:  [num_tokens, 2,
                                  // num_heads*head_size] token_major=false: [2,
                                  // num_tokens, num_heads*head_size]
    std::vector<torch::Tensor>    // separate format： list[k_tensor, v_tensor]
        &vllm_kv_caches, // k_tensor/v_tensor = [num_blocks，block_size,
                         // num_heads, head_size]
                         //  Mergeed format：
                         //  vllm_two_major=true:  [2, num_blocks, block_size,
                         //  num_heads, head_size] vllm_two_major=false:
                         //  [num_blocks, 2, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping_full, // [num_tokens]
    std::vector<int64_t>
        &chunk_offsets,                // token offset in staging for each chunk
    std::vector<int64_t> &chunk_sizes, // token count for each chunk
    const bool direction, // false: CPU -> staging -> paged (to_gpu) true: paged
                          // -> staging -> CPU (from_gpu)
    const int kvcache_format_raw,
    const bool
        token_major, // true: [tokens, 2, hidden], false: [2, tokens, hidden]
    const bool vllm_two_major // true: [2, blocks, ...], false: [blocks, 2, ...]
) {

  bool is_separate = validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);

  const c10::OptionalDeviceGuard slot_device_guard(
      device_of(slot_mapping_full));

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      staging_cache, vllm_kv_caches, slot_mapping_full, direction, token_major,
      vllm_two_major, is_separate);

  int64_t element_size = staging_cache.element_size();

  if (!is_separate) {
    auto launcher = [config](bool is_gather) {
      kvcache_ops::single_layer_kv_transfer_kernel_v2(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_val_offset, config.strides.vllm_k_bytes,
          config.strides.lmc_token_stride, config.strides.lmc_val_offset,
          config.strides.lmc_bytes, config.ub_params.max_tokens_per_loop,
          config.dims.num_heads, config.dims.head_dims, config.dims.num_tokens,
          config.dims.block_size, is_gather, config.token_major);
    };
    run_batched_fused_transfer(config, lmc_tensors, chunk_offsets, chunk_sizes,
                               element_size, launcher);

  } else {
    auto launcher = [config](bool is_gather) {
      kvcache_ops::single_layer_kv_transfer_kernel_v2_separate(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_v_stride, config.strides.vllm_k_bytes,
          config.strides.vllm_v_bytes, config.strides.lmc_token_stride,
          config.strides.lmc_val_offset, config.strides.lmc_bytes,
          config.ub_params.max_tokens_per_loop, config.dims.num_heads,
          config.dims.head_dims, config.dims.num_tokens, config.dims.block_size,
          is_gather, config.token_major);
    };
    run_batched_fused_transfer(config, lmc_tensors, chunk_offsets, chunk_sizes,
                               element_size, launcher);
  }
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

// Multi-plane KV transfer: per-plane slot pointers must reference dense mappings
// (no -1). Starts/counts index the chunk slice within each plane's mapping.
void multi_layer_kv_transfer_multi_plane(
    torch::Tensor &key_value, const torch::Tensor &key_value_ptrs,
    const torch::Tensor &slot_mapping_ptrs,
    const torch::Tensor &slot_mapping_starts,
    const torch::Tensor &slot_mapping_counts,
    const torch::Tensor &page_buffer_sizes, const torch::Tensor &block_sizes,
    const torch::Tensor &hidden_dim_bytes, const int64_t max_hidden_dim_bytes,
    const torch::Device &paged_memory_device, const bool direction,
    const int num_planes, const torch::Tensor &lmc_row_offsets) {
  TORCH_CHECK(num_planes > 0, "num_planes must be positive");
  TORCH_CHECK(num_planes <= 32, "num_planes cannot exceed 32 (kMaxPlanes)");
  TORCH_CHECK(slot_mapping_ptrs.dim() == 1 &&
                  slot_mapping_ptrs.size(0) == num_planes,
              "slot_mapping_ptrs length mismatch");
  TORCH_CHECK(slot_mapping_starts.dim() == 1 &&
                  slot_mapping_starts.size(0) == num_planes,
              "slot_mapping_starts length mismatch");
  TORCH_CHECK(slot_mapping_counts.dim() == 1 &&
                  slot_mapping_counts.size(0) == num_planes,
              "slot_mapping_counts length mismatch");
  TORCH_CHECK(page_buffer_sizes.dim() == 1 && page_buffer_sizes.size(0) == num_planes,
              "page_buffer_sizes length mismatch");
  TORCH_CHECK(block_sizes.dim() == 1 && block_sizes.size(0) == num_planes,
              "block_sizes length mismatch");
  TORCH_CHECK(hidden_dim_bytes.dim() == 1 &&
                  hidden_dim_bytes.size(0) == num_planes,
              "hidden_dim_bytes length mismatch");
  TORCH_CHECK(slot_mapping_ptrs.scalar_type() == torch::kInt64,
              "slot_mapping_ptrs must be int64");
  TORCH_CHECK(slot_mapping_starts.scalar_type() == torch::kInt32,
              "slot_mapping_starts must be int32");
  TORCH_CHECK(slot_mapping_counts.scalar_type() == torch::kInt32,
              "slot_mapping_counts must be int32");
  TORCH_CHECK(lmc_row_offsets.dim() == 1 && lmc_row_offsets.size(0) == num_planes,
              "lmc_row_offsets length mismatch");
  TORCH_CHECK(lmc_row_offsets.scalar_type() == torch::kInt32,
              "lmc_row_offsets must be int32");

  const int64_t lmc_chunk_last_dim_bytes =
      key_value.size(-1) * key_value.element_size();
  TORCH_CHECK(lmc_chunk_last_dim_bytes > 0, "lmc_chunk last dim must be positive");
  TORCH_CHECK(key_value_ptrs.size(0) % num_planes == 0,
              "key_value_ptrs length must be num_layers * num_planes");
  const int32_t num_layers =
      static_cast<int32_t>(key_value_ptrs.size(0) / num_planes);

  TORCH_CHECK(max_hidden_dim_bytes > 0, "max_hidden_dim_bytes must be positive");

  const int32_t num_tokens_lmc_chunk =
      key_value.dim() >= 3 ? static_cast<int32_t>(key_value.size(2)) : 1;

  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
  uint8_t *paged_ptrs =
      get_kernel_ptr<uint8_t, const torch::Tensor>(key_value_ptrs);
  int64_t *slot_ptrs =
      get_kernel_ptr<int64_t, const torch::Tensor>(slot_mapping_ptrs);
  int32_t *slot_starts_ptr =
      get_kernel_ptr<int32_t, const torch::Tensor>(slot_mapping_starts);
  int32_t *slot_counts_ptr =
      get_kernel_ptr<int32_t, const torch::Tensor>(slot_mapping_counts);
  int32_t *hd_ptr =
      get_kernel_ptr<int32_t, const torch::Tensor>(hidden_dim_bytes);
  int32_t *bs_ptr = get_kernel_ptr<int32_t, const torch::Tensor>(block_sizes);
  int32_t *pbs_ptr =
      get_kernel_ptr<int32_t, const torch::Tensor>(page_buffer_sizes);
  int32_t *lmc_row_off_ptr =
      get_kernel_ptr<int32_t, const torch::Tensor>(lmc_row_offsets);

  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  void *stream = c10_npu::getCurrentNPUStream().stream();

  const char *socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  uint64_t ubSize = 0;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  const uint32_t aiv_num =
      static_cast<uint32_t>(std::min(num_layers, 4));
  constexpr int32_t numBuffsOnDev = 2;
  const int64_t baseBuffSize = numBuffsOnDev * max_hidden_dim_bytes;
  TORCH_CHECK(ubSize >= static_cast<uint64_t>(baseBuffSize),
              "UB too small for multi-plane KV transfer");
  int32_t maxTokensPerLoop =
      static_cast<int32_t>(ubSize / baseBuffSize) - 1;
  maxTokensPerLoop = std::min(maxTokensPerLoop, num_tokens_lmc_chunk);
  const int64_t totalPerLoopBuffer =
      static_cast<int64_t>(maxTokensPerLoop) * baseBuffSize;
  const int64_t singlePerLoopBuffer = totalPerLoopBuffer / numBuffsOnDev;

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_multi_plane_kernel_v2");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::multi_layer_kv_transfer_multi_plane_kernel_v2(
        aiv_num, stream, paged_ptrs, key_value_ptr, slot_ptrs, slot_starts_ptr,
        slot_counts_ptr, hd_ptr, bs_ptr, pbs_ptr, lmc_row_off_ptr, num_planes,
        num_layers, lmc_chunk_last_dim_bytes, num_tokens_lmc_chunk,
        singlePerLoopBuffer, maxTokensPerLoop, direction);
    return 0;
  });
  cmd.Run();
}
