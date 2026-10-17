#include "mem_kernels.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/npu/Module.h>

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
    const int64_t dsa_hidden_dims) {
  // Direct launches below run on the calling thread (not torch_npu's ACL
  // queue thread), so hold the device guard for the whole function.
  const c10::OptionalDeviceGuard device_guard(paged_memory_device);

  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw, k_hidden_dims,
      v_hidden_dims, dsa_hidden_dims);

  // Calculate UB buffer parameters
  compute_multi_layer_ub_params(config, key_value, paged_memory_device,
                                key_value_ptrs);

  // Direct launch (CUDA cudaLaunchKernel style). config.stream was obtained
  // via getCurrentNPUStream().stream() in compute_multi_layer_ub_params, which
  // first drains torch_npu's ACL task queue, so these stream enqueues keep
  // Python-call order without going through the queue thread.
  auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
  auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);

  kvcache_ops::multi_layer_kv_transfer_kernel_v2(
      dtype_num, slot_num, config.kvcache_format, config.aiv_num, config.stream,
      config.page_buffer_ptrs, key_value_ptr, config.slot_mapping_ptr,
      config.hidden_dims, config.kv_size, config.num_layers,
      config.page_buffer_size, config.num_tokens, config.singlePerLoopBuffer,
      config.maxTokensPerLoop, config.direction, config.k_hidden_dims,
      config.v_hidden_dims, config.dsa_hidden_dims);
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
    const int64_t dsa_hidden_dims) {
  // Direct launches below run on the calling thread (not torch_npu's ACL
  // queue thread), so hold the device guard for the whole function.
  const c10::OptionalDeviceGuard device_guard(paged_memory_device);

  // get host cpu buffer pointer for aclrtMemcpyAsync
  uint8_t *key_value_ptr = static_cast<uint8_t *>(key_value.data_ptr());
  uint8_t *staging_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(staging_cache);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw, k_hidden_dims,
      v_hidden_dims, dsa_hidden_dims);

  compute_multi_layer_ub_params(config, key_value, paged_memory_device,
                                key_value_ptrs);

  // Calculate and verify the CPU buffer size
  // For MLA_KV and DSA_KV, K/V have different hidden_dims
  // Use staging_cache's actual size for verification
  size_t staging_cache_size =
      static_cast<size_t>(staging_cache.numel()) * staging_cache.element_size();

  size_t required_size = 0;
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

  TORCH_CHECK(staging_cache_size >= required_size,
              "staging_cache size insufficient: need ", required_size,
              " bytes, got ", staging_cache_size);

  // Direct launch: see multi_layer_kv_transfer for the ordering contract.
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
  {
    kvcache_ops::multi_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, config.kvcache_format, config.aiv_num,
        config.stream, config.page_buffer_ptrs, staging_cache_ptr,
        config.slot_mapping_ptr, config.hidden_dims, config.kv_size,
        config.num_layers, config.page_buffer_size, config.num_tokens,
        config.singlePerLoopBuffer, config.maxTokensPerLoop, config.direction,
        config.k_hidden_dims, config.v_hidden_dims, config.dsa_hidden_dims);
  }

  // Step 3: D2H memcpy (from_gpu)
  if (!isH2D) {
    ret = aclrtMemcpyAsync(key_value_ptr, required_size, staging_cache_ptr,
                           required_size, ACL_MEMCPY_DEVICE_TO_HOST,
                           config.stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE,
                "D2H memcpy failed: staging_cache -> cpu_buffer, ret=", ret);
  }
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

  // Direct launch: stream was obtained via getCurrentNPUStream().stream()
  // above, which drains torch_npu's ACL task queue first.
  auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
  auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(config.socName);
  uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
  kvcache_ops::multi_layer_kv_transfer_kernel_310p(
      dtype_num, slot_num, config.kvcache_format, aiv_num, stream,
      config.page_buffer_ptrs, key_value_ptr, config.slot_mapping_ptr,
      config.hidden_dims, config.kv_size, config.num_layers,
      config.page_buffer_size, config.num_tokens, config.direction, num_kv_head,
      head_size, blockSize);
  return;
};

void multi_layer_kv_transfer_unilateral(
    torch::Tensor &key_value, const torch::Tensor &key_ptrs,
    const torch::Tensor &value_ptrs, const torch::Tensor &slot_mapping,
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction) {
  // GIL-free error path: this may be called with the GIL released (see the
  // pybind bindings), so raise via TORCH_CHECK instead of the Python C API.
  TORCH_CHECK(false, "multi_layer_kv_transfer_unilateral is not supported yet. "
                     "Please contact LMCache Ascend.");
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

  // Direct launch: ub_params.stream was obtained via
  // getCurrentNPUStream().stream() in prepare_single_layer_kv_config, which
  // drains torch_npu's ACL task queue first, keeping Python-call order.
  if (!is_separate) {
    // Merged KV Kernel
    kvcache_ops::single_layer_kv_transfer_kernel_v2(
        config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
        config.ub_params.aiv_num, config.ub_params.stream, config.ptrs.lmc_ptr,
        config.ptrs.vllm_k_ptr, config.ptrs.slot_mapping_ptr,
        config.strides.vllm_k_stride, config.strides.vllm_val_offset,
        config.strides.vllm_k_bytes, config.strides.lmc_token_stride,
        config.strides.lmc_val_offset, config.strides.lmc_bytes,
        config.ub_params.max_tokens_per_loop, config.dims.num_heads,
        config.dims.head_dims, config.dims.num_tokens, config.dims.block_size,
        config.direction, config.token_major);
  } else {
    // Separate KV Kernel
    kvcache_ops::single_layer_kv_transfer_kernel_v2_separate(
        config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
        config.ub_params.aiv_num, config.ub_params.stream, config.ptrs.lmc_ptr,
        config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
        config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
        config.strides.vllm_v_stride, config.strides.vllm_k_bytes,
        config.strides.vllm_v_bytes, config.strides.lmc_token_stride,
        config.strides.lmc_val_offset, config.strides.lmc_bytes,
        config.ub_params.max_tokens_per_loop, config.dims.num_heads,
        config.dims.head_dims, config.dims.num_tokens, config.dims.block_size,
        config.direction, config.token_major);
  }
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

  // Direct launch: stream was obtained via getCurrentNPUStream().stream()
  // above, which drains torch_npu's ACL task queue first.
  auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
  auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
  kvcache_ops::load_and_reshape_flash_kernel(
      dtype_num, slot_num, aiv_num, stream, key_value_ptr, key_cache_ptr,
      value_cache_ptr, slot_mapping_ptr, hidden_dims, num_blocks, block_size,
      num_tokens, num_layers, layer_idx, true);
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

  // Direct launch: stream was obtained via getCurrentNPUStream().stream()
  // above, which drains torch_npu's ACL task queue first.
  auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
  auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
  kvcache_ops::load_and_reshape_flash_kernel(
      dtype_num, slot_num, aiv_num, stream, key_value_ptr, key_cache_ptr,
      value_cache_ptr, slot_mapping_ptr, hidden_dims, num_blocks, block_size,
      num_tokens, num_layers, layer_idx, false);
  return;
};
