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

  kvcache_ops::KVCacheFormat kvcache_format =
      static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw);

  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  // we require the kv ptr list to be on the device too
  const c10::OptionalDeviceGuard kv_device_guard(device_of(key_value_ptrs));

  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at::ScalarType scalar_type = key_value.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();
  const char *socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  uint64_t ubSize;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  // we only launched with at most 4 aiv
  uint32_t aiv_num = (uint32_t)std::min(num_layers, 4);

  int32_t numBuffsOnDev = 2;
  // step 1. use per tokens buff size to derive how many tokens can be allocated
  // per loop
  int64_t baseBuffSize = numBuffsOnDev * hidden_dims * key_value.element_size();

  if (ubSize < baseBuffSize) {
    std::string errStr =
        "Per TokenBuffer Size: " + std::to_string(baseBuffSize) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError, errStr + " Please contact us.");
    throw py::error_already_set();
  }

  // step 2. work out how many tokens per loop
  int32_t maxTokensPerLoop =
      (ubSize / baseBuffSize) -
      1; // Subtract 1 to provide a safety margin and avoid over-allocating the
         // UB buffer, ensuring we do not exceed hardware limits due to possible
         // rounding or small additional allocations.
  maxTokensPerLoop = static_cast<int32_t>(
      std::min(maxTokensPerLoop, static_cast<int32_t>(num_tokens)));

  // step 3. double check whether the perloop buffer can accommodate everything
  int64_t totalPerLoopBuffer =
      static_cast<int64_t>(maxTokensPerLoop) * baseBuffSize;
  if (ubSize < totalPerLoopBuffer) {
    std::string errStr =
        "Per Loop Buffer Size: " + std::to_string(totalPerLoopBuffer) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError, errStr + " Please contact us.");
    throw py::error_already_set();
  }

  // using double buffs mean we actually want to allocate half of this per
  // round.
  int64_t singlePerLoopBuffer = totalPerLoopBuffer / numBuffsOnDev;

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([scalar_type, slot_type, kvcache_format, aiv_num, stream,
                        page_buffer_ptrs, key_value_ptr, slot_mapping_ptr,
                        hidden_dims, kv_size, num_layers, page_buffer_size,
                        num_tokens, singlePerLoopBuffer, maxTokensPerLoop,
                        direction]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
    kvcache_ops::multi_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, kvcache_format, aiv_num, stream, page_buffer_ptrs,
        key_value_ptr, slot_mapping_ptr, hidden_dims, kv_size, num_layers,
        page_buffer_size, num_tokens, singlePerLoopBuffer, maxTokensPerLoop,
        direction);
    return 0;
  });
  cmd.Run();
  return;
};

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
  uint8_t *lmc_key_value_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(lmc_key_value_cache);
  uint8_t *vllm_key_value_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(vllm_key_value_cache);
  uint8_t *slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

  int32_t num_tokens = slot_mapping.size(0);
  int32_t num_heads = vllm_key_value_cache.size(-2);
  int32_t head_dims = vllm_key_value_cache.size(-1);
  int32_t block_size = vllm_key_value_cache.size(-3);
  bool is_mla = false;
  int16_t kv_size = 2;
  if (token_major) {
    is_mla = lmc_key_value_cache.size(1) == 1;
  } else {
    is_mla = lmc_key_value_cache.size(0) == 1;
  }

  if (is_mla) {
    PyErr_SetString(
        PyExc_RuntimeError,
        " MLA is not supported yet. Please contact LMCache Ascend.");
    throw py::error_already_set();
  }

  const c10::OptionalDeviceGuard device_guard(device_of(vllm_key_value_cache));
  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at::ScalarType scalar_type = vllm_key_value_cache.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();

  const char *socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  uint32_t aiv_num = (uint32_t)std::min(4, num_tokens);
  uint32_t numBuffsOnDev = 2;
  // each token buffer is kv * heads * headdims * size
  uint64_t baseBuffSize = numBuffsOnDev * kv_size * num_heads * head_dims *
                          vllm_key_value_cache.element_size();
  uint64_t ubSize;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  // first check whether one token k cache actually passes the ub
  if (ubSize < baseBuffSize) {
    std::string errStr =
        "Per Token Cache Buffer Size: " + std::to_string(baseBuffSize) +
        " exceeds UB Size: " + std::to_string(ubSize);
    PyErr_SetString(PyExc_RuntimeError,
                    errStr + " Please contact LMCache Ascend.");
    throw py::error_already_set();
  }

  // we are going to work out how many tokens to copy maximally per innerloop
  int32_t max_tokens_per_loop = ubSize / baseBuffSize;
  max_tokens_per_loop =
      static_cast<int32_t>(std::min(max_tokens_per_loop, num_tokens));

  // precompute the strides for lmc_buffer & vllm_buffer
  int64_t lmc_token_stride;
  int64_t lmc_value_offset;
  if (token_major) {
    // [tokens, 2, heads*headdim]
    lmc_token_stride = lmc_key_value_cache.stride(0);
    lmc_value_offset = lmc_key_value_cache.stride(1);
  } else {
    // [2, tokens, heads*headdim]
    lmc_token_stride = lmc_key_value_cache.stride(1);
    lmc_value_offset = lmc_key_value_cache.stride(0);
  }

  int64_t vllm_block_stride;
  int64_t vllm_value_offset;
  if (vllm_two_major) {
    // 2, num_blocks, block_size, num_heads, head_size
    vllm_block_stride = vllm_key_value_cache.stride(1);
    vllm_value_offset = vllm_key_value_cache.stride(0);
  } else {
    // num_blocks, 2, block_size, num_heads, head_size
    vllm_block_stride = vllm_key_value_cache.stride(0);
    vllm_value_offset = vllm_key_value_cache.stride(1);
  }
  int64_t vllm_buffer_size =
      static_cast<int64_t>(vllm_key_value_cache.nbytes());
  int64_t lmc_buffer_size = static_cast<int64_t>(lmc_key_value_cache.nbytes());

  at_npu::native::OpCommand cmd;
  cmd.Name("single_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler(
      [scalar_type, slot_type, aiv_num, stream, lmc_key_value_cache_ptr,
       vllm_key_value_cache_ptr, slot_mapping_ptr, vllm_block_stride,
       vllm_value_offset, vllm_buffer_size, lmc_token_stride, lmc_value_offset,
       lmc_buffer_size, max_tokens_per_loop, num_heads, head_dims, num_tokens,
       block_size, direction, token_major]() -> int {
        auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
        auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);

        kvcache_ops::single_layer_kv_transfer_kernel_v2(
            dtype_num, slot_num, aiv_num, stream, lmc_key_value_cache_ptr,
            vllm_key_value_cache_ptr, slot_mapping_ptr, vllm_block_stride,
            vllm_value_offset, vllm_buffer_size, lmc_token_stride,
            lmc_value_offset, lmc_buffer_size, max_tokens_per_loop, num_heads,
            head_dims, num_tokens, block_size, direction, token_major);

        return 0;
      });
  cmd.Run();
  return;
};

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
