#include "mem_kernels.h"
#include "aclnn/opdev/platform.h"
#include "tiling/platform/platform_ascendc.h"
#include <Python.h>
#include <pybind11/pybind11.h>

#include "kernels/types.h"
#include "managed_mem.h"
#include <Python.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

kvcache_ops::AscendType get_dtype_from_np(const py::array &arr) {
  py::object array_dtype_obj = arr.dtype();
  std::string array_dtype_repr = py::repr(array_dtype_obj).cast<std::string>();

  if (array_dtype_repr.find("bfloat16") != std::string::npos) {
    // HACK: Mindspore np weirdness
    return kvcache_ops::AscendType::BF16;
  }

  // Fallback to format string for other common dtypes
  std::string format_str = arr.request().format;

  if (format_str == "f" || format_str == "f4") { // float32
    return kvcache_ops::AscendType::FP32;
  } else if (format_str == "e" || format_str == "f2") { // float16
    return kvcache_ops::AscendType::FP16;
  } else if (format_str == "b" ||
             format_str == "i1") { // <--- ADD THIS for signed 8-bit integer
    return kvcache_ops::AscendType::INT8;
  }

  throw std::runtime_error("Unsupported numpy dtype: " + format_str +
                           ". Only float32, float16, and int8 are supported.");
}

kvcache_ops::AscendType get_dtype_from_ms(ms::TypeId scalarType) {
  if (scalarType == ms::TypeId::kNumberTypeFloat32) {
    return kvcache_ops::AscendType::FP32;
  } else if (scalarType == ms::TypeId::kNumberTypeBFloat16) {
    return kvcache_ops::AscendType::BF16;
  } else if (scalarType == ms::TypeId::kNumberTypeFloat16) {
    return kvcache_ops::AscendType::FP16;
  } else if (scalarType == ms::TypeId::kNumberTypeInt64) {
    return kvcache_ops::AscendType::INT64;
  } else if (scalarType == ms::TypeId::kNumberTypeInt32) {
    return kvcache_ops::AscendType::INT32;
  } else {
    throw std::runtime_error("ScalarType not supported.");
  }
};

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
class MultiLayerKvTransferOp : public ms::pynative::PyboostRunner {
public:
  using PyboostRunner::PyboostRunner;
  void LaunchKernel() override {
    auto &key_value_ptrs = inputs()[0];
    auto &slot_mappings = inputs()[1];

    int num_tokens = slot_mappings.shape()[0];

    int kv_size = 2;
    if (use_mla_) {
      kv_size = 1;
    }

    int num_layers = key_value_ptrs.shape()[0] / kv_size;

    ms::TypeId slot_mapping_type = slot_mappings.data_type();
    auto slot_type = get_dtype_from_ms(slot_mapping_type);

    const char *socName = aclrtGetSocName();
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();

    uint8_t *paged_kv_dev_ptr =
        static_cast<uint8_t *>(key_value_ptrs.GetDataPtr());
    uint8_t *slot_mapping_ptr =
        static_cast<uint8_t *>(slot_mappings.GetDataPtr());

    kvcache_ops::multi_layer_kv_transfer_kernel(
        key_value_type_, slot_type, kvcache_ops::KVCacheFormat::SEPARATE_KV,
        aiv_num, stream(), paged_kv_dev_ptr,
        reinterpret_cast<uint8_t *>(key_value_), slot_mapping_ptr, hidden_dims_,
        kv_size, num_layers, page_buffer_size_, num_tokens, direction_);
  }

  static void Eval(uintptr_t key_value, kvcache_ops::AscendType key_value_type,
                   int hidden_dims, ms::Tensor key_value_ptrs,
                   ms::Tensor slot_mappings, const int page_buffer_size,
                   const bool direction, const bool use_mla) {
    auto runner =
        std::make_shared<MultiLayerKvTransferOp>("MultiLayerKvTransfer");
    runner->key_value_ = key_value;
    runner->key_value_type_ = key_value_type;
    runner->hidden_dims_ = hidden_dims;
    runner->page_buffer_size_ = page_buffer_size;
    runner->direction_ = direction;
    runner->use_mla_ = use_mla;
    runner->Run({key_value_ptrs, slot_mappings}, {});
  }

  uintptr_t key_value_{0};
  kvcache_ops::AscendType key_value_type_{0};
  int hidden_dims_{0};
  int page_buffer_size_{0};
  bool direction_{0};
  bool use_mla_{0};
};

void multi_layer_kv_transfer(
    py::array &key_value,      // [kv, num_layer, num_tokens, hidden]
    ms::Tensor key_value_ptrs, // [num_layers]
    ms::Tensor slot_mapping,   // [num_tokens]
    const int page_buffer_size, const bool direction, const bool use_mla,
    const int kvcache_format_raw) {
  // reset
  if (direction) {
    memset(static_cast<void *>(key_value.mutable_data()), 0,
           key_value.nbytes());
  }
  uintptr_t lmc_offset_dptr =
      reinterpret_cast<uintptr_t>(get_device_ptr(key_value.mutable_data()));
  kvcache_ops::AscendType key_value_type = get_dtype_from_np(key_value);

  int ndim = key_value.ndim();
  int hidden_dims = static_cast<int>(key_value.shape(ndim - 1));

  ms::pynative::PyboostRunner::Call<0>(
      MultiLayerKvTransferOp::Eval, lmc_offset_dptr, key_value_type,
      hidden_dims, key_value_ptrs, slot_mapping, page_buffer_size, direction,
      use_mla);
}

class MultiLayerKvTransferOp310p : public ms::pynative::PyboostRunner {
public:
  using PyboostRunner::PyboostRunner;
  void LaunchKernel() override {

    auto &lmc_buffer = inputs()[0];
    auto &key_value_ptrs = inputs()[1];
    auto &slot_mappings = inputs()[2];

    int num_tokens = slot_mappings.shape()[0];

    int kv_size = use_mla_ ? 1 : 2;

    kvcache_ops::KVCacheFormat kvcache_format =
        static_cast<kvcache_ops::KVCacheFormat>(kvcache_format_raw_);

    int num_layers = key_value_ptrs.shape()[0] / kv_size;

    ms::TypeId slot_mapping_type = slot_mappings.data_type();
    auto slot_type = get_dtype_from_ms(slot_mapping_type);

    const char *socName = aclrtGetSocName();
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();

    uint8_t *lmc_offset_dptr = static_cast<uint8_t *>(lmc_buffer.GetDataPtr());
    uint8_t *paged_kv_dev_ptr =
        static_cast<uint8_t *>(key_value_ptrs.GetDataPtr());
    uint8_t *slot_mapping_ptr =
        static_cast<uint8_t *>(slot_mappings.GetDataPtr());
    kvcache_ops::multi_layer_kv_transfer_kernel(
        key_value_type_, slot_type, kvcache_format, aiv_num, stream(),
        paged_kv_dev_ptr, lmc_offset_dptr, slot_mapping_ptr, hidden_dims_,
        kv_size, num_layers, page_buffer_size_, num_tokens, direction_);
  }

  static void Eval(ms::Tensor key_value, ms::Tensor key_value_ptrs,
                   ms::Tensor slot_mappings, const int page_buffer_size,
                   const bool direction, const bool use_mla,
                   const int kvcache_format_raw) {
    auto runner =
        std::make_shared<MultiLayerKvTransferOp310p>("MultiLayerKvTransfer");
    runner->key_value_type_ = get_dtype_from_ms(key_value.data_type());
    runner->hidden_dims_ = key_value.shape().back();
    runner->page_buffer_size_ = page_buffer_size;
    runner->direction_ = direction;
    runner->use_mla_ = use_mla;
    runner->kvcache_format_raw_ = kvcache_format_raw;
    runner->Run({key_value, key_value_ptrs, slot_mappings}, {});
  }

  kvcache_ops::AscendType key_value_type_{0};
  int hidden_dims_{0};
  int page_buffer_size_{0};
  bool direction_{0};
  bool use_mla_{0};
  int kvcache_format_raw_{0};
};

void multi_layer_kv_transfer_ms(
    ms::Tensor &key_value,     // [kv, num_layer, num_tokens, hidden]
    ms::Tensor key_value_ptrs, // [num_layers]
    ms::Tensor slot_mapping,   // [num_tokens]
    const int page_buffer_size, const bool direction, const bool use_mla,
    const int kvcache_format_raw) {

  ms::pynative::PyboostRunner::Call<0>(
      MultiLayerKvTransferOp310p::Eval, key_value, key_value_ptrs, slot_mapping,
      page_buffer_size, direction, use_mla, kvcache_format_raw);
}

void multi_layer_kv_transfer_unilateral(ms::Tensor &key_value,
                                        const ms::Tensor &key_ptrs,
                                        const ms::Tensor &value_ptrs,
                                        const ms::Tensor &slot_mapping,
                                        const int page_buffer_size,
                                        const bool direction) {
  PyErr_SetString(PyExc_NotImplementedError,
                  "multi_layer_kv_transfer_unilateral Not Supported");
  throw py::error_already_set();
}

void single_layer_kv_transfer(ms::Tensor &lmc_key_value_cache,
                              ms::Tensor &vllm_key_value_cache,
                              ms::Tensor &slot_mapping, const bool direction,
                              const bool token_major,
                              const bool vllm_two_major) {
  PyErr_SetString(PyExc_NotImplementedError,
                  "single_layer_kv_transfer Not Supported");
  throw py::error_already_set();
}

void load_and_reshape_flash(ms::Tensor &key_value, ms::Tensor &key_cache,
                            ms::Tensor &value_cache, ms::Tensor &slot_mapping,
                            const int layer_idx) {
  PyErr_SetString(PyExc_NotImplementedError,
                  "load_and_reshape_flash Not Supported");
  throw py::error_already_set();
}

void reshape_and_cache_back_flash(ms::Tensor &key_value, ms::Tensor &key_cache,
                                  ms::Tensor &value_cache,
                                  ms::Tensor &slot_mapping,
                                  const int layer_idx) {
  PyErr_SetString(PyExc_NotImplementedError,
                  "reshape_and_cache_back_flash Not Supported");
  throw py::error_already_set();
}
