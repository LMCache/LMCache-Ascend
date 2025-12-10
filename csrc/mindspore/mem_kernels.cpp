#include "mem_kernels.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclnn/opdev/platform.h"
#include <pybind11/pybind11.h>
#include <Python.h>

#include "kernels/types.h"
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 
#include <string>

kvcache_ops::AscendType get_dtype_from_np(const py::array& arr) {
    py::object array_dtype_obj = arr.dtype();
    std::string array_dtype_repr =  py::repr(array_dtype_obj).cast<std::string>();

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
    } else if (format_str == "b" || format_str == "i1") { // <--- ADD THIS for signed 8-bit integer
        return kvcache_ops::AscendType::INT8;
    }

    throw std::runtime_error( 
        "Unsupported numpy dtype: " + format_str + ". Only float32, float16, and int8 are supported."); 
}

kvcache_ops::AscendType get_dtype_from_ms(ms::TypeId scalarType)
{
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
 * ptrs[mem_offset(kv, layer, tokenId, hiddenDims)] = key_value[mem_offset(kv, layer, pages, pageSize, slot_id, hiddenDims)]
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

    const char* socName = aclrtGetSocName();
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();

    uint8_t* paged_kv_dev_ptr = static_cast<uint8_t *>(key_value_ptrs.GetDataPtr());
    uint8_t* slot_mapping_ptr = static_cast<uint8_t *>(slot_mappings.GetDataPtr());

    kvcache_ops::multi_layer_kv_transfer_kernel(key_value_type_, slot_type, kvcache_ops::KVCacheFormat::SEPARATE_KV, aiv_num, stream(), paged_kv_dev_ptr, reinterpret_cast<uint8_t*>(key_value_),
                      slot_mapping_ptr, hidden_dims_, kv_size, num_layers, page_buffer_size_,
                      num_tokens, direction_);
  }

  static void Eval(uintptr_t key_value,
                   kvcache_ops::AscendType key_value_type,
                   int hidden_dims,
                   ms::Tensor key_value_ptrs,
                   ms::Tensor slot_mappings,
                   const int page_buffer_size,
                   const bool direction,
                   const bool use_mla)
    {
    auto runner = std::make_shared<MultiLayerKvTransferOp>("MultiLayerKvTransfer");
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

void multi_layer_kv_transfer(py::array& key_value, // [kv, num_layer, num_tokens, hidden]
                             ms::Tensor key_value_ptrs, // [num_layers]
                             ms::Tensor slot_mapping, // [num_tokens]
                             const int page_buffer_size, const bool direction,
                             const bool use_mla, const int kvcache_format_raw) {
// void multi_layer_kv_transfer(
//     py::array& key_value,
//     ms::Tensor key_value_ptrs,
//     ms::Tensor slot_mappings,
//     const int page_buffer_size,
//     const bool direction,
//     const bool use_mla) {
        // reset
        if (direction) {
            memset(static_cast<void*>(key_value.mutable_data()), 0, key_value.nbytes());
        }
        uintptr_t lmc_offset_dptr = reinterpret_cast<uintptr_t>(lmc::get_device_ptr(key_value.mutable_data()));
        kvcache_ops::AscendType key_value_type = get_dtype_from_np(key_value);

        int ndim = key_value.ndim();
        int hidden_dims = static_cast<int>(key_value.shape(ndim - 1));

        ms::pynative::PyboostRunner::Call<0>(
        MultiLayerKvTransferOp::Eval, lmc_offset_dptr, key_value_type, hidden_dims, key_value_ptrs,
        slot_mapping, page_buffer_size, direction, use_mla);
}

// TODO: implement
// void single_layer_kv_transfer(py::array& lmc_key_value_cache, // [num_tokens, 2, num_heads*head_size] or [2, num_tokens, num_heads*head_size],
//                                                               // determined by token_major
//                               BaseTensorPtr& vllm_key_cache, // [num_blocks, block_size, num_heads, head_size]
//                               BaseTensorPtr& vllm_value_cache, // [num_blocks, block_size, num_heads, head_size]
//                               BaseTensorPtr& slot_mapping, // [num_tokens] && vals in range(0, num_blocks * block_size)
//                               const bool direction, // true = PagedBuffer to LMCache 
//                               const bool token_major // conditions lmc_key_value_cache format
// ) {
//     int stream_id = PyBoostUtils::cur_stream_id();
//     mindspore::device::DeviceContext* device_context = mindspore::runtime::OpRunner::GetDeviceContext("Ascend");

//     auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(aclrtGetSocName());
//     uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();

//     int num_tokens = slot_mapping->shape()[0];

//     int ndim = lmc_key_value_cache.ndim();
//     int hidden_dims = static_cast<int>(lmc_key_value_cache.shape(ndim - 1));
//     auto cache_type = get_dtype_from_np(lmc_key_value_cache);
//     kvcache_ops::AscendType slot_type = get_dtype_from_ms(slot_mapping->data_type());

//     uintptr_t lmc_kv_dest_ptr = reinterpret_cast<uintptr_t>(lmc::get_device_ptr(lmc_key_value_cache.mutable_data()));

//     PyBoostUtils::PrepareOpInputs(
//         device_context,
//         stream_id,
//         vllm_key_cache,
//         vllm_value_cache,
//         slot_mapping);

//     PyBoostUtils::DispatchRun(std::make_shared<mindspore::runtime::PyBoostDeviceTask>([=]() {
//         PyBoostUtils::MallocOpInputs(
//             device_context,
//             vllm_key_cache,
//             vllm_value_cache,
//             slot_mapping);
        
//         uint8_t* paged_key_dev_ptr = GetMSDataPtr(vllm_key_cache);
//         uint8_t* paged_value_dev_ptr = GetMSDataPtr(vllm_value_cache);
//         uint8_t* slot_mapping_ptr = GetMSDataPtr(slot_mapping);

//         auto acl_stream = device_context->device_res_manager_->GetStream(stream_id);

//         mindspore::runtime::OpExecutor::DispatchLaunchTask([=]() {
//             kvcache_ops::single_layer_kv_transfer_kernel(
//                 cache_type,
//                 slot_type,
//                 aiv_num,
//                 acl_stream,
//                 reinterpret_cast<uint8_t*>(lmc_kv_dest_ptr),
//                 paged_key_dev_ptr,
//                 paged_value_dev_ptr,
//                 slot_mapping_ptr,
//                 hidden_dims,
//                 num_tokens,
//                 direction, // page2L in interface
//                 token_major,
//                 false // is_MLA
//             );
//         });
//     }));
// }
