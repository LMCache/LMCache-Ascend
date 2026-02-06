// SPDX-License-Identifier: Apache-2.0

#include "cachegen_kernels.h"
#include "dcmi_management.h"
#include "managed_mem.h"
#include "mem_alloc.h"
#include "mem_kernels.h"
#include "pos_kernels.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

namespace py = pybind11;

std::vector<torch::Tensor> normalize_kv_caches(const py::object &input) {
  if (THPVariable_Check(input.ptr())) {
    return {input.cast<torch::Tensor>()};
  } else if (py::isinstance<py::tuple>(input)) {
    return input.cast<std::vector<torch::Tensor>>();
  } else {
    throw std::runtime_error(
        "vllm_kv_caches must be a Tensor or a tuple of Tensors");
  }
}

void single_layer_kv_transfer_wrapper(torch::Tensor &lmc_key_value_cache,
                                      const py::object &vllm_kv_caches_obj,
                                      torch::Tensor &slot_mapping,
                                      bool direction, int kvcache_format_raw,
                                      bool token_major, bool vllm_two_major) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  single_layer_kv_transfer(lmc_key_value_cache, vllm_kv_caches, slot_mapping,
                           direction, kvcache_format_raw, token_major,
                           vllm_two_major);
}

void batched_fused_single_layer_kv_transfer_wrapper(
    std::vector<torch::Tensor> &lmc_tensors, torch::Tensor &staging_cache,
    const py::object &vllm_kv_caches_obj, torch::Tensor &slot_mapping_full,
    std::vector<int64_t> &chunk_offsets, std::vector<int64_t> &chunk_sizes,
    bool direction, int kvcache_format_raw, bool token_major,
    bool vllm_two_major) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  batched_fused_single_layer_kv_transfer(
      lmc_tensors, staging_cache, vllm_kv_caches, slot_mapping_full,
      chunk_offsets, chunk_sizes, direction, kvcache_format_raw, token_major,
      vllm_two_major);
}

PYBIND11_MODULE(c_ops, m) {
  m.def("get_device_ptr", [](uintptr_t ptr_addr) {
    return reinterpret_cast<uintptr_t>(
        get_device_ptr(reinterpret_cast<void *>(ptr_addr)));
  });
  m.def("register_mapping",
        [](uintptr_t host_ptr, uintptr_t dev_ptr, size_t size) {
          return reinterpret_cast<uintptr_t>(
              register_mapping(reinterpret_cast<void *>(host_ptr),
                               reinterpret_cast<void *>(dev_ptr), size));
        });
  m.def("unregister_ptr", [](uintptr_t ptr_addr) {
    return unregister_ptr(reinterpret_cast<void *>(ptr_addr));
  });
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("fused_multi_layer_kv_transfer", &fused_multi_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_310p", &multi_layer_kv_transfer_310p);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer_wrapper);
  m.def("batched_fused_single_layer_kv_transfer",
        &batched_fused_single_layer_kv_transfer_wrapper);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("encode_fast_new", &encode_ascend_new);
  m.def("decode_fast_new", &decode_ascend_new);
  m.def("decode_fast_prefsum", &decode_ascend_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_npu_pci_bus_id);
}
