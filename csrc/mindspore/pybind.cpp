// SPDX-License-Identifier: Apache-2.0

#include "dcmi_management.h"
#include "managed_mem.h"
#include "mem_alloc.h"
#include "mem_kernels.h"
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_ms", &multi_layer_kv_transfer_ms);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_npu_pci_bus_id);
}