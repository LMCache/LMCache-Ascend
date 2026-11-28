// SPDX-License-Identifier: Apache-2.0

#include "managed_mem.h"
#include <cstdint>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Host/device memory registration shared by the hccl and hixl transfer
// channels. The MP and in-process kernels moved to LMCache (lmcache.npu_ops)
// and to the dsv4_support_045 branch respectively.
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
}
