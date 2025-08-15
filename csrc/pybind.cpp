// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include "mem_kernels.h"
#include "managed_mem.h"
#include "cachegen_kernels.h"
#include "pos_kernels.h"
#include <torch/torch.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  m.def("host_register", &register_memory);
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("encode_fast_new", &encode_cuda_new);
  m.def("decode_fast_new", &decode_cuda_new);
  m.def("decode_fast_prefsum", &decode_cuda_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
}