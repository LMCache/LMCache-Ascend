/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "kernels/types.h"
#include "managed_mem.h"
#include <c10/core/ScalarType.h>
#include <string>
#include <torch/torch.h>

namespace vllm_ascend {
kvcache_ops::AscendType get_dtype_from_torch(at::ScalarType scalarType);
} // namespace vllm_ascend

template <typename T, typename TENSOR_TYPE>
T *get_kernel_ptr(TENSOR_TYPE &tensor) {
  torch::Device device = tensor.device();
  // NPU should be using PrivateUse1
  if (device.is_privateuseone() || device.is_cuda()) {
    return static_cast<T *>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    // find device ptr based on the host pinned ptr
    // because acl does not currently support HostGetDevicePointer API
    void *devPtr = get_device_ptr(tensor.data_ptr());
    TORCH_CHECK(
        devPtr != nullptr,
        "Unable to retrieve device ptr, is this a host registered pointer ?");
    return reinterpret_cast<T *>(devPtr);
  } else {
    TORCH_CHECK(
        false,
        "Invalid device. Device must be ascend (PrivateUseOne) or pinned cpu.");
  }
}
