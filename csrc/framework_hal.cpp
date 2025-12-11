#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <iostream>
#include <string> 
#include <sstream>
#include "torch/torch.h"
#include "torch/extension.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"


namespace py = pybind11;
namespace framework_hal {

int8_t GetDeviceIdx() {
    return  c10_npu::getCurrentNPUStream().device_index();
};
} // namespace framework_hal
