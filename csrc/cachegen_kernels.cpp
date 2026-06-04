#include <Python.h>

#include "torch/extension.h"
#include <torch/torch.h>

#include "tiling/platform/platform_ascendc.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void encode_ascend_new(const at::Tensor &cdf, const at::Tensor &input_sym,
                       at::Tensor &output_buffer, at::Tensor &output_lengths) {

  PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
  throw py::error_already_set();
};

void decode_ascend_new(const at::Tensor &cdf, const at::Tensor &bytestreams,
                       const at::Tensor &lengths, at::Tensor &output) {

  PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
  throw py::error_already_set();
};

void decode_ascend_prefsum(const at::Tensor &cdf, const at::Tensor &bytestreams,
                           const at::Tensor &lengths, at::Tensor &output) {

  PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
  throw py::error_already_set();
};

at::Tensor calculate_cdf(const at::Tensor &input, const int n_bins) {

  PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
  throw py::error_already_set();
};
