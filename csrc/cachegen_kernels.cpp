#include "cachegen_kernels.h"
#include <Python.h>

#include "torch/extension.h"
#include <torch/torch.h>

#include "tiling/platform/platform_ascendc.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

static constexpr uint32_t AIV_MAX = 20;

void encode_ascend_new(const at::Tensor &cdf, const at::Tensor &input_sym,
                       at::Tensor &output_buffer, at::Tensor &output_lengths) {

  const auto input_shape = input_sym.sizes();
  const int nlayers = input_shape[0];
  const int ntokens = input_shape[1];
  const int nchannels = input_shape[2];
  const int nbins = cdf.sizes()[2] - 1;

  const auto cdf_sizes = cdf.sizes();
  const auto output_buffer_shape = output_buffer.sizes();
  const int chunk_size = output_buffer_shape[2];

  TORCH_CHECK(cdf.device().is_privateuseone());
  TORCH_CHECK(input_sym.device().is_privateuseone());
  TORCH_CHECK(output_buffer.device().is_privateuseone());
  TORCH_CHECK(output_lengths.device().is_privateuseone());
  TORCH_CHECK(nchannels <= 1 << 15,
              "Number of channels exceeds that supported be encode, contact "
              "LMCache Ascend about changing this limitation");
  const c10::OptionalDeviceGuard device_guard(device_of(input_sym));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  auto cdf_data_ptr = static_cast<uint8_t *>(cdf.data_ptr());
  auto input_data_ptr = static_cast<uint8_t *>(input_sym.data_ptr());
  auto output_data_ptr = static_cast<uint8_t *>(output_buffer.data_ptr());
  auto output_lengths_data_ptr =
      static_cast<uint8_t *>(output_lengths.data_ptr());

  const char *socName = aclrtGetSocName();
  auto _custom_handler = [=]() -> int {
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t n_aiv = AIV_MAX;
    kvcache_ops::cachegen::encode_v2(
        cdf_data_ptr, input_data_ptr, output_data_ptr, output_lengths_data_ptr,
        stream, n_aiv, nbins, ntokens, nlayers, nchannels, chunk_size);
    return 0;
  };

  at_npu::native::OpCommand cmd;
  cmd.Name("encode").SetCustomHandler(_custom_handler).Run();

  return;
};

void decode_ascend_new(const at::Tensor &cdf, const at::Tensor &bytestreams,
                       const at::Tensor &lengths, at::Tensor &output) {
  // TODO:
  PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
  throw py::error_already_set();
};

void decode_ascend_prefsum(const at::Tensor &cdf, const at::Tensor &bytestreams,
                           const at::Tensor &lengths, at::Tensor &output) {

  const auto cdf_shape = cdf.sizes();
  const int nlayers = cdf_shape[0];
  const int nchannels = cdf_shape[1];
  const int nbins = cdf_shape[2] - 1; // To match calculate cdf
  const int ntokens = output.sizes()[1];

  const auto bytestreams_shape = bytestreams.sizes();
  const auto cdf_sizes = cdf.sizes();
  const auto output_buffer_shape = output.sizes();

  TORCH_CHECK(cdf.device().is_privateuseone(),
              "CDFs tensor should be on the NPU");
  TORCH_CHECK(bytestreams.device().is_privateuseone(),
              "Bytestreams tensor should be on the NPU");
  TORCH_CHECK(lengths.device().is_privateuseone(),
              "Lengths tensor should be on the NPU");
  TORCH_CHECK(output.device().is_privateuseone(),
              "Output tensor should be on the NPU");
  TORCH_CHECK(
      nchannels % 32 == 0,
      "Decode implementation relies on the number of channels being a multiple "
      "of 32, contact LMCache Ascend about changing this limitation");
  TORCH_CHECK(nbins < 64, "Max bins constraint violated");

  const c10::OptionalDeviceGuard device_guard(device_of(bytestreams));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  auto cdf_data_ptr = static_cast<uint8_t *>(cdf.data_ptr());
  auto bytestreams_data_ptr = static_cast<uint8_t *>(bytestreams.data_ptr());
  auto lengths_data_ptr = static_cast<uint8_t *>(lengths.data_ptr());
  auto output_data_ptr = static_cast<uint8_t *>(output.data_ptr());

  const char *socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  uint32_t n_aiv = AIV_MAX;

  auto _custom_handler = [=]() -> int {
    kvcache_ops::cachegen::decode_v2(cdf_data_ptr, bytestreams_data_ptr,
                                     lengths_data_ptr, output_data_ptr, stream,
                                     n_aiv, nbins, ntokens, nlayers, nchannels);
    return 0;
  };

  at_npu::native::OpCommand cmd;
  cmd.Name("decode").SetCustomHandler(_custom_handler).Run();

  return;
};

// Calculate the CDF of the input which is a NPU resident tensor of values
// quantized into n_bins
//
// input - uint8 tensor of shape [nlayers, ntokens, nchannels] with values in
// the range [0, n_bins) n_bins - the number of bins used
at::Tensor calculate_cdf(const at::Tensor &input, const int n_bins) {

  const auto input_shape = input.sizes();
  const int nlayers = input_shape[0];
  const int ntokens = input_shape[1];
  const int nchannels = input_shape[2];

  // The input tensor should be on the NPU - torch_npu registers the npu device
  // as PrivateUse1 with torch so check for that.
  TORCH_CHECK(input.device().is_privateuseone());

  const c10::OptionalDeviceGuard device_guard(device_of(input));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  auto output = torch::zeros({nlayers, nchannels, n_bins + 1},
                             input.options().dtype(torch::kI16) // torch::kI16
  );

  auto input_data_ptr = static_cast<uint8_t *>(input.data_ptr());
  auto output_data_ptr = static_cast<uint8_t *>(output.data_ptr());

  const char *socName = aclrtGetSocName();
  auto _custom_handler = [=]() -> int {
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t n_aiv = AIV_MAX;
    kvcache_ops::cachegen::calculate_cdf(input_data_ptr, output_data_ptr,
                                         stream, n_aiv, n_bins, ntokens,
                                         nlayers, nchannels);
    return 0;
  };

  at_npu::native::OpCommand cmd;
  cmd.Name("calculate_cdf").SetCustomHandler(_custom_handler).Run();

  return output;
};
