#include "pac_kernels.h"
#include <Python.h>

#include "torch/extension.h"
#include <torch/torch.h>

#include "tiling/platform/platform_ascendc.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

static constexpr uint32_t AIV_MAX = 20;

void pac_prepare_enc_metadata(const at::Tensor &input_sym,
                              const at::Tensor &meta_data) {

  const auto input_shape = input_sym.sizes();
  const int nlayers = input_shape[0];
  const int ntokens = input_shape[1];
  const int nchannels = input_shape[2];
  const int nbins = meta_data.sizes()[2];

  TORCH_CHECK(meta_data.device().is_privateuseone());
  TORCH_CHECK(input_sym.device().is_privateuseone());
  TORCH_CHECK(nchannels <= 4096,
              "Number of channels exceeds that supported be encode, contact "
              "LMCache Ascend about changing this limitation");
  TORCH_CHECK(nbins <= 32,
              "Number of bins exceeds that supported be encode, contact."
              "LMCache Ascend about changing this limitation");
  TORCH_CHECK(
      nchannels % 32 == 0,
      "Encode implementation relies on the number of channels being a multiple "
      "of 32, contact LMCache Ascend about changing this limitation");

  const c10::OptionalDeviceGuard device_guard(device_of(input_sym));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  auto meta_data_data_ptr = static_cast<uint8_t *>(meta_data.data_ptr());
  auto input_data_ptr = static_cast<uint8_t *>(input_sym.data_ptr());

  const char *socName = aclrtGetSocName();
  auto _custom_handler = [=]() -> int {
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t n_aiv = std::min(AIV_MAX, ascendcPlatform->GetCoreNumAiv());
    kvcache_ops::pac_coder::pac_prep_enc_metadata(
        input_data_ptr, meta_data_data_ptr, stream, n_aiv, nbins, ntokens,
        nlayers, nchannels);
    return 0;
  };

  at_npu::native::OpCommand cmd;
  cmd.Name("prep_encode_metadata").SetCustomHandler(_custom_handler).Run();
}

void pac_encode(const at::Tensor &input_sym, const at::Tensor &meta_data,
                at::Tensor &output_buffer, at::Tensor &output_lengths) {

  const auto input_shape = input_sym.sizes();
  const int nlayers = input_shape[0];
  const int ntokens = input_shape[1];
  const int nchannels = input_shape[2];
  const int nbins = meta_data.sizes()[2];

  const auto output_buffer_shape = output_buffer.sizes();
  const int chunk_size = output_buffer_shape[2];

  TORCH_CHECK(meta_data.device().is_privateuseone());
  TORCH_CHECK(input_sym.device().is_privateuseone());
  TORCH_CHECK(output_buffer.device().is_privateuseone());
  TORCH_CHECK(output_lengths.device().is_privateuseone());
  TORCH_CHECK(nchannels <= 4096,
              "Number of channels exceeds that supported be encode, contact "
              "LMCache Ascend about changing this limitation");
  TORCH_CHECK(ntokens <= 256,
              "Number of tokens exceeds that supported be encode, handle by "
              "chunking the input."
              "Contact LMCache Ascend about changing this limitation");
  TORCH_CHECK(nbins <= 32,
              "Number of bins exceeds that supported be encode, contact."
              "LMCache Ascend about changing this limitation");
  TORCH_CHECK(
      nchannels % 32 == 0,
      "Encode implementation relies on the number of channels being a multiple "
      "of 32, contact LMCache Ascend about changing this limitation");

  const c10::OptionalDeviceGuard device_guard(device_of(input_sym));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  auto meta_data_data_ptr = static_cast<uint8_t *>(meta_data.data_ptr());
  auto input_data_ptr = static_cast<uint8_t *>(input_sym.data_ptr());
  auto output_data_ptr = static_cast<uint8_t *>(output_buffer.data_ptr());
  auto output_lengths_data_ptr =
      static_cast<uint8_t *>(output_lengths.data_ptr());

  auto workGM = torch::zeros({40 * 32}, input_sym.options().dtype(torch::kI32));
  auto workGM_ptr = static_cast<uint8_t *>(workGM.data_ptr());

  const char *socName = aclrtGetSocName();
  auto _custom_handler = [=]() -> int {
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t n_aiv = std::min(AIV_MAX, ascendcPlatform->GetCoreNumAiv());
    kvcache_ops::pac_coder::pac_encode(input_data_ptr, meta_data_data_ptr,
                                       output_data_ptr, output_lengths_data_ptr,
                                       stream, n_aiv, nbins, ntokens, nlayers,
                                       nchannels, chunk_size, workGM_ptr);
    return 0;
  };

  at_npu::native::OpCommand cmd;
  cmd.Name("pac_encode").SetCustomHandler(_custom_handler).Run();
};

void pac_decode(const at::Tensor &meta_data, const at::Tensor &bytestreams,
                const at::Tensor &lengths, at::Tensor &output) {

  const auto meta_data_shape = meta_data.sizes();
  const int nlayers = meta_data_shape[0];
  const int nchannels = meta_data_shape[1];
  const int nbins = meta_data_shape[2];
  const int ntokens = output.sizes()[1];

  TORCH_CHECK(meta_data.device().is_privateuseone(),
              "meta_datas tensor should be on the NPU");
  TORCH_CHECK(bytestreams.device().is_privateuseone(),
              "Bytestreams tensor should be on the NPU");
  TORCH_CHECK(lengths.device().is_privateuseone(),
              "Lengths tensor should be on the NPU");
  TORCH_CHECK(output.device().is_privateuseone(),
              "Output tensor should be on the NPU");
  TORCH_CHECK(ntokens <= 256,
              "Number of tokens exceeds that supported be decode, handle by "
              "chunking the input."
              "Contact LMCache Ascend about changing this limitation");
  TORCH_CHECK(nbins <= 32,
              "Number of bins exceeds that supported be decode, contact."
              "LMCache Ascend about changing this limitation");
  TORCH_CHECK(
      nchannels % 32 == 0,
      "Decode implementation relies on the number of channels being a multiple "
      "of 32, contact LMCache Ascend about changing this limitation");

  const c10::OptionalDeviceGuard device_guard(device_of(bytestreams));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  auto meta_data_data_ptr = static_cast<uint8_t *>(meta_data.data_ptr());
  auto bytestreams_data_ptr = static_cast<uint8_t *>(bytestreams.data_ptr());
  auto lengths_data_ptr = static_cast<uint8_t *>(lengths.data_ptr());
  auto output_data_ptr = static_cast<uint8_t *>(output.data_ptr());

  const char *socName = aclrtGetSocName();
  auto _custom_handler = [=]() -> int {
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t n_aiv = std::min(AIV_MAX, ascendcPlatform->GetCoreNumAiv());
    kvcache_ops::pac_coder::pac_decode(
        meta_data_data_ptr, lengths_data_ptr, bytestreams_data_ptr,
        output_data_ptr, stream, n_aiv, nbins, ntokens, nlayers, nchannels);
    return 0;
  };

  at_npu::native::OpCommand cmd;
  cmd.Name("pac_decode").SetCustomHandler(_custom_handler).Run();
};
