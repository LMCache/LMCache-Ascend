#pragma once
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

void encode_ascend_new(const at::Tensor &cdf, const at::Tensor &input_sym,
                       at::Tensor &output_buffer, at::Tensor &output_lengths);

void decode_ascend_new(const at::Tensor &cdf, const at::Tensor &bytestreams,
                       const at::Tensor &lengths, at::Tensor &output);

void decode_ascend_prefsum(const at::Tensor &cdf, const at::Tensor &bytestreams,
                           const at::Tensor &lengths, at::Tensor &output);

at::Tensor calculate_cdf(const at::Tensor &input, const int n_bins);

namespace kvcache_ops {
namespace cachegen {
void calculate_cdf(uint8_t *input, uint8_t *output, void *stream,
                   const int n_aiv, const int n_bins, const int n_tokens,
                   const int n_layers, const int n_channels);

void encode_v2(uint8_t *cdf_data_ptr, uint8_t *input_data_ptr,
               uint8_t *output_data_ptr, uint8_t *output_lengths_data_ptr,
               void *stream, const int n_aiv, const int n_bins,
               const int n_tokens, const int n_layers, const int n_channels,
               const int chunk_size);

void decode_v2(uint8_t *cdf_data_ptr, uint8_t *bytestreams_data_ptr,
               uint8_t *lengths_data_ptr, uint8_t *output_data_ptr,
               void *stream, const int n_aiv, const int n_bins,
               const int n_tokens, const int n_layers, const int n_channels);
} // namespace cachegen
} // namespace kvcache_ops
