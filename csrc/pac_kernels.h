#pragma once
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

void pac_prepare_enc_metadata(const at::Tensor &input_sym,
                              const at::Tensor &meta_data);

void pac_encode(const at::Tensor &input_sym, const at::Tensor &meta_data,
                at::Tensor &output_buffer, at::Tensor &output_lengths);

void pac_decode(const at::Tensor &meta_data, const at::Tensor &bytestreams,
                const at::Tensor &lengths, at::Tensor &output);

namespace kvcache_ops {
namespace pac_coder {

void pac_prep_enc_metadata(uint8_t *input_data_ptr, uint8_t *meta_data_ptr,
                           void *stream, const int n_aiv, const int n_bins,
                           const int n_tokens, const int n_layers,
                           const int n_channels);

void pac_encode(uint8_t *input_data_ptr, uint8_t *meta_data_ptr,
                uint8_t *output_data_ptr, uint8_t *output_lengths_data_ptr,
                void *stream, const int n_aiv, const int n_bins,
                const int n_tokens, const int n_layers, const int n_channels,
                const int chunk_size, uint8_t *workGM_ptr);

void pac_decode(uint8_t *meta_data_ptr, uint8_t *cum_lens_ptr,
                uint8_t *bytestream_ptr, uint8_t *output_data_ptr, void *stream,
                const int n_aiv, const int n_bins, const int n_tokens,
                const int n_layers, const int n_channels);
} // namespace pac_coder
} // namespace kvcache_ops
