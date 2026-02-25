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
