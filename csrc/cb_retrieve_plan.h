// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mp_mem_kernels.h"

#include <cstdint>
#include <torch/extension.h>
#include <vector>

// Field-for-field compatible with LMCache's CUDA CacheBlend plan contract.
struct CBGroupSpec {
  uintptr_t paged_kv_ptrs;
  std::vector<int64_t> temp_buffer_ptrs;
  int num_layers;
  int slot_tokens;
  int hidden_elems;
  int element_size;
  EngineKVFormat engine_kv_format;
  int page_buffer_size;
  int block_size;
  int head_size;
  int64_t block_stride_elems;
  uintptr_t slot_mapping_base;
  int64_t slot_mapping_capacity;
  uintptr_t cos_sin_cache;
  int rot_dim;
  int rope_num_kv_heads;
  int64_t rope_head_stride;
  int key_scalar_type;
  int cache_scalar_type;
  bool is_neox;
  int64_t rope_base_offset;
};

struct CBRopeVar {
  int group_idx;
  int slot_idx;
  int64_t old_st;
  int64_t cur_st;
};

struct CBScatterVar {
  int group_idx;
  int slot_idx;
  int64_t slot_mapping_offset;
  int n_tok;
};

struct CBRetrieveStep {
  std::vector<StagingCopy> staging;
  std::vector<CBRopeVar> ropes;
  std::vector<CBScatterVar> scatters;
};

// Execute the whole request on the caller's current NPU stream. P1 is
// intentionally serial: staging, then K-only re-RoPE, then scatter per step.
void execute_cb_retrieve_plan(const torch::Device &device,
                              size_t host_buffer_alignment,
                              const std::vector<CBGroupSpec> &group_specs,
                              const std::vector<CBRetrieveStep> &steps);
