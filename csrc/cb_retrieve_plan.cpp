// SPDX-License-Identifier: Apache-2.0

#include "cb_retrieve_plan.h"

#include "pos_kernels.h"
#include "utils.h"

#include <acl/acl.h>
#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <vector>

void execute_cb_retrieve_plan(const torch::Device &device,
                              size_t host_buffer_alignment,
                              const std::vector<CBGroupSpec> &group_specs,
                              const std::vector<CBRetrieveStep> &steps) {
  const c10::OptionalDeviceGuard device_guard(device);
  const char *soc_name = aclrtGetSocName();
  auto *platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(soc_name);
  const uint32_t aiv_num = platform->GetCoreNumAiv();
  // .stream() drains torch_npu's queued torch operations once. This orders a
  // preceding async slot_mapping.copy_ before all direct ACL/kernel enqueues.
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  for (const auto &step : steps) {
    for (const auto &copy : step.staging) {
      lmcache_memcpy_async_on_stream(copy.dest, copy.src, copy.nbytes,
                                     TransferDirection::H2D, copy.host_offset,
                                     host_buffer_alignment, stream);
    }
    // Match the CUDA executor's group filtering and boundary checks.
    // Kernel-specific layout restrictions are checked by the native entries.
    for (int group_idx = 0; group_idx < static_cast<int>(group_specs.size());
         ++group_idx) {
      const auto &group = group_specs[group_idx];
      if (group.cos_sin_cache != 0) {
        for (const auto &rope : step.ropes) {
          if (rope.group_idx != group_idx) {
            continue;
          }
          TORCH_CHECK(rope.slot_idx >= 0 &&
                          rope.slot_idx <
                              static_cast<int>(group.temp_buffer_ptrs.size()),
                      "CBRopeVar.slot_idx out of range: ", rope.slot_idx);
          const uintptr_t key =
              static_cast<uintptr_t>(group.temp_buffer_ptrs[rope.slot_idx]) +
              static_cast<uintptr_t>(group.rope_base_offset);
          rotary_embedding_k_fused_range_on_stream(
              key, group.cos_sin_cache, rope.old_st, rope.cur_st,
              group.slot_tokens, group.num_layers, group.head_size,
              group.rope_head_stride, group.rope_num_kv_heads, group.rot_dim,
              group.key_scalar_type, group.cache_scalar_type, group.is_neox,
              stream);
        }
      }
      for (const auto &scatter : step.scatters) {
        if (scatter.group_idx != group_idx) {
          continue;
        }
        TORCH_CHECK(scatter.slot_idx >= 0 &&
                        scatter.slot_idx <
                            static_cast<int>(group.temp_buffer_ptrs.size()),
                    "CBScatterVar.slot_idx out of range: ", scatter.slot_idx);
        TORCH_CHECK(scatter.n_tok >= 0 && scatter.n_tok <= group.slot_tokens,
                    "CBScatterVar.n_tok (", scatter.n_tok,
                    ") exceeds slot capacity ", group.slot_tokens);
        TORCH_CHECK(scatter.slot_mapping_offset >= 0 &&
                        scatter.slot_mapping_offset + scatter.n_tok <=
                            group.slot_mapping_capacity,
                    "CBScatterVar slot_mapping slice [",
                    scatter.slot_mapping_offset, ", ",
                    scatter.slot_mapping_offset + scatter.n_tok,
                    ") exceeds capacity ", group.slot_mapping_capacity);
        if (scatter.n_tok == 0) {
          continue;
        }
        const uintptr_t mapping =
            group.slot_mapping_base +
            static_cast<uintptr_t>(scatter.slot_mapping_offset) *
                sizeof(int64_t);
        multi_layer_token_kv_transfer_on_stream(
            group.paged_kv_ptrs,
            static_cast<uintptr_t>(group.temp_buffer_ptrs[scatter.slot_idx]),
            mapping, scatter.n_tok, TransferDirection::H2D,
            group.engine_kv_format, group.num_layers, group.slot_tokens,
            group.hidden_elems, group.element_size, group.page_buffer_size,
            group.block_size, group.block_stride_elems, aiv_num, stream);
      }
    }
  }
}
