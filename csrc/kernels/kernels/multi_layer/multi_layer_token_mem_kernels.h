// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace kvcache_ops {

constexpr int64_t kTokenTransferUbBytes = 128 * 1024;

// Dense token rows; only the engine block axis may carry padding. LMC is
// [2, num_layers, slot_tokens, row_bytes]. Capacity and transfer count differ.
// Format 16: pointers[layer * 2 + plane], including each view's storage offset.
// Format 2: pointers[layer] + plane * block_size * row_bytes.
struct TokenKVTransferLayout {
  int32_t num_layers;
  int32_t slot_tokens;
  int32_t block_size;
  int64_t page_buffer_size;
  int64_t row_bytes;
  int64_t block_stride_bytes;
  bool one_pointer_per_layer;
};

// One launch covers all layers, K/V planes and valid tokens. Mapping is int64;
// invalid slots are skipped. H2D mappings must not target overlapping rows.
// K/V use a common block stride; their views may share the same allocation.
// The caller owns all buffers until stream completion.
// H2D scatters LMC -> engine; D2H gathers engine -> LMC. No CPU copy occurs
// here.
void multi_layer_token_kv_transfer_kernel(
    uint32_t block_dim, void* stream, uint8_t* paged_kv_ptrs,
    uint8_t* lmc_buffer, uint8_t* slot_mapping, int32_t num_tokens,
    TokenKVTransferLayout layout, bool page_to_lmc);

} // namespace kvcache_ops
