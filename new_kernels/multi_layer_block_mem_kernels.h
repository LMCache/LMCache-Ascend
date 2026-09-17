// SPDX-License-Identifier: Apache-2.0

#ifndef MULTI_LAYER_BLOCK_MEM_KERNELS_H
#define MULTI_LAYER_BLOCK_MEM_KERNELS_H

#include "block_transfer_layout.h"

namespace kvcache_ops {

// Host-side launcher for the block-level multi-layer KV transfer kernel.
// One launch moves one LMCache memory object (phase-1 semantics: the C++
// boundary in csrc/mp_mem_kernels.cpp loops over the object batch).
//
// Layout semantics per engine KV format:
//   16 (NL_X_TWO_X_NB_BS_NH_HS): per-layer (K, V) equal-width NHD tensors
//       -> LMC [2, L, T, NH*HS] (2LTD: K slab first, V slab after); work
//       items are (layer, plane, block) since K/V live in disjoint slabs.
//   17 (NL_X_NP_X_NB_BS_ONE_HS): per-layer tuple of 1..4 single-head planes
//       (MLA/DSA/DSv4) -> LMC [L, T, sum(payload)] packed byte rows; work
//       items are (layer, block) and one AIV core moves every plane of the
//       page, so no two cores ever share a 32B LMC line.
//   13 (NL_X_NB_BS_NH_CS): one fused plane per layer -> LMC [1, L, T, NH*CS].
//
// Pointer table layout: ptrs[layer * num_planes + plane], each entry the
// plane tensor view's data_ptr() (storage_offset already included; planes
// may be independently allocated or views into one shared pool).
//
// The kernel is a pure byte mover: no dtype conversion, no HND transpose,
// no model-name dispatch. All geometry (payload/block stride/offsets) is
// derived and validated by the host-side prepare_group and passed in
// through the layout POD.
void multi_layer_block_transfer_kernel(
    uint32_t block_dim,            // min(AIV cores, total work items)
    void *stream,                  // aclrtStream (current NPU stream)
    uint8_t *paged_buffer_ptrs,    // interleaved pointer table (see above)
    uint8_t *lmcache_obj,          // LMC object device VA (maybe staged)
    uint8_t *engine_block_ids,     // int64 array: logical -> engine block
    int32_t num_blocks_per_object, // == lmcache_chunk_size / bs (host-checked)
    int32_t skip_prefix_n_blocks,  // leading blocks to skip (prefix dedup)
    int32_t nl,                    // layers in this group
    int32_t nb,                    // engine pool block count (ID upper bound)
    int32_t bs,                    // tokens/slots per block
    bool separate_plane,           // true: (layer, plane, block) work items
    BlockTransferLayout layout,    // full byte layout (block_transfer_layout.h)
    bool to_engine);               // true = H2D (retrieve), false = D2H (store)

} // namespace kvcache_ops

#endif // MULTI_LAYER_BLOCK_MEM_KERNELS_H
