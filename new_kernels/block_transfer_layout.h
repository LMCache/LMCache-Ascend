// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// Host/device shared POD for the MP block-level KV transfer.
//
// The host parser (csrc/mp_mem_kernels.cpp, prepare_group) resolves a
// PageBufferShapeDesc plus EngineKVFormat into this flat byte layout; the
// device kernel consumes it verbatim and never re-interprets the format.
// All sizes/offsets are in bytes (int64): planes may mix dtypes, and
// engine pools may exceed 32-bit element counts.
//
// Layout contracts:
//   - Engine plane p: [NB, BS, NH, HS] with dense token rows, so the
//     engine-side token step equals payload_bytes. Only the block axis may
//     carry padding, expressed by engine_block_stride_bytes.
//   - LMC side (format 17): one packed row per token of
//     lmc_token_stride_bytes = sum(payload_bytes); plane p sits at column
//     offset lmc_base_offset_bytes (prefix sum) inside the row.
//   - Format 16: two independent slabs of nl * lmc_layer_stride_bytes;
//     planes[1].lmc_base_offset_bytes == nl * lmc_layer_stride_bytes.
//   - Format 13: single plane, lmc_base_offset_bytes == 0.

namespace kvcache_ops {

constexpr int kMaxPlanes = 4;
// Conservative UB budget assumed by the device kernel (see
// multi_layer_block_mem_kernels.cpp): depth-2 double buffering means a
// single segment may use at most half of it.
constexpr int64_t kBlockTransferUbBytes = 128 * 1024;
constexpr int32_t kBlockTransferQueueDepth = 2;

struct PlaneLayout {
  int64_t payload_bytes;             // valid bytes per token slot
  int64_t engine_block_stride_bytes; // dim-0 block step on the engine side
  int64_t lmc_base_offset_bytes;     // plane start inside the LMC object
};

struct BlockTransferLayout {
  int32_t num_planes;              // 1..kMaxPlanes (physical planes/layer)
  int32_t reserved;                // keeps the trailing fields 8B-aligned
  int64_t lmc_token_stride_bytes;  // LMC packed row width (all planes)
  int64_t lmc_layer_stride_bytes;  // bytes per layer = slots_per_object * row
  int64_t lmc_object_bytes;        // total object size
  PlaneLayout planes[kMaxPlanes];
};

// ABI guards: the host fills this POD field-by-field and the device reads
// it through the flattened kernel signature; both sides must agree on the
// layout. int32 + padding keeps every int64 8B-aligned.
static_assert(sizeof(PlaneLayout) == 24, "PlaneLayout ABI size mismatch");
static_assert(sizeof(BlockTransferLayout) == 128,
              "BlockTransferLayout ABI size mismatch");
static_assert(offsetof(BlockTransferLayout, planes) == 32,
              "BlockTransferLayout planes offset mismatch");

}  // namespace kvcache_ops
