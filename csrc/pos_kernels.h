#pragma once
#include "utils.h"
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace kvcache_ops {

void rotary_embedding_kernel_dispatch(
    uint64_t blockDim, void *stream, uint8_t *oldPositions,
    uint8_t *newPositions, uint8_t *key, uint8_t *cosSinCache, uint8_t *keyOut,
    uint64_t numTokens, uint64_t numHeads, uint64_t headSize,
    uint64_t rotaryDim, uint64_t kLeadingDimension, uint64_t isNeoxStyle,
    uint64_t frontCore, uint64_t tailCore, uint64_t numTokensFrontCoreEachLoop,
    uint64_t numTokensTailCoreEachLoop, uint64_t numTokensEachFrontCore,
    uint64_t numTokensEachTailCore, uint64_t loopTimeEachFrontCore,
    uint64_t loopTimeEachTailCore, uint64_t numTokensFrontCoreLastLoop,
    uint64_t numTokensTailCoreLastLoop, uint64_t tilingKey);

void rotary_embedding_range_kernel_dispatch(
    uint64_t blockDim, void *stream, uint8_t *key, uint8_t *cosSinCache,
    uint64_t oldPositionStart, uint64_t newPositionStart,
    uint64_t positionSlots, uint64_t numTokens, uint64_t numHeads,
    uint64_t headSize, uint64_t rotaryDim, uint64_t kLeadingDimension,
    uint64_t isNeoxStyle, uint64_t frontCore, uint64_t tailCore,
    uint64_t numTokensFrontCoreEachLoop, uint64_t numTokensTailCoreEachLoop,
    uint64_t numTokensEachFrontCore, uint64_t numTokensEachTailCore,
    uint64_t loopTimeEachFrontCore, uint64_t loopTimeEachTailCore,
    uint64_t numTokensFrontCoreLastLoop, uint64_t numTokensTailCoreLastLoop,
    uint64_t tilingKey);

} // namespace kvcache_ops

void rotary_embedding_k_fused(torch::Tensor &oldPositions,
                              torch::Tensor &newPositions, torch::Tensor &key,
                              int64_t headSize, torch::Tensor &cosSinCache,
                              bool isNeox);

// Re-RoPE a contiguous CacheBlend K plane in place. Positions are generated
// inside the kernel and repeat every slot_tokens across flattened layers.
void rotary_embedding_k_fused_range_on_stream(
    uintptr_t key, uintptr_t cos_sin_cache, int64_t old_st, int64_t cur_st,
    int64_t slot_tokens, int64_t num_layers, int64_t head_size,
    int64_t head_stride, int64_t num_kv_heads, int64_t rot_dim,
    int key_scalar_type, int cache_scalar_type, bool is_neox, void *stream);
