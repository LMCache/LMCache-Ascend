# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from typing import List, Sequence, Tuple, Union

# Third Party
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)

# Shared type alias: unifies single-tensor layers with multi-view tuple/list
# entries so _is_shared_storage_blob and callers accept all per-layer shapes.
_BlobEntry = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]


# Extracts the block_size dimension from a KV plane tensor (dim-1 for 3-D/4-D).
# Used by _is_multi_plane_tuple and the NPU kernel param builder to derive
# per-plane paging geometry.
def _plane_block_size(tensor: torch.Tensor) -> int:
    # block_size lives at dim-1 for both 3-D (nb, bs, hidden) and
    # 4-D (nb, bs, nh, hs) layouts; ndim < 3 has no paging geometry.
    if tensor.ndim >= 3:
        return int(tensor.shape[1])
    raise ValueError(f"Unexpected KV plane ndim={tensor.ndim}")


# Detects heterogeneous-block-size tuples produced by BUNDLE_DSV4_TUPLES so
# KVCacheFormat.detect returns MULTI_PLANE_KV and the NPU connector dispatches
# to the multi-plane kernel path instead of the single-tensor SEPARATE_KV path.
def _is_multi_plane_tuple(cache_entry: Sequence[torch.Tensor]) -> bool:
    """True when planes have heterogeneous block sizes (independent paging)."""
    # Shared-storage blobs are one physical allocation under one scheduler group,
    # so block_size is uniform by construction — they are never multi-plane.
    if _is_shared_storage_blob(cache_entry):
        return False
    tensors = [
        t
        for t in cache_entry
        if isinstance(t, torch.Tensor) and t.ndim >= 3
    ]
    if len(tensors) < 2:
        return False
    block_sizes = {_plane_block_size(t) for t in tensors}
    return len(block_sizes) > 1


# Distinguishes vLLM-Ascend multi-dtype reinterpretation views (bf16 + int8
# over one allocation) from genuinely independent multi-plane tuples, so the
# bundling path and format detector do not misclassify shared blobs as MULTI_PLANE_KV.
def _is_shared_storage_blob(cache_entry: _BlobEntry) -> bool:
    """True when all tensors in a per-layer entry are full reinterpretations
    of one physical allocation (same storage *and* same starting address).

    Multi-spec compress layers expose multiple dtype/block-size views over a
    single ``int8`` blob allocated by vLLM-Ascend (see ``model_runner_v1``).

    This must NOT match Mamba/state entries where multiple tensors are packed
    as *slices* of one allocation at different offsets — those have identical
    ``untyped_storage().data_ptr()`` but distinct ``data_ptr()`` values.
    """
    if not isinstance(cache_entry, (tuple, list)) or len(cache_entry) < 2:
        return False
    tensors = [t for t in cache_entry if isinstance(t, torch.Tensor)]
    if len(tensors) < 2:
        return False
    storages = {t.untyped_storage().data_ptr() for t in tensors}
    if len(storages) != 1:
        return False
    # All views must start at the same byte offset within the storage.
    # Packed state slices (Mamba) start at different offsets; true
    # reinterpretation blobs (e.g. bfloat16 + int8 views) all start at 0.
    data_ptrs = {t.data_ptr() for t in tensors}
    return len(data_ptrs) == 1


# Picks the canonical (largest-byte-coverage) view from a shared-storage blob
# so the connector and layer-grouping logic derive block_size, hidden_bytes,
# and dtype from a single consistent tensor rather than an arbitrary sub-view.
def _get_primary_blob_view(cache_entry: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the view with the largest byte coverage (canonical blob layout)."""
    tensors = [t for t in cache_entry if isinstance(t, torch.Tensor)]
    if not tensors:
        raise ValueError("Shared-storage blob entry has no tensors")
    return max(tensors, key=lambda t: t.numel() * t.element_size())


class KVCacheFormat(Enum):
    """
    The storage format enumeration of KV cache is used to distinguish
    the KV cache data structures of different versions of vLLM.

    The order of enum values MUST match the KVCacheFormat
    definition in kernels/types.h to ensure correct interoperability
    between Python and C++ code.
    """

    UNDEFINED = 0

    MERGED_KV = auto()
    """Merge format (eg: vLLM 0.9.2 ...)
    layer: [num_kv, num_blocks, block_size, num_heads, head_dim]
    """

    SEPARATE_KV = auto()
    """Separation format (eg: vLLM 0.11.0+ ...)
    layer: tuple: (K_tensor, V_tensor)
    - K_tensor.shape = [num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [num_blocks, block_size, num_heads, head_dim]

    eg: kvcaches[0] = (K, V)

    SGLang NPU Layer-Concatenated
    kvcaches = [K_all_layers, V_all_layers]
    - K_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    """

    MLA_KV = auto()
    """MLA format for DeepSeek V2/V3 models
    layer: tuple: (k_cache, v_cache) where K and V have different dimensions
    - k_cache.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - v_cache.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]

    This format is used when K/V shapes differ (detected automatically).
    """

    DSA_KV = auto()
    """DSA (Deep Sparse Attention) format for DeepSeek V3.2 sparse models
    layer: tuple: (k_cache, v_cache, dsa_k_cache)
    - k_cache.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - v_cache.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]
    - dsa_k_cache.shape = [num_blocks, block_size, 1, 128]

    This format is used for sparse attention with lightning indexer.
    """

    DSA_C8_KV = auto()
    """DSA + C8 indexer (multi-spec). Tuple (k, v, dsa_k, dsa_k_scale).
    dtypes may differ (e.g. bf16 k/v, int8 indexer k, fp16 scale).
    Slot indices align across tuple members (same num_blocks x block_size).
    Must match ``DSA_C8_KV`` in ``third_party/kvcache-ops/kernels/types.h``.
       layer: tuple: (k_cache, v_cache, dsa_k_cache, dsa_k_scale_cache)
    - k_cache.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - v_cache.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]
    - dsa_k_cache.shape = [num_blocks, block_size, 1, 128]  (int8)
    - dsa_k_scale_cache.shape = [num_blocks, block_size, 1, 1]  (float16)
    """

    MULTI_PLANE_KV = auto()
    """Multi-spec layer: tuple of N page buffers with heterogeneous block sizes
    and/or independent scheduler slot mappings per plane.
    Must match ``MULTI_PLANE_KV`` (6) in ``third_party/kvcache-ops/kernels/types.h``.
    """

    def is_separate_format(self) -> bool:
        return self == KVCacheFormat.SEPARATE_KV

    def is_merged_format(self) -> bool:
        return self == KVCacheFormat.MERGED_KV

    def is_mla_format(self) -> bool:
        return self == KVCacheFormat.MLA_KV

    def is_dsa_format(self) -> bool:
        return self in (KVCacheFormat.DSA_KV, KVCacheFormat.DSA_C8_KV)

    def is_tuple_format(self) -> bool:
        return self in (
            KVCacheFormat.SEPARATE_KV,
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
            KVCacheFormat.DSA_C8_KV,
            KVCacheFormat.MULTI_PLANE_KV,
        )

    def get_kv_size(self) -> int:
        # MULTI_PLANE_KV has a variable number of planes per layer, so there
        # is no fixed stride for a flat pointer table; 0 signals "use per-group pointers".
        if self == KVCacheFormat.MULTI_PLANE_KV:
            return 0
        if self == KVCacheFormat.DSA_C8_KV:
            return 4
        elif self == KVCacheFormat.DSA_KV:
            return 3
        elif self in (KVCacheFormat.SEPARATE_KV, KVCacheFormat.MLA_KV):
            return 2
        elif self == KVCacheFormat.MERGED_KV:
            return 1
        return 0

    @staticmethod
    def detect(
        kvcaches: List[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        use_mla: bool = False,
    ) -> "KVCacheFormat":
        """
        Automatically detect KV cache format based on data structure.

        Detection logic:
        1. DSA_KV: tuple with 3 elements (k_cache, v_cache, dsa_k_cache)
        2. MLA_KV: tuple with 2 elements where K/V shapes differ
        3. SEPARATE_KV: tuple with 2 elements where K/V shapes are same
        4. MERGED_KV: single tensor with specific shape patterns
        """
        if not kvcaches:
            return KVCacheFormat.UNDEFINED

        first_cache = kvcaches[0]

        # SGLang NPU: kvcaches = [K_tensor, V_tensor]
        if isinstance(kvcaches, list) and len(kvcaches) == 2:
            if isinstance(first_cache, torch.Tensor) and first_cache.ndim == 5:
                return KVCacheFormat.SEPARATE_KV

        if isinstance(first_cache, (tuple, list)):
            tuple_len = len(first_cache)
            tensors = [t for t in first_cache if isinstance(t, torch.Tensor)]

            if len(tensors) >= 2 and _is_multi_plane_tuple(first_cache):
                logger.debug(
                    "Detected MULTI_PLANE_KV format: %d planes, block_sizes=%s",
                    len(tensors),
                    sorted({_plane_block_size(t) for t in tensors}),
                )
                return KVCacheFormat.MULTI_PLANE_KV

            # DSA_C8_KV: tuple with 4 elements
            # (k_cache, v_cache, dsa_k_cache, dsa_k_scale_cache)
            if tuple_len == 4:
                if all(isinstance(t, torch.Tensor) for t in first_cache):
                    k_cache, v_cache, dsa_k_cache, dsa_k_scale = first_cache
                    if k_cache.shape != v_cache.shape:
                        logger.debug(
                            f"Detected DSA_C8_KV format: k_shape={k_cache.shape}, "
                            f"v_shape={v_cache.shape}, dsa_k_shape={dsa_k_cache.shape}"
                            f", dsa_k_scale_shape={dsa_k_scale.shape}"
                        )
                        return KVCacheFormat.DSA_C8_KV

            # DSA_KV: tuple with 3 elements (k_cache, v_cache, dsa_k_cache)
            if tuple_len == 3:
                k_cache, v_cache, dsa_k_cache = first_cache
                if all(isinstance(t, torch.Tensor) for t in first_cache):
                    if k_cache.shape != v_cache.shape:
                        logger.debug(
                            f"Detected DSA_KV format: k_shape={k_cache.shape}, "
                            f"v_shape={v_cache.shape}, dsa_k_shape={dsa_k_cache.shape}"
                        )
                        return KVCacheFormat.DSA_KV

            # MLA_KV or SEPARATE_KV: tuple with 2 elements
            if tuple_len == 2:
                k_cache, v_cache = first_cache
                if isinstance(k_cache, torch.Tensor) and isinstance(
                    v_cache, torch.Tensor
                ):
                    # MLA_KV: K/V shapes differ
                    if k_cache.shape != v_cache.shape:
                        logger.debug(
                            f"Detected MLA_KV format: k_shape={k_cache.shape}, "
                            f"v_shape={v_cache.shape}"
                        )
                        return KVCacheFormat.MLA_KV
                    # SEPARATE_KV: K/V shapes are same
                    return KVCacheFormat.SEPARATE_KV

            return KVCacheFormat.SEPARATE_KV

        elif isinstance(first_cache, torch.Tensor):
            ndim = first_cache.ndim
            shape = first_cache.shape

            # Flattened multi-spec / MLA page buffer: [num_blocks, block_size, hidden]
            # Only taken when tuple bundling is disabled (LMCACHE_ASCEND_BUNDLE_MULTI_SPEC=0).
            if ndim == 3:
                return KVCacheFormat.SEPARATE_KV

            # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
            if ndim == 5 and shape[0] == 2:
                return KVCacheFormat.MERGED_KV

            # Flash Infer: [num_blocks, 2, block_size, num_heads, head_size]
            if ndim == 5 and shape[1] == 2:
                return KVCacheFormat.MERGED_KV

        return KVCacheFormat.UNDEFINED
