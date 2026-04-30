# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from typing import List, Tuple, Union

# Third Party
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)


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

    def is_separate_format(self) -> bool:
        return self == KVCacheFormat.SEPARATE_KV

    def is_merged_format(self) -> bool:
        return self == KVCacheFormat.MERGED_KV

    def is_mla_format(self) -> bool:
        return self == KVCacheFormat.MLA_KV

    def is_dsa_format(self) -> bool:
        return self == KVCacheFormat.DSA_KV

    def is_tuple_format(self) -> bool:
        return self in (
            KVCacheFormat.SEPARATE_KV,
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
        )

    def get_kv_size(self) -> int:
        if self == KVCacheFormat.DSA_KV:
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

        if isinstance(first_cache, tuple):
            tuple_len = len(first_cache)

            # DSA_KV: tuple with 3 elements (k_cache, v_cache, dsa_k_cache)
            if tuple_len == 3:
                k_cache, v_cache, dsa_k_cache = first_cache
                assert k_cache.shape[2] == 1, "DSA_KV num_kv_heads != 1"
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
                assert k_cache.shape[2] == 1, "MLA_KV num_kv_heads != 1"
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

            # MLA detect (single tensor format)
            # MLA Shape: [num_blocks, block_size, head_size] (3D)
            #         or: [1, num_blocks, block_size, head_size] (4D with first dim = 1)
            is_mla_shape = (ndim == 3) or (ndim == 4 and shape[0] == 1)
            if use_mla or is_mla_shape:
                return KVCacheFormat.MERGED_KV

            # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
            if ndim == 5 and shape[0] == 2:
                return KVCacheFormat.MERGED_KV

            # Flash Infer: [num_blocks, 2, block_size, num_heads, head_size]
            if ndim == 5 and shape[1] == 2:
                return KVCacheFormat.MERGED_KV

        return KVCacheFormat.UNDEFINED
