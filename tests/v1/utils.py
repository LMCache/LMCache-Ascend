# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple

# Third Party
from lmcache_tests.v1.utils import *
import torch

# First Party
from lmcache_ascend.v1.npu_connector.npu_connectors import VLLMPagedMemNPUConnectorV2


def create_npu_connector(hidden_dim, num_layers):
    return VLLMPagedMemNPUConnectorV2(hidden_dim, num_layers)


def generate_kv_cache_paged_list_tensors(
    num_blocks,
    device,
    block_size=16,
    dtype=torch.bfloat16,
    use_mla=False,
    num_layers=32,
    num_heads=8,
    head_size=128,
    vllm_two_major=True,
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tensor]
    where KV are in the same tensor
    """
    ret = []
    vllm_shapes = (
        [2, num_blocks, block_size, num_heads, head_size]
        if vllm_two_major
        else [num_blocks, 2, block_size, num_heads, head_size]
    )
    shape = [num_blocks, block_size, head_size] if use_mla else vllm_shapes

    for i in range(num_layers):
        kv = torch.rand(shape, dtype=dtype, device=device)
        ret.append(kv)

    return ret


def generate_kv_cache_paged_list_tuple_tensors(
    num_blocks,
    device,
    num_layers,
    num_heads,
    head_size,
    block_size=16,
    dtype=torch.bfloat16,
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tensor]
    where KV are in the same tensor
    """
    ret = []
    key_shape = [num_blocks, block_size, num_heads, head_size]
    value_shape = [num_blocks, block_size, num_heads, head_size]

    for i in range(num_layers):
        key = torch.rand(key_shape, dtype=dtype, device=device)
        value = torch.rand(value_shape, dtype=dtype, device=device)
        ret.append((key, value))

    return ret


def check_paged_kv_cache_equal(
    left,
    right,
    slot_mapping,
    num_heads=8,
    head_size=128,
    vllm_two_major=True,
    kv_format=1,  # 1:MERGED KV 2:SEPARATE KV 3:MLA_KV 4:DSA_KV
    kv_lora_rank=0,
    qk_rope_head_dim=0,
    dsa_head_dim=0,
):
    """
    Check whether two paged kv caches are the same at slot_mapping.
    Supports MERGED_KV, SEPARATE_KV, MLA_KV, and DSA_KV formats.
    """
    token_dim = 0
    num_tokens = slot_mapping.shape[0]
    for left_kv, right_kv in zip(left, right, strict=False):
        # MERGED_KV only
        if kv_format == 1 and not vllm_two_major:
            left_kv = left_kv.transpose(0, 1)
            right_kv = right_kv.transpose(0, 1)

        # Handle different KV cache formats
        if kv_format == 3:  # MLA_KV: (k_cache, v_cache) with different shapes
            # MLA format: k_cache=[blocks, block_size, kv_lora_rank],
            #             v_cache=[blocks, block_size, qk_rope_head_dim]
            left_k = left_kv[0].reshape(-1, kv_lora_rank)
            left_v = left_kv[1].reshape(-1, qk_rope_head_dim)
            right_k = right_kv[0].reshape(-1, kv_lora_rank)
            right_v = right_kv[1].reshape(-1, qk_rope_head_dim)
        elif kv_format == 4:  # DSA_KV: (k_cache, v_cache, dsa_k_cache)
            # DSA format: k_cache=[blocks, block_size, kv_lora_rank],
            #             v_cache=[blocks, block_size, qk_rope_head_dim],
            #             dsa_k_cache=[blocks, block_size, 1, 128]
            left_k = left_kv[0].reshape(-1, kv_lora_rank)
            left_v = left_kv[1].reshape(-1, qk_rope_head_dim)
            left_dsa = left_kv[2].reshape(-1, 128)
            right_k = right_kv[0].reshape(-1, kv_lora_rank)
            right_v = right_kv[1].reshape(-1, qk_rope_head_dim)
            right_dsa = right_kv[2].reshape(-1, 128)

            assert len(left_dsa.shape) == 2
            assert len(right_dsa.shape) == 2
            assert left_dsa.shape[token_dim] >= num_tokens
            assert right_dsa.shape[token_dim] >= num_tokens
            assert (left_dsa[slot_mapping, :] == right_dsa[slot_mapping, :]).all()
        else:  # MERGED_KV or SEPARATE_KV with same K/V shapes
            left_k = left_kv[0].reshape(-1, num_heads, head_size)
            left_v = left_kv[1].reshape(-1, num_heads, head_size)
            right_k = right_kv[0].reshape(-1, num_heads, head_size)
            right_v = right_kv[1].reshape(-1, num_heads, head_size)

        assert len(left_k.shape) >= 2
        assert len(left_v.shape) >= 2
        assert len(right_k.shape) >= 2
        assert len(right_v.shape) >= 2

        assert left_k.shape[token_dim] >= num_tokens
        assert left_v.shape[token_dim] >= num_tokens
        assert right_k.shape[token_dim] >= num_tokens
        assert right_v.shape[token_dim] >= num_tokens

        if kv_format in (3, 4):  # MLA/DSA format (2D tensors after reshape)
            assert (left_k[slot_mapping, :] == right_k[slot_mapping, :]).all()
            assert (left_v[slot_mapping, :] == right_v[slot_mapping, :]).all()
        else:  # MERGED/SEPARATE format (3D tensors)
            assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
            assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()


def generate_sglang_npu_kv_cache(
    num_layers,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    device="npu",
    dtype=torch.bfloat16,
):
    """
    Generate SGLang NPU Layer-Concatenated format KV cache.

    Format: [2, layer_nums, num_blocks, block_size, num_heads, head_dim]
    kvcaches = [K_all_layers, V_all_layers]
    - K_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    """
    shape = [num_layers, num_blocks, block_size, num_heads, head_size]

    k_tensor = torch.rand(shape, dtype=dtype, device=device)
    v_tensor = torch.rand(shape, dtype=dtype, device=device)

    return [k_tensor, v_tensor]


def check_sglang_npu_kv_cache_equal(
    left, right, slot_mapping, num_heads=8, head_size=128
):
    """
    Check whether two SGLang NPU KV caches are the same at slot_mapping.

    Format: [2, layer_nums, num_blocks, block_size, num_heads, head_dim]
    """
    num_tokens = slot_mapping.shape[0]

    left_k = left[0]
    left_v = left[1]
    right_k = right[0]
    right_v = right[1]

    for layer_id in range(left_k.shape[0]):
        left_k_layer = left_k[layer_id].reshape(-1, num_heads, head_size)
        left_v_layer = left_v[layer_id].reshape(-1, num_heads, head_size)
        right_k_layer = right_k[layer_id].reshape(-1, num_heads, head_size)
        right_v_layer = right_v[layer_id].reshape(-1, num_heads, head_size)

        assert left_k_layer.shape[0] >= num_tokens
        assert left_v_layer.shape[0] >= num_tokens
        assert right_k_layer.shape[0] >= num_tokens
        assert right_v_layer.shape[0] >= num_tokens

        assert (
            left_k_layer[slot_mapping, :, :] == right_k_layer[slot_mapping, :, :]
        ).all()

        assert (
            left_v_layer[slot_mapping, :, :] == right_v_layer[slot_mapping, :, :]
        ).all()


def generate_mla_kv_cache(
    num_blocks: int,
    device: str,
    num_layers: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate MLA (Multilayer Attention) format KV cache.
    k.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    v.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]

    Returns: List of tuples, each tuple is (k_cache, v_cache) for one layer
    """
    ret = []
    k_shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    v_shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]

    for _ in range(num_layers):
        k_cache = torch.rand(k_shape, dtype=dtype, device=device)
        v_cache = torch.rand(v_shape, dtype=dtype, device=device)
        ret.append((k_cache, v_cache))
    return ret


def generate_dsa_kv_cache(
    num_blocks: int,
    device: str,
    num_layers: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dsa_head_dim: int = 128,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate DSA (Deep Sparse Attention) format KV cache with dsa_k_cache.
    k.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    v.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]
    dsa_k.shape = [num_blocks, block_size, 1, dsa_head_dim]

    Returns: List of tuples, each tuple is (k_cache, v_cache, dsa_k_cache) for one layer
    """
    ret = []
    k_shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    v_shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]
    dsa_k_shape = [num_blocks, block_size, 1, dsa_head_dim]

    for _ in range(num_layers):
        k_cache = torch.rand(k_shape, dtype=dtype, device=device)
        v_cache = torch.rand(v_shape, dtype=dtype, device=device)
        dsa_k_cache = torch.rand(dsa_k_shape, dtype=dtype, device=device)
        ret.append((k_cache, v_cache, dsa_k_cache))
    return ret
