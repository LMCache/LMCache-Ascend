# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache_tests.v1.utils import *
import torch

# First Party
from lmcache_ascend.v1.npu_connector import VLLMPagedMemNPUConnectorV2


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
    kv_format=1,  # 1:MERGED KV 2:SEPARATE KV
):
    """
    Check whether two paged kv caches are the same at slot_mapping.
    Supports both MERGED_KV and SEPARATE_KV formats.
    """
    token_dim = 0
    num_tokens = slot_mapping.shape[0]
    for left_kv, right_kv in zip(left, right, strict=False):
        # MERGED_KV only
        if kv_format == 1 and not vllm_two_major:
            left_kv = left_kv.transpose(0, 1)
            right_kv = right_kv.transpose(0, 1)

        left_k = left_kv[0].reshape(-1, num_heads, head_size)
        left_v = left_kv[1].reshape(-1, num_heads, head_size)
        right_k = right_kv[0].reshape(-1, num_heads, head_size)
        right_v = right_kv[1].reshape(-1, num_heads, head_size)

        assert len(left_k.shape) == 3
        assert len(left_v.shape) == 3
        assert len(right_k.shape) == 3
        assert len(right_v.shape) == 3

        assert left_k.shape[token_dim] >= num_tokens
        assert left_v.shape[token_dim] >= num_tokens
        assert right_k.shape[token_dim] >= num_tokens
        assert right_v.shape[token_dim] >= num_tokens

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
