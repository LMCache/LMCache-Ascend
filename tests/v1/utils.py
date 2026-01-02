# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache_ascend.v1.npu_connector import VLLMPagedMemNPUConnectorV2


def create_npu_connector(hidden_dim, num_layers):
    return VLLMPagedMemNPUConnectorV2(hidden_dim, num_layers)


def generate_kv_cache_paged_list_tensors(
    num_blocks,
    device,
    num_layers,
    num_heads,
    head_size,
    block_size=16,
    dtype=torch.bfloat16,
    use_mla=False,
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
    left, right, slot_mapping, num_heads=8, head_size=128, vllm_two_major=True
):
    """
    check whether two paged kv caches are the same at slot_mapping
    """
    token_dim = 0
    num_tokens = slot_mapping.shape[0]
    for left_kv, right_kv in zip(left, right, strict=False):
        if not vllm_two_major:
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
