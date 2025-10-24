# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Callable, Dict, Optional
import pytest

# Third Party
from vllm.model_executor.layers.rotary_embedding import get_rope as vllm_get_rope
import torch
from torch.utils.benchmark import Timer 
import numpy as np

from lmcache_ascend.v1.blend.positional_encoding import (
    BasicReverseRope,
    FusedRope,
)

# First Party
from lmcache.logging import init_logger
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)
# Add and test more types of rope
# (e.g., rope scaling, (non-)neox style, dtype, etc.)

class DummyFusedRope:
    """
    This implementation directly uses two separate RoPE kernel calls to ratate K cache from
    the old positions to the new positions.
    """

    def __init__(self, rope, reverse_rope, is_neox_style):
        self.rope = rope
        self.reverse_rope = reverse_rope
        self.is_neox_style = is_neox_style
        self.head_size = rope.head_size
        self.cos_sin_cache = rope.cos_sin_cache

    def fused_encode(self, old_positions, new_positions, k):
        q = torch.zeros_like(k)
        q, k = self.reverse_rope(old_positions, q, k)
        q, k = self.rope(new_positions, q, k)
        return k

    def __call__(self, old_positions, new_positions, k):
        return self.fused_encode(old_positions, new_positions, k)


def validate_rope_params(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
):
    if rotary_dim != head_size:
        logger.error("Currently KV blending only support rotary_dim == head_size.")
        return False

    if rope_scaling is not None:
        logger.error("Currently KV blending do not support rope scaling.")
        return False

    if partial_rotary_factor != 1.0:
        logger.error(
            "Currently KV blending do not support rotary factor other than 1.0."
        )
        return False

    return True

def validate_reverse_correctness(
    rope, 
    reverse_rope, 
    fused_rope, 
    fused_rope2, 
    head_size, 
    test_sizes,
    repeats=10
) -> bool:

    hidden_dim = head_size * 8
    all_passed = True

    for num_tokens in test_sizes:

        q_errors = []
        k_errors = []
        fused_k_errors = []
        dummy_fused_k_errors = []

        logger.info(f"=====  num_tokens = {num_tokens} =====")

        for run in range(repeats):

            dumb_q = torch.rand((num_tokens, hidden_dim), device="npu", dtype=rope.dtype)
            dumb_k = torch.rand((num_tokens, hidden_dim), device="npu", dtype=rope.dtype)
            positions = torch.arange(num_tokens, device="npu") 

            q1 = dumb_q.clone()
            k1 = dumb_k.clone()
            q1, k1 = rope(positions, q1, k1)
            q1, k1 = reverse_rope(positions, q1, k1)

            current_q_err = (dumb_q - q1).abs().max().item()
            current_k_err = (dumb_k - k1).abs().max().item()

            q_no_pos = dumb_q.clone()
            k_no_pos = dumb_k.clone()

            positions2 = torch.arange(100, 100 + num_tokens, device="npu")
            _, k_pos2 = rope(positions2, q_no_pos, k_no_pos) 
            
            # FusedRope
            k_no_pos_fused = dumb_k.clone()
            _, k_pos1_fused = rope(positions, q_no_pos, k_no_pos_fused)
            k_pos2_fused = fused_rope(positions, positions2, k_pos1_fused)
            current_fused_err = (k_pos2 - k_pos2_fused).abs().max().item()
            
            # DummyFusedRope
            k_no_pos_dummy = dumb_k.clone()
            _, k_pos1_dummy = rope(positions, q_no_pos, k_no_pos_dummy)
            k_pos2_dummy = fused_rope2(positions, positions2, k_pos1_dummy)
            current_dummy_err = (k_pos2 - k_pos2_dummy).abs().max().item()

            q_errors.append(current_q_err)
            k_errors.append(current_k_err)
            fused_k_errors.append(current_fused_err)
            dummy_fused_k_errors.append(current_dummy_err)

        avg_q = np.mean(q_errors)
        std_q = np.std(q_errors)
        avg_k = np.mean(k_errors)
        std_k = np.std(k_errors)
        avg_fused = np.mean(fused_k_errors)
        std_fused = np.std(fused_k_errors)
        avg_dummy = np.mean(dummy_fused_k_errors)
        std_dummy = np.std(dummy_fused_k_errors)

        logger.info(f"\n  Scale {num_tokens} Summary:")
        logger.info(f"    Q Error - Mean: {avg_q:.6f}, Std Dev: {std_q:.6f}")
        logger.info(f"    K Error - Mean: {avg_k:.6f}, Std Dev: {std_k:.6f}")
        logger.info(f"    Fused Rope K Error - Mean: {avg_fused:.6f}, Std Dev: {std_fused:.6f}")
        logger.info(f"    Dummy Fused Rope K Error - Mean: {avg_dummy:.6f}, Std Dev: {std_dummy:.6f}\n")

        threshold = 1
        size_passed = (avg_q < threshold and avg_k < threshold
                                and avg_fused < threshold and avg_dummy < threshold)
        if not size_passed:
            logger.error(f"Scale {num_tokens} failed the accuracy test!")
            all_passed = False
        else:
            logger.info(f"Scale {num_tokens} passed the accuracy test\n")

    return all_passed

def get_fused_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    test_sizes = None,
    partial_rotary_factor: float = 1.0,
) -> Optional[Callable[..., Any]]:

    if not validate_rope_params(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling,
        dtype,
        partial_rotary_factor,
    ):
        logger.warning(
            "The rope parameters is not supported! Cannot use cacheblend in this case"
        )
        return None

    rope = vllm_get_rope(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling,
        dtype,
        partial_rotary_factor,
    )

    reverse_rope = BasicReverseRope(rope, rotary_dim, is_neox_style)
    fused_rope = FusedRope(rope, is_neox_style)
    fused_rope2 = DummyFusedRope(rope, reverse_rope, is_neox_style)

    correct = validate_reverse_correctness(rope, reverse_rope, fused_rope, fused_rope2, head_size, test_sizes)
    if not correct:
        logger.error(
            "Fused/reverse rotary encoding is not correct! Will disable blending!"
        )
        return None

    return fused_rope


def verify_rope(
    head_dim,
    max_position,
    rope_theta,
    is_neox_style,
    dtype,
    test_sizes,
):
    
    fused_rotary_emb = get_fused_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=rope_theta,
        rope_scaling=None,
        is_neox_style=is_neox_style,
        dtype=dtype,
        test_sizes = test_sizes
    )

    assert fused_rotary_emb is not None, "Failed to get fused rotary embedding"


def benchmark_rope(fused_rope, dummy_rope, num_tokens, hidden_dim, device, dtype, repeats=100):

    old_positions = torch.arange(num_tokens, device=device)
    new_positions = torch.arange(100, 100 + num_tokens, device=device)  
    k = torch.randn((num_tokens, hidden_dim), device=device, dtype=dtype)  
    
    # warmup
    for _ in range(10):
        fused_rope(old_positions, new_positions, k.clone())
        dummy_rope(old_positions, new_positions, k.clone())
    torch.npu.synchronize()  

    fused_timer = Timer(
        stmt="fused_rope(old_pos, new_pos, k_clone)",
        setup="k_clone = k.clone()", 
        globals={
            "fused_rope": fused_rope,
            "old_pos": old_positions,
            "new_pos": new_positions,
            "k": k
        }
    )
    fused_stats = fused_timer.timeit(repeats) 

    dummy_timer = Timer(
        stmt="dummy_rope(old_pos, new_pos, k_clone)",
        setup="k_clone = k.clone()",
        globals={
            "dummy_rope": dummy_rope,
            "old_pos": old_positions,
            "new_pos": new_positions,
            "k": k
        }
    )
    dummy_stats = dummy_timer.timeit(repeats)

    fused_avg_ms = fused_stats.mean * 1000
    dummy_avg_ms = dummy_stats.mean * 1000
    speedup = dummy_avg_ms / fused_avg_ms
    
    return {
        "num_tokens": num_tokens,
        "fused_avg_ms": fused_avg_ms,
        "dummy_avg_ms": dummy_avg_ms,
        "speedup": speedup,
    }


def run_performance_tests(
    head_dim,
    max_position,
    rope_theta,
    is_neox_style,
    dtype,
    test_sizes,
    device="npu"
):

    hidden_dim = head_dim * 8  # hidden_dim = head_size * num_heads

    rope = vllm_get_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=rope_theta,
        is_neox_style=is_neox_style,
        rope_scaling=None,
        dtype=dtype,
        partial_rotary_factor=1.0,
    )
    reverse_rope = BasicReverseRope(rope, head_dim, is_neox_style)
    fused_rope = FusedRope(rope, is_neox_style)
    dummy_rope = DummyFusedRope(rope, reverse_rope, is_neox_style)

    print(f"Performance test started (Device: {device}, Dtype: {dtype})\n")
    print(f"{'Token Count':<12} | {'FusedRope (Fused Op) Avg Time (ms)':<30} | {'Dummy (Small Op) Avg Time (ms)':<30} | {'Speedup Ratio':<15} ")

    for num_tokens in test_sizes:
        result = benchmark_rope(
            fused_rope=fused_rope,
            dummy_rope=dummy_rope,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            device=device,
            dtype=dtype,
            repeats=100
        )
        print(
            f"{result['num_tokens']:<10} | "
            f"{result['fused_avg_ms']:<33.4f} | "
            f"{result['dummy_avg_ms']:<33.4f} | "
            f"{result['speedup']:<15.2f}  "
        )


if __name__ == "__main__":
    head_dim = 128
    max_position = 10000
    rope_theta = 500000.0
    is_neox_style = False
    # choices=["bfloat16", "float16", "float32"]
    dtype = torch.float32
    test_sizes = [256, 512, 1024, 2048, 4096, 8192] # num_tokens

    verify_rope(
        head_dim=head_dim,
        max_position=max_position,
        rope_theta=rope_theta,
        is_neox_style=is_neox_style,
        dtype=dtype,
        test_sizes=test_sizes
    )

    run_performance_tests(
        head_dim=head_dim,
        max_position=max_position,
        rope_theta=rope_theta,
        is_neox_style=is_neox_style,
        dtype=dtype,
        test_sizes=test_sizes
    )
