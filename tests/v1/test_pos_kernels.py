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


# --- Validation Functions ---

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
    rope: Callable, 
    reverse_rope: Callable, 
    fused_rope: Callable, 
    dummy_fused_rope: Callable, 
    head_size: int, 
    test_sizes,
    repeats: int = 10
) -> bool:

    hidden_dim = head_size * 8
    all_passed = True

    for num_tokens in test_sizes:
        
        q_errors = []
        k_errors = []
        fused_k_errors = []
        dummy_fused_k_errors = []

        print(f"\n===================== num_tokens = {num_tokens} =====================")

        for run in range(repeats):
            initial_query = torch.rand((num_tokens, hidden_dim), device="npu", dtype=rope.dtype)
            initial_key = torch.rand((num_tokens, hidden_dim), device="npu", dtype=rope.dtype)
            old_positions = torch.arange(num_tokens, device="npu")

            query_test = initial_query.clone()
            key_test = initial_key.clone()
            query_test, key_test = rope(old_positions, query_test, key_test) # Forward
            query_test, key_test = reverse_rope(old_positions, query_test, key_test) # Reverse

            current_q_err = (initial_query - query_test).abs().max().item()
            current_k_err = (initial_key - key_test).abs().max().item()
            
            unrotated_query = initial_query.clone()
            unrotated_key = initial_key.clone()
            
            new_positions = torch.arange(100, 100 + num_tokens, device="npu")
            _, target_k_new_pos = rope(new_positions, unrotated_query, unrotated_key) 
            
            k_at_old_pos = unrotated_key.clone()
            _, k_at_old_pos = rope(old_positions, unrotated_query, k_at_old_pos)
            
            # FusedRope
            k_fused_result = fused_rope(old_positions, new_positions, k_at_old_pos.clone())
            current_fused_err = (target_k_new_pos - k_fused_result).abs().max().item()
            
            # DummyFusedRope
            k_dummy_fused_result = dummy_fused_rope(old_positions, new_positions, k_at_old_pos.clone())
            current_dummy_err = (target_k_new_pos - k_dummy_fused_result).abs().max().item()

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

        # print(f"Reverse RoPE Q Error - Mean: {avg_q:.6f}, Std Dev: {std_q:.6f}")
        # print(f"Reverse RoPE K Error - Mean: {avg_k:.6f}, Std Dev: {std_k:.6f}")
        print(f"Fused Rope K Error - Mean: {avg_fused:.6f}, Std Dev: {std_fused:.6f}")
        print(f"Dummy Fused Rope K Error - Mean: {avg_dummy:.6f}, Std Dev: {std_dummy:.6f}")

        threshold = 1
        size_passed = (avg_q < threshold and avg_k < threshold
                       and avg_fused < threshold and avg_dummy < threshold)
        if not size_passed:
            print(f"Scale {num_tokens} **FAILED** the accuracy test! (Threshold: {threshold})")
            all_passed = False
        else:
            print(f"Scale {num_tokens} **PASSED** the accuracy test.")

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
    # Validate the ROPE parameters
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
    
    print("\n" + "="*85)
    print(f"**1. Accuracy Test** === Head Size: {head_size} | Neox Style: {is_neox_style} | DType: {dtype} ")
    print("="*85)

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


def benchmark_rope(
    fused_rope: FusedRope, 
    dummy_rope: DummyFusedRope, 
    num_tokens: int, 
    hidden_dim: int, 
    device: str, 
    dtype: torch.dtype, 
    repeats: int = 100
):
    old_positions = torch.arange(num_tokens, device=device)
    new_positions = torch.arange(100, 100 + num_tokens, device=device)  
    k = torch.randn((num_tokens, hidden_dim), device=device, dtype=dtype)  
    
    # warmup
    for _ in range(10):
        fused_rope(old_positions, new_positions, k.clone())
        dummy_rope(old_positions, new_positions, k.clone())
    torch.npu.synchronize()  

    fused_start_ev = torch.npu.Event(enable_timing=True)
    fused_end_ev = torch.npu.Event(enable_timing=True)
    fused_times = []

    for _ in range(repeats):
        k_clone = k
        torch.npu.synchronize()
        fused_start_ev.record()
        fused_rope(old_positions, new_positions, k_clone)
        fused_end_ev.record()
        torch.npu.synchronize()
        elapsed = fused_start_ev.elapsed_time(fused_end_ev)
        fused_times.append(elapsed)

    dummy_start_ev = torch.npu.Event(enable_timing=True)
    dummy_end_ev = torch.npu.Event(enable_timing=True)
    dummy_times = []

    for _ in range(repeats):
        k_clone = k
        torch.npu.synchronize()
        dummy_start_ev.record()
        dummy_rope(old_positions, new_positions, k_clone)
        dummy_end_ev.record()
        torch.npu.synchronize()
        elapsed = dummy_start_ev.elapsed_time(dummy_end_ev)
        dummy_times.append(elapsed)
    
    fused_avg_ms = sum(fused_times) / repeats
    dummy_avg_ms = sum(dummy_times) / repeats
    speedup = dummy_avg_ms / fused_avg_ms if fused_avg_ms > 0 else 0
    
    return {
        "num_tokens": num_tokens,
        "fused_avg_ms": fused_avg_ms,
        "dummy_avg_ms": dummy_avg_ms,
        "speedup": speedup
    }


def run_performance_tests(
    head_dim: int,
    max_position: int,
    rope_theta: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    test_sizes,
    device: str = "npu"
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

    print("\n" + "="*95)
    print(f"**2. Performance Test** === Device: {device} | DType: {dtype} | Head Dim: {head_dim} ")
    print("="*95)
    print(f"{'Token Count':<12} | {'FusedRope (Fused Op) Avg Time (ms)':<32} | {'Dummy (Small Op) Avg Time (ms)':<32} | {'Speedup Ratio':<15}")
    print("-" * 95)

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
            f"{result['num_tokens']:<12} | "
            f"{result['fused_avg_ms']:<32.4f} | "
            f"{result['dummy_avg_ms']:<32.4f} | "
            f"{result['speedup']:<15.2f}"
        )


if __name__ == "__main__":
    head_size = 64
    max_position = 33000
    rope_theta = 500000.0
    is_neox_style = True
    # choices=["bfloat16", "float16", "float32"]
    dtype = torch.bfloat16
    test_sizes = [512, 1024, 4096] # num_tokens

    verify_rope(
        head_dim=head_size, 
        max_position=max_position, 
        rope_theta=rope_theta, 
        is_neox_style=is_neox_style, 
        dtype=dtype, 
        test_sizes=test_sizes
    )

    run_performance_tests(
        head_dim=head_size, 
        max_position=max_position, 
        rope_theta=rope_theta, 
        is_neox_style=is_neox_style, 
        dtype=dtype, 
        test_sizes=test_sizes
    )