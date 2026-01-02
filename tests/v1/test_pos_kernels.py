# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Callable, List

# Third Party
from vllm.model_executor.layers.rotary_embedding import get_rope as vllm_get_rope
import numpy as np
import pytest
import torch

# First Party
from lmcache_ascend.v1.blend.positional_encoding import BasicReverseRope, FusedRope

# ==============================================================================
# 1. Dummy Rope Implementation (for comparison)
# ==============================================================================


class DummyFusedRope:
    """
    Directly use the fused kernel to rotate K cache from
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


# ==============================================================================
# 2. ACCURACY VALIDATION FUNCTIONS
# ==============================================================================


def validate_correctness(
    rope: Callable,
    reverse_rope: BasicReverseRope,
    fused_rope: FusedRope,
    dummy_fused_rope: DummyFusedRope,
    head_size: int,
    test_sizes: List[int],
    repeats: int = 10,
) -> bool:
    """
    Validate correctness of fused RoPE by checking error against thresholds.
    """

    hidden_dim = head_size * 8
    all_passed = True
    threshold = 0.1

    for num_tokens in test_sizes:
        q_errors = []
        k_errors = []
        fused_k_errors = []
        dummy_fused_k_errors = []

        print(f"\n{'=' * 20} num_tokens = {num_tokens} {'=' * 20}")

        for _ in range(repeats):
            initial_query = torch.rand(
                (num_tokens, hidden_dim), device="npu", dtype=rope.dtype
            )
            initial_key = torch.rand(
                (num_tokens, hidden_dim), device="npu", dtype=rope.dtype
            )
            old_positions = torch.arange(num_tokens, device="npu")

            query_test = initial_query.clone()
            key_test = initial_key.clone()
            query_test, key_test = rope(old_positions, query_test, key_test)  # Forward
            query_test, key_test = reverse_rope(
                old_positions, query_test, key_test
            )  # Reverse

            current_q_err = (initial_query - query_test).abs().max().item()
            current_k_err = (initial_key - key_test).abs().max().item()

            unrotated_query = initial_query.clone()
            unrotated_key = initial_key.clone()

            new_positions = torch.arange(100, 100 + num_tokens, device="npu")
            _, target_k_new_pos = rope(new_positions, unrotated_query, unrotated_key)

            k_at_old_pos = unrotated_key.clone()
            _, k_at_old_pos = rope(old_positions, unrotated_query, k_at_old_pos)

            # FusedRope
            k_fused_result = fused_rope(
                old_positions, new_positions, k_at_old_pos.clone()
            )
            current_fused_err = (target_k_new_pos - k_fused_result).abs().max().item()

            # DummyFusedRope
            k_dummy_fused_result = dummy_fused_rope(
                old_positions, new_positions, k_at_old_pos.clone()
            )
            current_dummy_err = (
                (target_k_new_pos - k_dummy_fused_result).abs().max().item()
            )

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

        print(f"Reverse RoPE Q Error - Mean: {avg_q:.6f}, Std Dev: {std_q:.6f}")
        print(f"Reverse RoPE K Error - Mean: {avg_k:.6f}, Std Dev: {std_k:.6f}")
        print(f"Fused Rope K Error - Mean: {avg_fused:.6f}, Std Dev: {std_fused:.6f}")
        print(
            f"Dummy Fused Rope K Error - Mean: {avg_dummy:.6f}, "
            f"Std Dev: {std_dummy:.6f}"
        )

        size_passed = (
            avg_q < threshold
            and avg_k < threshold
            and avg_fused < threshold
            and avg_dummy < threshold
        )

        if not size_passed:
            print(
                f"Scale {num_tokens} **FAILED** the accuracy test!"
                f" (Threshold: {threshold})"
            )
            all_passed = False
        else:
            print(f"Scale {num_tokens} **PASSED** the accuracy test.")

    return all_passed


# ==============================================================================
# 3. PERFORMANCE BENCHMARK FUNCTIONS
# ==============================================================================


def benchmark_rope(
    fused_rope: FusedRope,
    dummy_rope: DummyFusedRope,
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    repeats: int = 100,
):
    """
    Benchmark performance of fused RoPE vs dummy fused RoPE.
    """

    old_positions = torch.arange(num_tokens, device="npu")
    new_positions = torch.arange(100, 100 + num_tokens, device="npu")
    k = torch.randn((num_tokens, hidden_dim), device="npu", dtype=dtype)

    # warmup
    for _ in range(10):
        fused_rope(old_positions, new_positions, k.clone())
        dummy_rope(old_positions, new_positions, k.clone())
    torch.npu.synchronize()

    def measure(op):
        times = []
        start_ev, end_ev = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        for _ in range(repeats):
            torch.npu.synchronize()
            start_ev.record()
            op()
            end_ev.record()
            torch.npu.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
        return sum(times) / repeats

    fused_avg_ms = measure(lambda: fused_rope(old_positions, new_positions, k.clone()))
    dummy_avg_ms = measure(lambda: dummy_rope(old_positions, new_positions, k.clone()))
    speedup = dummy_avg_ms / fused_avg_ms if fused_avg_ms > 0 else 0

    return {
        "num_tokens": num_tokens,
        "fused_avg_ms": fused_avg_ms,
        "dummy_avg_ms": dummy_avg_ms,
        "speedup": speedup,
    }


def validate_performance(
    fused_rope: FusedRope,
    dummy_rope: DummyFusedRope,
    head_size: int,
    dtype: torch.dtype,
    test_sizes: List[int],
    repeats: int = 10,
):
    hidden_dim = head_size * 8  # hidden_dim = head_size * num_heads

    print("\n" + "=" * 95)
    print(
        f"**2. Performance Test** === Device: npu | DType: {dtype} |"
        f" Head Dim: {head_size} "
    )
    print("=" * 95)
    print(
        f"{'Token Count':<12} | {'FusedRope (Fused Op) Avg Time (ms)':<32} |"
        f" {'Dummy (Small Op) Avg Time (ms)':<32} | {'Speedup Ratio':<15}"
    )
    print("-" * 95)

    for num_tokens in test_sizes:
        result = benchmark_rope(
            fused_rope=fused_rope,
            dummy_rope=dummy_rope,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            dtype=dtype,
            repeats=repeats,
        )
        print(
            f"{result['num_tokens']:<12} | "
            f"{result['fused_avg_ms']:<32.4f} | "
            f"{result['dummy_avg_ms']:<32.4f} | "
            f"{result['speedup']:<15.2f}"
        )


# ==============================================================================
# 4. Pytest Fixtures and Test Cases
# ==============================================================================


@pytest.fixture
def rope_modules(head_size, max_position, rope_theta, is_neox_style, dtype):
    """
    Fixture to initialize the base RoPE, ReverseRoPE, FusedRope,
    and DummyFusedRope modules.
    """

    base_rope = vllm_get_rope(
        head_size,
        rotary_dim=head_size,
        max_position=max_position,
        base=rope_theta,
        is_neox_style=is_neox_style,
        rope_scaling=None,
        dtype=dtype,
        partial_rotary_factor=1.0,
    )

    base_rope.cos_sin_cache = base_rope.cos_sin_cache.to("npu")

    reverse_rope = BasicReverseRope(base_rope, head_size, is_neox_style)
    fused_rope = FusedRope(base_rope, is_neox_style)
    dummy_rope = DummyFusedRope(base_rope, reverse_rope, is_neox_style)

    return base_rope, reverse_rope, fused_rope, dummy_rope


@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("max_position", [40960])
@pytest.mark.parametrize("rope_theta", [500000.0])
@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("test_sizes", [[512, 1024, 4096]])
def test_rope_correctness(rope_modules, head_size, test_sizes):
    rope, reverse_rope, fused_rope, dummy_rope = rope_modules

    passed = validate_correctness(
        rope, reverse_rope, fused_rope, dummy_rope, head_size, test_sizes
    )

    assert passed, f"Accuracy test failed for head_size={head_size}, dtype={rope.dtype}"


@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("max_position", [40960])
@pytest.mark.parametrize("rope_theta", [500000.0])
@pytest.mark.parametrize("is_neox_style", [True])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("test_sizes", [[512, 1024, 4096]])
def test_rope_performance(rope_modules, head_size, dtype, test_sizes):
    rope, reverse_rope, fused_rope, dummy_rope = rope_modules

    validate_performance(fused_rope, dummy_rope, head_size, dtype, test_sizes)
