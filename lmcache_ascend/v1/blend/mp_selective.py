# SPDX-License-Identifier: Apache-2.0
"""Qwen3 selective prefill over MP-scattered KV, independent of legacy LMCache.

Compute all positions through the check layer's QKV projection. Select cached
positions by squared K or V deviation, retaining every uncached position. Later
layers compute only those queries against the full, causally masked KV.
"""

# Standard
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
import math

# Third Party
import torch

TensorOp = Callable[[torch.Tensor], torch.Tensor]
KVTransfer = Callable[[torch.Tensor, Sequence[torch.Tensor], torch.Tensor, bool], None]
AttentionOp = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor
]


@dataclass(frozen=True)
class SelectiveConfig:
    """One zero-based check layer and fraction of cached tokens to refresh."""

    ratio: float = 0.15
    check_layer: int = 1
    selection_metric: Literal["k", "v"] = "k"
    refresh_check_layer: bool = False

    def __post_init__(self) -> None:
        if self.selection_metric not in ("k", "v"):
            raise ValueError("selection_metric must be k or v")
        if not isinstance(self.refresh_check_layer, bool):
            raise ValueError("refresh_check_layer must be a bool")
        if not math.isfinite(self.ratio) or not 0 < self.ratio <= 1:
            raise ValueError("selective ratio must be finite and in (0, 1]")
        if not isinstance(self.check_layer, int) or self.check_layer < 1:
            raise ValueError("check_layer must be an integer >= 1")


def identity(tensor: torch.Tensor) -> torch.Tensor:
    """Single-rank reduction/broadcast used by the standalone executor."""
    return tensor


def select_positions(
    fresh_k: torch.Tensor,
    cached_k: torch.Tensor,
    hit_positions: torch.Tensor,
    ratio: float,
    reduce_scores: TensorOp = identity,
    broadcast: TensorOp = identity,
) -> torch.Tensor:
    """Return sorted mandatory misses plus top-deviation cache hits.

    The historical parameter names refer to K; callers may supply V instead.

    Keys have shape [tokens, flattened local KV heads]. Scores are summed over
    TP ranks, and selected indices are broadcast from rank zero to avoid tie
    ordering differences. All ranks must call with the same hit positions.
    The cached tensor must contain valid keys at hit_positions only.
    """
    if hit_positions.numel() == 0:
        # All TP ranks agree on the empty hit set before entering the executor.
        return torch.arange(fresh_k.shape[0], device=fresh_k.device)
    scores = torch.sum(
        (
            fresh_k[hit_positions].to(torch.float32)
            - cached_k[hit_positions].to(torch.float32)
        )
        ** 2,
        dim=[1],
    )
    scores = reduce_scores(scores)
    count = min(hit_positions.numel(), max(1, int(hit_positions.numel() * ratio)))
    chosen = torch.topk(scores, k=count).indices
    chosen = broadcast(chosen)
    keep = torch.ones(fresh_k.shape[0], dtype=torch.bool, device=fresh_k.device)
    keep[hit_positions] = False
    keep[hit_positions[chosen]] = True
    return torch.where(keep)[0]


def npu_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run Ascend fused GQA with absolute-position causal masking.

    Inputs are [tokens, local heads, head_dim]. Unlike a triangular mask over
    packed query indices, this mask permits exactly keys <= each original
    query position. Returns the same shape as q.
    """
    # Third Party
    import torch_npu

    key_positions = torch.arange(k.shape[0], device=q.device)
    blocked = (key_positions[None, :] > positions[:, None])[None, None]
    result, _ = torch_npu.npu_fused_infer_attention_score(
        query=q.contiguous().unsqueeze(0),
        key=k.contiguous().unsqueeze(0),
        value=v.contiguous().unsqueeze(0),
        atten_mask=blocked,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=scale,
        input_layout="BSND",
        sparse_mode=0,
        softmax_lse_flag=False,
    )
    return result.squeeze(0)


def npu_kv_transfer(
    buffer: torch.Tensor,
    caches: Sequence[torch.Tensor],
    slots: torch.Tensor,
    from_paged: bool,
) -> None:
    """Use the legacy connector's fused NHD paged/contiguous KV transfer.

    Buffer is [2, tokens, flattened KV heads]; -1 slots skip absent pages.
    Gaps in a loaded buffer must be overwritten before attention or writeback.
    """
    # First Party
    from lmcache_ascend import c_ops

    # SEPARATE_KV=2; both K/V and the token axis are contiguous inside a block.
    c_ops.single_layer_kv_transfer(
        buffer, tuple(caches), slots, from_paged, 2, False, False
    )


@torch.inference_mode()
def selective_prefill(
    model: Any,
    caches: Mapping[str, Sequence[torch.Tensor]],
    token_ids: Sequence[int],
    block_ids: Sequence[int],
    hit_ranges: Sequence[tuple[int, int]],
    config: SelectiveConfig,
    attention: AttentionOp = npu_causal_attention,
    reduce_scores: TensorOp = identity,
    broadcast: TensorOp = identity,
    transfer: KVTransfer = npu_kv_transfer,
    computed_prefix_tokens: int = 0,
) -> dict[str, Any]:
    """Refresh Qwen3 paged KV through len(token_ids), without producing logits.

    Supports eager, dense Qwen3, a single PP/DP group and unquantized NHD KV
    [blocks, block_size, local_kv_heads, head_dim]. The native transfer supports
    FP16/BF16; an injected reference transfer can support FP32 numerical tests.
    All layers share one layout and one reusable contiguous buffer. Blocks must be
    exclusively allocated to this request. Hits must have finished scattering.
    Empty hit_ranges computes the entire suffix after computed_prefix_tokens
    without reading unready KV, for confirmed load-failure fallback.
    The caller must apply TP-wide scatter consensus before entering this
    function; projection layers and score reduction perform TP collectives.
    computed_prefix_tokens is the scheduler-confirmed ready prefix, not a lookup
    claim. Its KV is read unchanged at every layer and never recomputed/written.

    Returns counts and selected absolute positions for auditable experiments.
    Unsupported layers/layouts raise ValueError; execution errors propagate,
    since silently continuing could expose partially updated KV.
    """
    layers = model.model.layers
    length = len(token_ids)
    if not 0 < config.check_layer < len(layers):
        raise ValueError("check_layer must be a nonzero layer inside the model")
    if length == 0:
        raise ValueError("selective prefill requires nonempty input")
    if not 0 <= computed_prefix_tokens < length:
        raise ValueError("computed prefix must be inside the input")
    previous_end = computed_prefix_tokens
    for start, end in hit_ranges:
        if not previous_end <= start < end <= length:
            raise ValueError("hit ranges must be sorted, disjoint and inside input")
        previous_end = end

    # Validate every layer before modifying any KV. Packed/NZ/HND layouts need
    # a different adapter; reshaping them as NHD can silently corrupt caches.
    layer_caches = []
    for layer in layers:
        attn = layer.self_attn
        if attn.attn.impl.use_bnsd_kv_cache:
            raise ValueError("selective prefill currently requires NHD KV")
        pair = caches[attn.attn.layer_name]
        if len(pair) != 2:
            raise ValueError("expected separate K and V cache tensors")
        ck, cv = pair
        if (
            ck.ndim != 4
            or ck.shape != cv.shape
            or ck.shape[2:] != (attn.num_kv_heads, attn.head_dim)
            or ck.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError("unsupported Qwen3 KV shape or dtype")
        if layer_caches:
            first = layer_caches[0][0]
            if (
                ck.shape[1:] != first.shape[1:]
                or ck.dtype != first.dtype
                or ck.device != first.device
            ):
                raise ValueError("shared KV buffer requires uniform layer layouts")
        if transfer is npu_kv_transfer and ck.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError("native KV transfer requires FP16 or BF16")
        if len(block_ids) * ck.shape[1] < length:
            raise ValueError("insufficient allocated KV blocks")
        if min(block_ids) < 0 or max(block_ids) >= ck.shape[0]:
            raise ValueError("KV block id out of bounds")
        layer_caches.append((ck, cv))

    device = layer_caches[0][0].device
    positions = torch.arange(length, device=device)
    hits = torch.tensor(
        [p for start, end in hit_ranges for p in range(start, end)],
        device=device,
        dtype=torch.long,
    )
    block_table = torch.tensor(block_ids, device=device, dtype=torch.long)
    block_size = layer_caches[0][0].shape[1]
    slots = block_table[positions // block_size] * block_size + positions % block_size
    hit_slots = torch.full_like(slots, -1)
    hit_slots[hits] = slots[hits]
    prefix_slots = torch.full_like(slots, -1)
    prefix_slots[:computed_prefix_tokens] = slots[:computed_prefix_tokens]
    hit_slots[:computed_prefix_tokens] = slots[:computed_prefix_tokens]
    write_slots = slots.clone()
    write_slots[:computed_prefix_tokens] = -1
    positions = positions[computed_prefix_tokens:]
    # Like the legacy layerwise connector, attention and writeback share one
    # contiguous buffer. Same-stream ordering makes reuse across layers safe.
    kv_buffer = torch.empty(
        (2, length, layers[0].self_attn.kv_size),
        dtype=layer_caches[0][0].dtype,
        device=device,
    )
    full_k, full_v = kv_buffer.unbind(0)
    hidden = model.embed_input_ids(
        torch.tensor(token_ids[computed_prefix_tokens:], device=device)
    )
    residual = None
    selected = positions
    layer_query_counts = []
    layer_qkv_counts = []
    for index, (layer, (ck, cv)) in enumerate(zip(layers, layer_caches, strict=True)):
        attn = layer.self_attn
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        layer_qkv_counts.append(hidden.shape[0])
        qkv, _ = attn.qkv_proj(hidden)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = attn.q_norm(q.reshape(-1, attn.num_heads, attn.head_dim)).reshape(q.shape)
        k = attn.k_norm(k.reshape(-1, attn.num_kv_heads, attn.head_dim)).reshape(
            k.shape
        )
        q, k = attn.rotary_emb(positions, q, k)
        # Never read gaps from paged memory: they may contain NaNs or another
        # request's data. Mandatory queries fill every skipped slot below.
        if index >= config.check_layer:
            transfer(kv_buffer, (ck, cv), hit_slots, True)
        elif computed_prefix_tokens:
            transfer(kv_buffer, (ck, cv), prefix_slots, True)
        if index == config.check_layer:
            fresh_selection = k if config.selection_metric == "k" else v
            cached_selection = full_k if config.selection_metric == "k" else full_v
            local_selected = select_positions(
                fresh_selection,
                cached_selection[computed_prefix_tokens:],
                hits - computed_prefix_tokens,
                config.ratio,
                reduce_scores,
                broadcast,
            )
            # The author implementation keeps all freshly projected KV at
            # the check layer, then sparsifies KV updates in later layers.
            if config.refresh_check_layer:
                full_k[computed_prefix_tokens:].copy_(k)
                full_v[computed_prefix_tokens:].copy_(v)
            selected = local_selected + computed_prefix_tokens
            positions = selected
            q, k, v, residual = (
                q[local_selected],
                k[local_selected],
                v[local_selected],
                residual[local_selected],
            )
        if index < config.check_layer:
            full_k[computed_prefix_tokens:].copy_(k)
            full_v[computed_prefix_tokens:].copy_(v)
        else:
            full_k.index_copy_(0, positions, k)
            full_v.index_copy_(0, positions, v)
        # Flush the completed contiguous buffer using the same fused transfer
        # as the old connector; retained hits are copied without modification.
        transfer(kv_buffer, (ck, cv), write_slots, False)
        attended = attention(
            q.reshape(-1, attn.num_heads, attn.head_dim),
            full_k.reshape(-1, attn.num_kv_heads, attn.head_dim),
            full_v.reshape(-1, attn.num_kv_heads, attn.head_dim),
            positions,
            attn.scaling,
        )
        hidden, _ = attn.o_proj(attended.reshape(-1, attn.q_size))
        hidden, residual = layer.post_attention_layernorm(hidden, residual)
        hidden = layer.mlp(hidden)
        layer_query_counts.append(positions.numel())
    # This PoC synchronizes once for observability; logs must distinguish actual
    # retained cache tokens from the original lookup's apparent hit coverage.
    selected_list = selected.cpu().tolist()
    mandatory = length - computed_prefix_tokens - hits.numel()
    return {
        "computed_end": length,
        "check_layer": config.check_layer,
        "recompute_ratio": config.ratio,
        "selection_metric": config.selection_metric,
        "refresh_check_layer": config.refresh_check_layer,
        "matched_tokens": hits.numel(),
        "mandatory_tokens": mandatory,
        "selected_hit_tokens": len(selected_list) - mandatory,
        "retained_hit_tokens": length - computed_prefix_tokens - len(selected_list),
        "retained_prefix_tokens": computed_prefix_tokens,
        "selected_positions": selected_list,
        "layer_query_counts": layer_query_counts,
        "layer_qkv_counts": layer_qkv_counts,
    }
