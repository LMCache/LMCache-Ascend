# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from typing import Any, Iterable, Optional
import json

DEFAULT_LAYER_TIMER_LAYERS = [0, 1, 2, 3, 4]
_LAYER_TIMER_BUCKETS = (
    "wait_reuse",
    "reuse_plan_lookup",
    "reuse_task_next",
    "reuse_storage_wait",
    "reuse_consumer_prime",
    "reuse_consumer_send",
    "reuse_consumer_finalize",
    "reuse_ref_count_down",
    "reuse_gap_mask",
    "reuse_buffer_alloc",
    "reuse_gap_assert",
    "reuse_sync",
    "reuse_flush",
    "reuse_flush_reuse_wait",
    "reuse_send_other",
    "reuse_rope",
    "reuse_zero_gap",
    "reuse_copy_enqueue",
    "topk_l1",
    "blend_kv_fetch",
    "blend_positions_init",
    "blend_qk_post",
    "blend_gap_positions",
    "blend_hit_positions",
    "blend_topk_diff",
    "blend_topk_prepare",
    "blend_topk_section_mask",
    "blend_topk_kernel",
    "blend_topk_gather",
    "blend_topk_sort",
    "blend_topk_select",
    "blend_topk_trace",
    "blend_recompute_sort",
    "blend_recompute_gather",
    "blend_attn_metadata",
    "blend_gap_materialize",
    "blend_gap_scatter",
    "blend_qkv_view",
    "blend_attention",
    "blend",
    "save",
)
_LAYER_TIMER_PREFIX = "[LMCACHE_LAYER_TIMER]"


def build_empty_layer_timer_metrics(
    layers: Iterable[int] | None = None,
) -> dict[str, Any]:
    layer_list = (
        DEFAULT_LAYER_TIMER_LAYERS.copy()
        if layers is None
        else [int(layer) for layer in layers]
    )
    metrics: dict[str, Any] = {
        "lmcache_timer_layers": layer_list,
    }
    for bucket in _LAYER_TIMER_BUCKETS:
        metrics[f"lmcache_timer_{bucket}_ms"] = [0.0] * len(layer_list)
        metrics[f"lmcache_timer_{bucket}_total_ms"] = 0.0
    return metrics


def parse_layer_timer_line(line: str) -> Optional[dict[str, Any]]:
    marker_idx = line.find(_LAYER_TIMER_PREFIX)
    if marker_idx < 0:
        return None

    payload_text = line[marker_idx + len(_LAYER_TIMER_PREFIX) :].strip()
    if not payload_text:
        return None

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    bucket = payload.get("bucket")
    req_id = payload.get("req_id")
    layer_id = payload.get("layer_id")
    duration_ms = payload.get("duration_ms")
    if bucket not in _LAYER_TIMER_BUCKETS:
        return None
    if req_id is None or layer_id is None or duration_ms is None:
        return None

    try:
        normalized = {
            "bucket": str(bucket),
            "duration_ms": float(duration_ms),
            "layer_id": int(layer_id),
            "req_id": str(req_id),
        }
    except (TypeError, ValueError):
        return None

    load_mode = payload.get("load_mode")
    if load_mode is not None:
        normalized["load_mode"] = str(load_mode)
    path = payload.get("path")
    if path is not None:
        normalized["path"] = str(path)
    return normalized


def aggregate_layer_timer_events(
    lines: Iterable[str],
    *,
    req_ids: Iterable[str] | None = None,
    layers: Iterable[int] | None = None,
) -> dict[str, dict[str, Any]]:
    layer_list = (
        DEFAULT_LAYER_TIMER_LAYERS.copy()
        if layers is None
        else [int(layer) for layer in layers]
    )
    layer_index = {layer_id: idx for idx, layer_id in enumerate(layer_list)}
    req_id_filter = None
    if req_ids is not None:
        req_id_filter = {str(req_id) for req_id in req_ids}

    aggregated: dict[str, dict[str, Any]] = {}
    for line in lines:
        payload = parse_layer_timer_line(line)
        if payload is None:
            continue

        req_id = str(payload["req_id"])
        if req_id_filter is not None and req_id not in req_id_filter:
            continue

        layer_id = int(payload["layer_id"])
        idx = layer_index.get(layer_id)
        if idx is None:
            continue

        bucket = str(payload["bucket"])
        row = aggregated.setdefault(req_id, build_empty_layer_timer_metrics(layer_list))
        values = row[f"lmcache_timer_{bucket}_ms"]
        values[idx] += float(payload["duration_ms"])

    if req_id_filter is not None:
        for req_id in req_id_filter:
            aggregated.setdefault(req_id, build_empty_layer_timer_metrics(layer_list))

    for row in aggregated.values():
        for bucket in _LAYER_TIMER_BUCKETS:
            values = row[f"lmcache_timer_{bucket}_ms"]
            row[f"lmcache_timer_{bucket}_total_ms"] = float(sum(values))
    return aggregated


def aggregate_layer_timer_file(
    log_path: str,
    *,
    req_ids: Iterable[str] | None = None,
    layers: Iterable[int] | None = None,
) -> dict[str, dict[str, Any]]:
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        return aggregate_layer_timer_events(handle, req_ids=req_ids, layers=layers)
