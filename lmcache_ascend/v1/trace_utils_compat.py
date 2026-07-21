# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import json
import os
import sys
import time
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - optional in some test contexts
    torch = None


def _env_enabled(name: str) -> bool:
    raw = str(os.environ.get(name, "0")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def trace_flow_enabled() -> bool:
    return _env_enabled("LMCACHE_TRACE_FLOW")


def layer_timers_enabled() -> bool:
    return _env_enabled("LMCACHE_ENABLE_LAYER_TIMERS")


@lru_cache(maxsize=None)
def _parse_layer_timer_layers(raw: str) -> set[int] | None:
    normalized = raw.strip().lower()
    if normalized in {"", "default"}:
        normalized = "0-4"
    if normalized in {"all", "*"}:
        return None
    if normalized in {"none", "off", "false"}:
        return set()

    layers: set[int] = set()
    for part in normalized.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                start, end = end, start
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    return layers


def layer_timer_enabled(layer_id: int | None) -> bool:
    if not layer_timers_enabled():
        return False
    if layer_id is None:
        return True

    raw = str(os.environ.get("LMCACHE_LAYER_TIMER_LAYERS", "0-4"))
    parsed = _parse_layer_timer_layers(raw)
    if parsed is None:
        return True
    return int(layer_id) in parsed


def _truncate_list(values: list[Any], max_items: int | None) -> list[Any]:
    if max_items is None or len(values) <= max_items:
        return values
    return values[:max_items] + ["..."]


def tensor_to_list(
    value: Any,
    *,
    dtype=None,
    max_items: int | None = 128,
):
    if value is None:
        return None

    if hasattr(value, "detach"):
        tensor = value.detach().to(device="cpu")
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        value = tensor.tolist()
    elif isinstance(value, tuple):
        value = list(value)
    elif isinstance(value, range):
        value = list(value)

    if isinstance(value, list):
        return _truncate_list(value, max_items=max_items)
    return value


def mask_to_string(mask: Any, *, max_items: int = 256) -> str:
    values = tensor_to_list(mask, max_items=max_items)
    if values is None:
        return "None"
    if not isinstance(values, list):
        return str(values)

    chars: list[str] = []
    for item in values:
        if item == "...":
            chars.append("...")
        else:
            chars.append("1" if bool(item) else "0")
    return "".join(chars)


def summarize_ranges(
    ranges: Iterable[Any] | None,
    *,
    max_items: int = 16,
):
    if ranges is None:
        return None

    values = list(ranges)
    summary: list[Any] = []
    for item in values[:max_items]:
        if isinstance(item, tuple):
            summary.append(list(item))
        else:
            summary.append(item)
    if len(values) > max_items:
        summary.append("...")
    return summary


def summarize_slot_mapping(slot_mapping: Any, *, max_items: int = 64):
    return tensor_to_list(slot_mapping, max_items=max_items)


def summarize_key(key: Any, *, max_chars: int = 240) -> str:
    text = str(key)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def trace_request_selected(
    req_id: str | int | None,
    *,
    env_name: str = "LMCACHE_TRACE_REQ_IDS",
) -> bool:
    if not trace_flow_enabled():
        return False
    raw = str(os.environ.get(env_name, "")).strip()
    if raw.lower() in {"", "none", "off", "false"}:
        return False
    if raw.lower() in {"all", "*"}:
        return True
    if req_id is None:
        return False
    return str(req_id) in {item.strip() for item in raw.split(",") if item.strip()}


def trace_first_decode_only() -> bool:
    return _env_enabled("LMCACHE_TRACE_FIRST_DECODE_ONLY") or (
        "LMCACHE_TRACE_FIRST_DECODE_ONLY" not in os.environ
    )


def trace_logit_topk() -> int:
    raw = str(os.environ.get("LMCACHE_TRACE_LOGIT_TOPK", "8")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 8
    return max(value, 1)


def summarize_topk_logits(value: Any, *, topk: int | None = None):
    if value is None or torch is None:
        return None
    if not hasattr(value, "detach"):
        return None
    tensor = value.detach()
    if tensor.dim() != 1:
        tensor = tensor.reshape(-1)
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "top_token_ids": [], "top_logits": []}
    k = min(topk or trace_logit_topk(), int(tensor.numel()))
    vals, idxs = torch.topk(tensor.to(dtype=torch.float32), k=k)
    vals = vals.to(device="cpu")
    idxs = idxs.to(device="cpu")
    output = {
        "shape": list(tensor.shape),
        "top_token_ids": idxs.tolist(),
        "top_logits": [float(v) for v in vals.tolist()],
        "argmax_token_id": int(idxs[0].item()),
        "argmax_logit": float(vals[0].item()),
    }
    if k >= 2:
        output["top_gap_01"] = float(vals[0].item() - vals[1].item())
    return output


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if torch is not None and hasattr(value, "detach"):
        return tensor_to_list(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return str(value)


def trace_flow(component: str, event: str, **fields: Any) -> None:
    if not trace_flow_enabled():
        return
    payload = {
        "component": component,
        "event": event,
    }
    for key, value in fields.items():
        payload[key] = _normalize(value)
    print(
        "[LMCACHE_TRACE_FLOW] " + json.dumps(payload, ensure_ascii=True, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def emit_layer_timer(
    bucket: str,
    *,
    req_id: str | int | None,
    layer_id: int | None,
    duration_ms: float,
    path: str | None = None,
    load_mode: str | None = None,
) -> None:
    if req_id is None or layer_id is None:
        return
    if not layer_timer_enabled(layer_id):
        return

    payload: dict[str, Any] = {
        "bucket": str(bucket),
        "duration_ms": float(duration_ms),
        "layer_id": int(layer_id),
        "req_id": str(req_id),
    }
    if path is not None:
        payload["path"] = str(path)
    if load_mode is not None:
        payload["load_mode"] = str(load_mode)

    print(
        "[LMCACHE_LAYER_TIMER] " + json.dumps(payload, ensure_ascii=True, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def advance_layerwise_storers_with_timing(
    layerwise_storers: Iterable[tuple[str, Any, str | None]],
    *,
    layer_id: int,
    timer_path: str,
) -> None:
    for req_id, layerwise_storer, load_mode in layerwise_storers:
        save_start = time.perf_counter()
        next(layerwise_storer)
        emit_layer_timer(
            "save",
            req_id=req_id,
            layer_id=layer_id,
            duration_ms=(time.perf_counter() - save_start) * 1000.0,
            path=timer_path,
            load_mode=load_mode,
        )


# --- Temporary verification stubs (missing instrumentation symbols) ---
# These were referenced by feature code but never committed. Stubbed as
# no-ops to make the rebased branch importable/runnable for verification.
# To be removed (with all tracing callsites) before the PR per maintainer guidance.
def summarize_kv_tensor_stats(*args, **kwargs):
    return None


def summarize_prefix_kv_tensor_stats(*args, **kwargs):
    return None


def emit_request_timer(*args, **kwargs):
    return None


def trace_layer_enabled(*args, **kwargs):
    return False


def trace_probe_positions(*args, **kwargs):
    return []


def trace_compare_prefix_len(*args, **kwargs):
    return None
