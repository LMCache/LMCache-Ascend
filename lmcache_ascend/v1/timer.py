# SPDX-License-Identifier: Apache-2.0
"""Unified request- and layer-level timer emission."""

# Standard
from typing import Optional, Union

# Third Party
from lmcache.v1.trace_utils import emit_layer_timer, emit_request_timer


def emit_timer(
    bucket: str,
    *,
    req_id: Optional[Union[str, int]],
    duration_ms: float,
    layer_id: Optional[int] = None,
    path: Optional[str] = None,
    load_mode: Optional[str] = None,
) -> None:
    """Emit a timer through the appropriate upstream timer backend."""
    kwargs = {
        "req_id": req_id,
        "duration_ms": duration_ms,
        "path": path,
        "load_mode": load_mode,
    }
    if layer_id is None:
        emit_request_timer(bucket, **kwargs)
        return

    emit_layer_timer(bucket, layer_id=layer_id, **kwargs)
