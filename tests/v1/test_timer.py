# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache_ascend.v1 import timer


def test_emit_timer_dispatches_request_timer(monkeypatch):
    captured = []
    monkeypatch.setattr(
        timer,
        "emit_request_timer",
        lambda bucket, **kwargs: captured.append((bucket, kwargs)),
    )

    timer.emit_timer(
        "lookup_total",
        req_id="req-1",
        duration_ms=2.5,
        path="hole",
        load_mode="legacy",
    )

    assert captured == [
        (
            "lookup_total",
            {
                "req_id": "req-1",
                "duration_ms": 2.5,
                "path": "hole",
                "load_mode": "legacy",
            },
        )
    ]


def test_emit_timer_dispatches_layer_timer(monkeypatch):
    captured = []
    monkeypatch.setattr(
        timer,
        "emit_layer_timer",
        lambda bucket, **kwargs: captured.append((bucket, kwargs)),
    )

    timer.emit_timer(
        "blend",
        req_id="req-2",
        layer_id=3,
        duration_ms=4.0,
        path="hole",
        load_mode="hole",
    )

    assert captured == [
        (
            "blend",
            {
                "req_id": "req-2",
                "duration_ms": 4.0,
                "path": "hole",
                "load_mode": "hole",
                "layer_id": 3,
            },
        )
    ]
