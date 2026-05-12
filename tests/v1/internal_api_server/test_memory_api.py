# SPDX-License-Identifier: Apache-2.0
"""Unit tests for /memory/prefetch and /memory/evict endpoints."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

starlette_testclient = pytest.importorskip(
    "starlette.testclient", reason="starlette not installed"
)
fastapi_mod = pytest.importorskip("fastapi", reason="fastapi not installed")
TestClient = starlette_testclient.TestClient
FastAPI = fastapi_mod.FastAPI


def _make_app(engine):
    # First Party
    from lmcache_ascend.v1.internal_api_server.memory.memory_api import (
        router,
    )

    config = SimpleNamespace(internal_api_server_port_start=9000)
    adapter = SimpleNamespace(
        lmcache_engine=engine,
        config=config,
    )
    app = FastAPI()
    app.state.lmcache_adapter = adapter
    app.include_router(router)
    return app


def _make_engine(*, backend_state=None):
    metadata = SimpleNamespace(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        kv_dtype=torch.float16,
    )
    config = SimpleNamespace(chunk_size=256)

    engine = SimpleNamespace(
        save_only_first_rank=False,
        metadata=metadata,
        config=config,
        retrieve_locations=None,
    )

    sm = MagicMock()
    sm.loop = MagicMock()
    sm.async_lookup_and_prefetch = MagicMock()

    if backend_state is not None:
        sm.batched_remove = MagicMock(
            side_effect=lambda keys, locations=None: sum(
                1 for k in keys if backend_state.pop(k.chunk_hash, None) is not None
            )
        )
    else:
        sm.batched_remove = MagicMock(return_value=0)

    engine.storage_manager = sm
    return engine


class TestPrefetchEndpoint:
    @patch(
        "lmcache_ascend.v1.internal_api_server.memory.memory_api."
        "_forward_to_all_workers"
    )
    def test_scheduler_forwards_to_workers(self, mock_forward):
        mock_forward.return_value = MagicMock(status_code=200)
        app = _make_app(engine=None)
        client = TestClient(app)
        client.post(
            "/memory/prefetch",
            json={"chunk_hashes": ["abc"], "lookup_id": "req-001"},
        )
        mock_forward.assert_called_once()

    def test_worker_prefetches_correct_keys(self):
        engine = _make_engine()
        app = _make_app(engine=engine)
        client = TestClient(app)

        with patch("asyncio.run_coroutine_threadsafe"):
            resp = client.post(
                "/memory/prefetch",
                json={
                    "chunk_hashes": ["abc123", "def456"],
                    "lookup_id": "req-001",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "prefetch_started"

        sm = engine.storage_manager
        sm.async_lookup_and_prefetch.assert_called_once()
        kwargs = sm.async_lookup_and_prefetch.call_args.kwargs
        assert kwargs["lookup_id"] == "req-001"
        assert kwargs["pin"] is False

        keys = kwargs["keys"]
        assert len(keys) == 2
        assert keys[0].chunk_hash == int("abc123", 16)
        assert keys[0].model_name == "test-model"
        assert keys[0].world_size == 1
        assert keys[0].worker_id == 0
        assert keys[1].chunk_hash == int("def456", 16)
        assert kwargs["cum_chunk_lengths"] == [0, 256, 512]

    def test_prefetch_missing_fields(self):
        engine = _make_engine()
        app = _make_app(engine=engine)
        client = TestClient(app)
        resp = client.post("/memory/prefetch", json={})
        assert resp.status_code == 400


class TestEvictEndpoint:
    @patch(
        "lmcache_ascend.v1.internal_api_server.memory.memory_api."
        "_forward_to_all_workers"
    )
    def test_scheduler_forwards_to_workers(self, mock_forward):
        mock_forward.return_value = MagicMock(status_code=200)
        app = _make_app(engine=None)
        client = TestClient(app)
        client.post("/memory/evict", json={"chunk_hashes": ["abc"]})
        mock_forward.assert_called_once()

    def test_worker_evicts_and_clears_backend(self):
        chunk_hashes = ["abc123", "def456", "789abc"]
        backend_state = {int(h, 16): f"data-for-{h}" for h in chunk_hashes}

        engine = _make_engine(backend_state=backend_state)
        app = _make_app(engine=engine)
        client = TestClient(app)

        resp = client.post(
            "/memory/evict",
            json={"chunk_hashes": chunk_hashes},
        )

        assert resp.status_code == 200
        assert resp.json()["num_evicted"] == 3

        sm = engine.storage_manager
        sm.batched_remove.assert_called_once()
        (keys,) = sm.batched_remove.call_args[0]
        assert len(keys) == 3
        assert sm.batched_remove.call_args.kwargs == {"locations": None}
        assert backend_state == {}

    def test_evict_reports_zero_when_nothing_to_evict(self):
        engine = _make_engine(backend_state={})
        app = _make_app(engine=engine)
        client = TestClient(app)

        resp = client.post(
            "/memory/evict",
            json={"chunk_hashes": ["abc123"]},
        )

        assert resp.json()["num_evicted"] == 0
        assert resp.json()["status"] == "success"

    def test_evict_missing_fields(self):
        engine = _make_engine()
        app = _make_app(engine=engine)
        client = TestClient(app)
        resp = client.post("/memory/evict", json={})
        assert resp.status_code == 400
