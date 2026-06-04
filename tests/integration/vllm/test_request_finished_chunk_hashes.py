# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest

_patched = False


def _import_adapter():
    global _patched
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    if not _patched:
        lmcache_ascend = pytest.importorskip("lmcache_ascend")
        lmcache_ascend._patch_vllm_v1_adapter()
        _patched = True
    return pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")


def _make_adapter(adapter_mod, *, lookup_client):
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.store_async = False
    adapter.kv_role = "kv_both"
    adapter.lmcache_engine = None
    adapter.lookup_client = lookup_client
    return adapter


class TestRequestFinished:
    def test_returns_chunk_hashes_in_response(self):
        adapter_mod = _import_adapter()

        inner = MagicMock()
        inner.get_cached_hashes.return_value = ["abc", "def"]

        adapter = _make_adapter(adapter_mod, lookup_client=inner)
        request = SimpleNamespace(request_id="req-001")

        with patch.object(
            adapter_mod.LMCacheConnectorV1Impl,
            "request_finished",
            return_value=(False, None),
        ):
            impl = adapter_mod.LMCacheAscendConnectorV1Impl
            _, return_params = impl.request_finished(adapter, request, [])

        assert return_params == {"chunk_hashes": ["abc", "def"]}
