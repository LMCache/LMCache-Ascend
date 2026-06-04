# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, patch


class TestLMCacheAsyncLookupClient:
    @staticmethod
    def _make_client(**attrs):
        # First Party
        from lmcache_ascend.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupClient,
        )

        client = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
        client._lookup_hashes_cache = {}
        client._max_cache_size = 0
        for k, v in attrs.items():
            setattr(client, k, v)
        return client

    def test_lookup_caches_hex_hashes(self):
        client = self._make_client(
            world_size=1,
            push_sockets=[MagicMock()],
            lookup_backoff_time=0.001,
            token_database=MagicMock(),
        )
        client._max_cache_size = 8
        client.token_database.process_tokens.return_value = [
            (0, 256, 0xABC123),
            (256, 512, 0xDEF456),
        ]
        with patch("msgspec.msgpack.encode", return_value=b"mock"):
            with patch("time.sleep", return_value=None):
                client.lookup([1, 2, 3], "req-001")

        assert client._lookup_hashes_cache == {
            "req-001": [f"{0xABC123:x}", f"{0xDEF456:x}"]
        }

    def test_get_cached_hashes_pops(self):
        client = self._make_client()
        client._lookup_hashes_cache = {"req-001": ["abc", "def"]}
        assert client.get_cached_hashes("req-001") == ["abc", "def"]
        assert "req-001" not in client._lookup_hashes_cache
        assert client.get_cached_hashes("req-missing") is None
