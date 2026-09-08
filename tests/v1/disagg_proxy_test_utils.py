# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from types import ModuleType
import importlib
import logging
import sys


def _install_lmcache_stubs() -> None:
    try:
        importlib.import_module("lmcache.v1.storage_backend.pd_backend")
        return
    except ModuleNotFoundError as exc:
        if exc.name is not None and not exc.name.startswith("lmcache"):
            raise

    lmcache = ModuleType("lmcache")
    lmcache.__path__ = []
    lmcache_logging = ModuleType("lmcache.logging")
    lmcache_v1 = ModuleType("lmcache.v1")
    lmcache_v1.__path__ = []
    storage_backend = ModuleType("lmcache.v1.storage_backend")
    storage_backend.__path__ = []
    pd_backend = ModuleType("lmcache.v1.storage_backend.pd_backend")

    class ProxyNotif:
        def __init__(self, req_id: str):
            self.req_id = req_id

    lmcache_logging.init_logger = logging.getLogger
    pd_backend.PDMsg = object
    pd_backend.ProxyNotif = ProxyNotif

    sys.modules.update(
        {
            "lmcache": lmcache,
            "lmcache.logging": lmcache_logging,
            "lmcache.v1": lmcache_v1,
            "lmcache.v1.storage_backend": storage_backend,
            "lmcache.v1.storage_backend.pd_backend": pd_backend,
        }
    )


def load_proxy_server():
    _install_lmcache_stubs()
    example_dir = Path(__file__).parents[2] / "examples" / "disagg_prefill"
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
    return importlib.import_module("disagg_proxy_server")


class FakeRequest:
    def __init__(self, payload: dict, headers: dict | None = None):
        self.payload = payload
        self.headers = headers or {}

    async def json(self) -> dict:
        return self.payload


class FakeResponse:
    def __init__(
        self,
        payload: dict | None = None,
        *,
        content: bytes = b"",
        status_code: int = 200,
        content_type: str = "application/json",
    ):
        self.payload = payload
        self.content = content
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    def json(self) -> dict:
        assert self.payload is not None
        return self.payload


async def collect_streaming_response(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.encode() if isinstance(chunk, str) else chunk)
    return b"".join(chunks)
