# SPDX-License-Identifier: Apache-2.0
"""Thread-owned synchronous all-rank GDN RPC behind a nonblocking client."""

# Standard
from copy import copy
from types import SimpleNamespace

# Third Party
from lmcache.v1.lookup_client.factory import LookupClientFactory
from lmcache.v1.manager import LMCacheManager

# First Party
from lmcache_ascend.v1.async_control import SerialOwner
from lmcache_ascend.v1.gdn_async_config import gdn_async_restore_enabled


def async_gdn(config):
    return gdn_async_restore_enabled(config)


class AsyncStateClient:
    """Own ZMQ sockets and token database on one RPC thread, never scheduler."""

    def __init__(self, config, metadata, create_client):
        # Third Party
        from lmcache.v1.token_database import ChunkedTokenDatabase

        self.config = copy(config)
        # This config describes the synchronous *wire protocol*, not H2D mode.
        self.config.enable_async_loading = False
        self.transport = SimpleNamespace(world_size=metadata.world_size)
        # Existing GDN initialization checks the exact DB type. This independent
        # DB is used only for that interface check; hashing uses the owner DB.
        self.token_database = ChunkedTokenDatabase(self.config, metadata)
        self._jobs = {}
        self._admitted = set()
        self._owner = SerialOwner(
            lambda: create_client(self.config, metadata, None), "gdn-async-lookup"
        )
        self._error = None

    def poll(self, req_id, signature, operation):
        if self._error:
            raise RuntimeError("GDN_FULL_ASYNC_LOOKUP_FATAL: " + self._error)
        existing = self._jobs.get(req_id)
        if existing is not None and existing[0] != signature:
            self.cancel_lookup(req_id)
            existing = None
        if existing is None:
            self._jobs[req_id] = (signature, self._owner.submit(operation))
            return None
        future = existing[1]
        if not future.done():
            return None
        try:
            return future.result()
        except BaseException as error:
            self._error = f"{type(error).__name__}: {error}"
            raise

    def cancel_lookup(self, req_id):
        # First Party
        from lmcache_ascend.v1.state_lookup import cancel_state_lookup

        self._jobs.pop(req_id, None)
        if req_id in self._admitted:
            # Scheduler has allocated receive targets. Worker owns cleanup via
            # receive ACK; a scheduler abort must not release its source lease
            # before receive-only metadata reaches the worker.
            return
        # Release follows any preceding probe on the SAME owner/socket thread.
        # Never Future.cancel(): a probe may already have acquired remote pins.
        self._owner.submit(lambda client: cancel_state_lookup(client, req_id))

    def clear_lookup_status(self, req_id):
        # Upstream calls this upon allocation to forget only the LOCAL result.
        # Remote cancellation here would destroy the selected receive lease.
        self._jobs.pop(req_id, None)

    def close(self):
        for req_id in list(self._jobs):
            self.cancel_lookup(req_id)
        self._owner.close()

    def mark_admitted(self, req_id):
        self._jobs.pop(req_id, None)
        self._admitted.add(req_id)

    def received(self, req_ids):
        self._admitted.difference_update(req_ids)


class AscendGDNLookupFactory(LookupClientFactory):
    """Preserve the ordinary-KV factory unless GDN restore is opted in."""

    @staticmethod
    def create_lookup_client(config, metadata, lmcache_engine=None):
        if not async_gdn(config):
            return LookupClientFactory.create_lookup_client(
                config, metadata, lmcache_engine
            )
        if not config.enable_async_loading:
            raise ValueError("gdn_async_restore requires enable_async_loading=true")
        return AsyncStateClient(
            config, metadata, LookupClientFactory.create_lookup_client
        )

    @staticmethod
    def create_lookup_server(engine, metadata):
        if not async_gdn(engine.config):
            return LookupClientFactory.create_lookup_server(engine, metadata)
        # Third Party
        from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

        workers = engine.config.get_lookup_server_worker_ids(False, metadata.world_size)
        if workers and sorted(workers) != list(range(metadata.world_size)):
            raise ValueError("GDN restore requires every lookup worker")
        transport = LookupClientFactory._create_zmq_server_transport(metadata)
        return LMCacheLookupServer(engine, metadata, transport)


class AscendGDNManager(LMCacheManager):
    """GDN uses checkpoint RPC, not the ordinary-KV prefetch server."""

    def post_init(self):
        if (
            not async_gdn(self.config)
            or self.lmcache_engine is None
            or self._init_failed
        ):
            return super().post_init()
        try:
            self.lmcache_engine.post_init(async_lookup_server=None)
            self._init_health_monitor()
        except Exception as error:
            self._handle_post_init_failure(error)
            raise
