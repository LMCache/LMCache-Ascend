# SPDX-License-Identifier: Apache-2.0
"""In-process vllm-ascend LMCacheAscendConnector entry (SupportsHMA + Ascend impl)."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)

# NOTE: Do not import LMCacheAscendConnectorV1Impl here for a second
# construction. ``lmcache_ascend._patch_vllm_v1_adapter`` rebinds the
# implementation class that the upstream base class constructor uses, so
# ``super().__init__()`` below already builds the Ascend impl (and starts its
# LMCacheManager services exactly once). Constructing another impl here would
# orphan the first manager's services without shutting them down.

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


class LMCacheAscendConnector(LMCacheConnectorV1, SupportsHMA):
    """vllm-ascend in-process connector: HMA-capable Ascend ``LMCacheConnectorV1``."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ) -> None:
        # Store kv_cache_config before super().__init__() so the patched impl
        # can retrieve it via parent._kv_cache_config fallback during construction.
        self._kv_cache_config = kv_cache_config
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

    def shutdown(self):
        """Delegate shutdown to the single implementation instance."""
        return self._lmcache_engine.shutdown()

    def update_state_after_alloc(
        self, request: "Request", blocks: Any, num_external_tokens: int
    ) -> None:
        self._lmcache_engine.update_state_after_alloc(
            request, num_external_tokens, blocks=blocks
        )

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._lmcache_engine.request_finished_all_groups(request, block_ids)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        impl = self._lmcache_engine
        if getattr(impl, "_gdn_restore_enabled", False):
            external = impl.get_num_new_matched_tokens(request, num_computed_tokens)
            # vLLM must wait for receive completion before forwarding the suffix.
            return external, external is not None and external > 0
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_connector_output(self, output):
        result = super().update_connector_output(output)
        # First Party
        from lmcache_ascend.v1.gdn_async_lookup import AsyncStateClient

        client = self._lmcache_engine.lookup_client
        if isinstance(client, AsyncStateClient):
            client.received(output.finished_recving or ())
        return result
