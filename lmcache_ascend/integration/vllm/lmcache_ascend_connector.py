# SPDX-License-Identifier: Apache-2.0
"""In-process vllm-ascend LMCacheAscendConnector entry (SupportsHMA + Ascend impl)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)

from lmcache_ascend.integration.vllm.vllm_v1_adapter import LMCacheAscendConnectorV1Impl

if TYPE_CHECKING:
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
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._lmcache_engine = LMCacheAscendConnectorV1Impl(
            vllm_config,
            role,
            self,
            kv_cache_config=kv_cache_config,
        )

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._lmcache_engine.request_finished_all_groups(request, block_ids)
