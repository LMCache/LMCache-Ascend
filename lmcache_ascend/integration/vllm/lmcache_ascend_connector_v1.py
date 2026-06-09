# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

# First Party
from lmcache_ascend import _build_info

if _build_info.__framework_name__ == "pytorch":
    # First Party
    import lmcache_ascend  # noqa: F401
elif _build_info.__framework_name__ == "mindspore":
    # First Party
    import lmcache_ascend.mindspore  # noqa: F401
else:
    raise ValueError("Unsupported Framework")

# Third Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import LMCacheAscendConnectorV1Impl

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        # Upstream LMCacheConnectorV1Dynamic does not pass kv_cache_config to the
        # impl; recreate with explicit config for multi-group discovery.
        self._lmcache_engine = LMCacheAscendConnectorV1Impl(
            vllm_config,
            role,
            self,
            kv_cache_config=kv_cache_config,
        )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._lmcache_engine.request_finished_all_groups(
            request, block_ids
        )
