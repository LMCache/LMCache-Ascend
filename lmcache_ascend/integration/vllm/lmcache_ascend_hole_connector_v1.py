# SPDX-License-Identifier: Apache-2.0

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.logger import init_logger

# First Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
from lmcache_ascend import _build_info

if _build_info.__framework_name__ == "pytorch":
    # First Party
    import lmcache_ascend  # noqa: F401
elif _build_info.__framework_name__ == "mindspore":
    # First Party
    import lmcache_ascend.mindspore  # noqa: F401
else:
    raise ValueError("Unsupported Framework")

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter_hole import (
    LMCacheConnectorV1ImplHole,
)

logger = init_logger(__name__)


class LMCacheAscendHoleConnectorV1Dynamic(LMCacheConnectorV1Dynamic):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
        KVConnectorBase_V1.__init__(self, vllm_config=vllm_config, role=role)
        self._lmcache_engine = LMCacheConnectorV1ImplHole(vllm_config, role, self)
