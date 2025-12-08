# SPDX-License-Identifier: Apache-2.0

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

# First Party

from lmcache_ascend import _build_info
if _build_info.__framework_name__ == "pytorch":
    import lmcache_ascend
elif _build_info.__framework_name__ == "mindspore":
    import lmcache_ascend.mindspore
else:
    raise ValueError("Unsupported Framework")

from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)
