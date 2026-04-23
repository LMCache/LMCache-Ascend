# SPDX-License-Identifier: Apache-2.0
"""Ascend-specific LMCache service factory for vLLM.

``VllmServiceFactory.get_or_create_lmcache_engine`` constructs the GPU/NPU
connector without ``vllm_config``. On Ascend we subclass the factory so
``CreateNPUConnector`` receives ``vllm_config`` and can pre-size reusable buffers
(e.g. slot-mapping staging tied to ``scheduler_config.max_num_batched_tokens``).
"""

# Standard
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

# Third Party
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)


class VllmAscendServiceFactory(VllmServiceFactory):
    def get_or_create_lmcache_engine(self) -> Optional["LMCacheEngine"]:
        self._ensure_metadata()
        assert self.metadata is not None

        if (
            self.role == "scheduler"
            and not self.lmcache_config.enable_scheduler_bypass_lookup
        ):
            # Third Party
            from lmcache.observability import PrometheusLogger

            PrometheusLogger.GetOrCreate(
                self.metadata,
                config=self.lmcache_config,
            )
            return None

        # Third Party
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.utils import EngineType
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.v1.gpu_connector import CreateGPUConnector

        if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
            self.lmcache_engine = curr_engine
            return curr_engine

        if self.role == "scheduler":
            tpg = SimpleNamespace()
            tpg.broadcast = lambda tensor, src: tensor
            tpg.broadcast_object = lambda obj, src: obj
            vllm_gpu_connector = None
        else:
            # Third Party
            from vllm.distributed.parallel_state import get_tp_group

            tpg = get_tp_group()
            # Third Party
            from lmcache.integration.vllm.utils import vllm_layout_hints

            vllm_gpu_connector = CreateGPUConnector(
                self.lmcache_config,
                self.metadata,
                EngineType.VLLM,
                layout_hints=vllm_layout_hints(),
                vllm_config=self.vllm_config,
            )

        engine = LMCacheEngineBuilder.get_or_create(
            ENGINE_NAME,
            self.lmcache_config,
            self.metadata,
            vllm_gpu_connector,
            tpg.broadcast,
            tpg.broadcast_object,
        )
        self.lmcache_engine = engine

        if (
            self.role == "scheduler"
            and self.lmcache_config.enable_scheduler_bypass_lookup
        ):
            assert engine.save_only_first_rank or (
                self.lmcache_config.get_extra_config_value(
                    "remote_enable_mla_worker_id_as0", self.metadata.use_mla
                )
            ), (
                "enable_scheduler_bypass_lookup is only supported with "
                "save_only_first_rank or remote_enable_mla_worker_id_as0"
            )

        return engine
