# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Dict

# Third Party
from torch import nn

# First Party
from lmcache.logging import init_logger
from lmcache_ascend.v1.blend.blender import LMCBlender
from lmcache.v1.compute.models.utils import VLLMModelTracker

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.gpu_connector import GPUConnectorInterface

logger = init_logger(__name__)

def get_or_create_blender(
    cls,
    instance_id: str,
    cache_engine: "LMCacheEngine",
    gpu_connector: "GPUConnectorInterface",
):
    """
    Get or create a blender for the given instance_id.
    """

    if instance_id not in cls._blenders:
        logger.info(f"Creating blender for {instance_id}")
        vllm_model = VLLMModelTracker.get_model(instance_id)
        blender = LMCBlender(
            cache_engine=cache_engine,
            gpu_connector=gpu_connector,
            vllm_model=vllm_model,
        )
        cls._blenders[instance_id] = blender
    else:
        logger.info(
            f"Blender for {instance_id} already exists, returning the original one."
        )
    return cls._blenders[instance_id]