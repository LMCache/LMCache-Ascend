# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.server import MPCacheEngine, parse_args, run_cache_server

logger = init_logger(__name__)


class NPUCacheContext:
    def __init__(self, kv_caches: KVCache, lmcache_chunk_size: int = 256):
        raise NotImplementedError("NPUCacheContext is not implemented yet.")


def register_npu_kv_cache(self, instance_id: int, kv_caches: KVCache) -> None:
    """
    Registers the KV cache tensors for a given GPU instance ID.

    Args:
        instance_id (int): The GPU instance ID (such as PID).
        kv_caches (KVCache): The KV cache tensor wrappers from vLLM.
    """
    npu_context = NPUCacheContext(kv_caches)
    self.gpu_contexts[instance_id] = npu_context
    logger.info(
        "Registered KV cache for NPU ID %d with %d layers",
        instance_id,
        npu_context.num_layers,
    )


MPCacheEngine.register_kv_cache = register_npu_kv_cache

if __name__ == "__main__":
    args = parse_args()
    run_cache_server(
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        cpu_buffer_size=args.cpu_buffer_size,
        max_workers=args.max_workers,
    )
