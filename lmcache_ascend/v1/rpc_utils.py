# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Literal, Optional
import hashlib
import os

# Third Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)

ServiceKind = Literal["lookup", "offload", "lookup_worker", "lookup_scheduler"]


def get_zmq_rpc_path_lmcache(
    vllm_config: Optional["VllmConfig"] = None,
    service_name: ServiceKind = "lookup",
    rpc_port: int = 0,
    rank: int = 0,
) -> str:
    """Get the ZMQ RPC path for LMCache lookup and offload communication."""
    # Try to import vllm.envs, fallback to default if not available
    try:
        # Third Party
        import vllm.envs as envs

        base_url = envs.VLLM_RPC_BASE_PATH
    except (ImportError, ModuleNotFoundError):
        # Fallback for testing environments without vllm
        base_url = "/tmp/vllm_rpc"
        logger.debug("vllm not available, using default base_url: %s", base_url)
        # Ensure the directory exists for IPC socket
        os.makedirs(base_url, exist_ok=True)

    if vllm_config is None or vllm_config.kv_transfer_config is None:
        raise ValueError("A valid kv_transfer_config with engine_id is required.")

    if service_name not in {"lookup", "offload", "lookup_worker", "lookup_scheduler"}:
        raise ValueError(
            f"service_name must be 'lookup' or 'offload', got {service_name!r}"
        )

    engine_id = vllm_config.kv_transfer_config.engine_id

    if isinstance(rpc_port, str):
        rpc_port = rpc_port + str(rank)
    else:
        rpc_port += rank

    logger.debug(
        "Base URL: %s, Engine: %s, Service Name: %s, RPC Port: %s",
        base_url,
        engine_id,
        service_name,
        rpc_port,
    )

    # reduce engine_id length
    short_engine_id = hashlib.md5(engine_id.encode()).hexdigest()[:8]

    socket_path = (
        f"{base_url}/engine_{short_engine_id}_service_{service_name}_"
        f"lmcache_rpc_port_{rpc_port}"
    )

    return socket_path
