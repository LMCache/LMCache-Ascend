# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Literal, Optional
import hashlib
import os
import socket

# Third Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

ServiceKind = Literal["lookup", "offload", "lookup_worker", "lookup_scheduler"]


def get_zmq_rpc_path_lmcache(
    engine_id: str,
    service_name: ServiceKind = "lookup",
    rpc_port: int = 0,
    rank: int = 0,
    base_url: Optional[str] = None,
) -> str:
    """Get the ZMQ RPC path for LMCache lookup and offload communication.

    Patched for Ascend to use shorter hash-based identifiers to ensure
    paths remain under the Unix domain socket 107 character limit.
    """
    if base_url is None:
        try:
            # Third Party
            import vllm.envs as envs

            base_url = envs.VLLM_RPC_BASE_PATH
        except (ImportError, ModuleNotFoundError):
            base_url = "/tmp/vllm_rpc"
            logger.debug("vllm not available, using default base_url: %s", base_url)
            os.makedirs(base_url, exist_ok=True)

    if service_name not in {"lookup", "offload", "lookup_worker", "lookup_scheduler"}:
        raise ValueError(
            f"service_name must be 'lookup' or 'offload', got {service_name!r}"
        )

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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
