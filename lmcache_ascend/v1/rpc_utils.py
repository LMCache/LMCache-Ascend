# SPDX-License-Identifier: Apache-2.0
# Standard
from functools import wraps
from typing import Callable, Literal, Optional
import hashlib
import os
import socket

# Third Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

ServiceKind = Literal[
    "lookup", "lookup_pure", "offload", "lookup_worker", "lookup_scheduler"
]


def use_short_engine_id(func: Callable) -> Callable:
    """Decorator that shortens engine_id via MD5 hash for Unix socket path limit.

    Converts engine_id to an 8-character hex string to ensure paths
    remain under the 107 character Unix domain socket limit.
    """

    @wraps(func)
    def wrapper(
        engine_id: str,
        service_name: ServiceKind = "lookup",
        rpc_port: int = 0,
        rank: int = 0,
        base_url: Optional[str] = None,
    ) -> str:
        short_engine_id = hashlib.md5(engine_id.encode()).hexdigest()[:8]

        # NOTE(lookup_pure-whitelist): upstream LMCache 0.4.4 tightened
        # get_zmq_rpc_path_lmcache's service_name whitelist to {"lookup",
        # "offload", "lookup_worker", "lookup_scheduler"}, rejecting "lookup_pure".
        # The hole connector's "pure" path is a genuinely distinct service:
        # different protocol (raw zmq+msgpack vs REQ/REP), different response
        # shape (int vs HoleLookupResult), used as a fast-path gate before hole
        # lookup. We accept "lookup_pure" here by constructing the path locally
        # rather than delegating to upstream. PR-design item: discuss with the
        # maintainer whether to (a) keep this local widening, (b) request upstream
        # whitelist extension, or (c) redesign the hole connector to use a single
        # lookup channel (would require unifying the protocols).
        if service_name == "lookup_pure":
            if base_url is None:
                try:
                    # Third Party
                    import vllm.envs as envs

                    base_url = envs.VLLM_RPC_BASE_PATH
                except (ImportError, ModuleNotFoundError):
                    base_url = "/tmp/vllm_rpc"
                    logger.debug(
                        "vllm not available, using default base_url: %s", base_url
                    )
                    os.makedirs(base_url, exist_ok=True)

            if isinstance(rpc_port, str):
                rpc_port = rpc_port + str(rank)
            else:
                rpc_port += rank

            logger.debug(
                "Base URL: %s, Engine: %s, Service Name: %s, RPC Port: %s",
                base_url,
                short_engine_id,
                service_name,
                rpc_port,
            )

            return (
                f"{base_url}/engine_{short_engine_id}_service_{service_name}_"
                f"lmcache_rpc_port_{rpc_port}"
            )

        return func(
            short_engine_id,
            service_name=service_name,
            rpc_port=rpc_port,
            rank=rank,
            base_url=base_url,
        )

    return wrapper


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
