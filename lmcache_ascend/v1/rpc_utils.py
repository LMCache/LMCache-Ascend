# SPDX-License-Identifier: Apache-2.0
# Standard
from functools import wraps
from typing import Callable, Literal, Optional
import hashlib
import socket

# Third Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

ServiceKind = Literal["lookup", "offload", "lookup_worker", "lookup_scheduler"]


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
