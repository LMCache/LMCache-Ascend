# SPDX-License-Identifier: Apache-2.0
"""Use LMCache's logger on both the release and MP development branches."""

try:
    # Third Party
    from lmcache.v1.common.logging import init_logger
except ModuleNotFoundError as exc:
    if exc.name not in {"lmcache.v1.common", "lmcache.v1.common.logging"}:
        raise
    # Third Party
    from lmcache.logging import init_logger

__all__ = ["init_logger"]
