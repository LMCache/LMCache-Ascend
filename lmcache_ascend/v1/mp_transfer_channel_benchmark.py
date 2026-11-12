# SPDX-License-Identifier: Apache-2.0
"""Run the upstream transfer benchmark with ACL-compatible L1 configuration."""

# Standard
from functools import partial
import os
import sys


def main() -> int:
    """Run the upstream CLI, disabling benchmark SHM only for ACL Host memory.

    Set LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR before starting this entry point.
    Ascend package activation installs the memory wrappers before this module
    runs. The upstream benchmark has no SHM option, so temporarily override its
    config constructor reference without changing the shared configuration class.
    Arguments, transfers, verification, cleanup and exit codes remain upstream's.
    Calls must be sequential; the constructor reference is restored on every exit.

    Returns:
        The upstream exit code, 0 on success or 1 on benchmark failure.

    Raises:
        Exception: Upstream errors propagate after restoring the constructor.
    """
    # Third Party
    from lmcache.tools.transfer_channel_benchmark.__main__ import main as benchmark_main

    if os.environ.get("LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR") != "aclrt_malloc_host":
        return benchmark_main()

    # Third Party
    from lmcache.tools.transfer_channel_benchmark import benchmark

    config_type = benchmark.L1MemoryManagerConfig
    benchmark.L1MemoryManagerConfig = partial(config_type, shm_name="")  # type: ignore[misc, assignment]
    try:
        return benchmark_main()
    finally:
        benchmark.L1MemoryManagerConfig = config_type  # type: ignore[misc]


if __name__ == "__main__":
    sys.exit(main())
