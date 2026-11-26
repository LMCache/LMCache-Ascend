# SPDX-License-Identifier: Apache-2.0
"""Run the upstream transfer benchmark with ACL-compatible L1 configuration."""

# Standard
from functools import partial
import sys


def main() -> int:
    """Run the upstream CLI with an empty SHM name.

    The upstream benchmark has no SHM option, so temporarily override its config
    constructor reference without changing the shared configuration class.
    Arguments, transfers, verification, cleanup and exit codes remain upstream's.
    Calls must be sequential; the constructor reference is restored on every exit.

    Returns:
        The upstream exit code, 0 on success or 1 on benchmark failure.

    Raises:
        Exception: Upstream errors propagate after restoring the constructor.
    """
    # Third Party
    from lmcache.tools.transfer_channel_benchmark import benchmark
    from lmcache.tools.transfer_channel_benchmark.__main__ import main as benchmark_main

    config_type = benchmark.L1MemoryManagerConfig
    benchmark.L1MemoryManagerConfig = partial(  # type: ignore[misc, assignment]
        config_type, shm_name=""
    )
    try:
        return benchmark_main()
    finally:
        benchmark.L1MemoryManagerConfig = config_type  # type: ignore[misc]


if __name__ == "__main__":
    sys.exit(main())
