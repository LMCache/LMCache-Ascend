# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401
# Standard
from typing import Generator
import multiprocessing as mp
import time

# Third Party
import pytest

# First Party
# NOTE (gingfung): we have to import the bootstrap and prepare here because
# multiprocessing will run from the top of the file here and if not bootstrapped,
# 'lmcache_tests' is not recognized, and the relevant patches won't be applied.
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.v1.multiprocess.server import run_cache_server  # noqa: E402
from lmcache_tests.v1.multiprocess.test_cache_server import (  # noqa: E402
    CHUNK_SIZE,
    CPU_BUFFER_SIZE,
    DEFAULT_TIMEOUT,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_URL,
    client,
    client_context,
    registered_instance,
    test_server_running,
    zmq_context,
)


def server_process_runner(
    host: str, port: int, chunk_size: int, cpu_buffer_size: float
):
    """
    Entry point for the server process.
    """
    run_cache_server(
        host=host, port=port, chunk_size=chunk_size, cpu_buffer_size=cpu_buffer_size
    )


@pytest.fixture(scope="module")
def server_process() -> Generator[mp.Process, None, None]:
    """
    Fixture that starts the cache server in a separate process.
    The server runs for the entire test module.
    """
    # Start server process
    mp.set_start_method("spawn", force=True)
    process = mp.Process(
        target=server_process_runner,
        args=(SERVER_HOST, SERVER_PORT, CHUNK_SIZE, CPU_BUFFER_SIZE),
        daemon=True,
    )
    process.start()

    # Wait for server to initialize
    time.sleep(2)

    yield process

    # Cleanup: terminate the server process
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
