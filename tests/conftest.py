# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from unittest.mock import patch
import random
import shlex
import socket
import subprocess
import time

# Third Party
import pytest
import torch
import torch_npu

# First Party
import lmcache_ascend
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.memory_management import MixedMemoryAllocator
from lmcache.v1.system_detection import NUMADetector


class MockRedis:
    def __init__(
        self, host=None, port=None, url=None, decode_responses=False, **kwargs
    ):
        self.store = {}
        self.host = host
        self.port = port
        self.url = url
        self.decode_responses = decode_responses

    def set(self, key, value):
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key, None)

    def exists(self, key):
        return key in self.store

    def delete(self, key):
        return self.store.pop(key, None) is not None

    def scan(self, cursor=0, match=None):
        keys = [s.encode("utf-8") for s in self.store.keys()]
        return (0, keys)

    def close(self):
        pass

    @classmethod
    def from_url(cls, url, decode_responses=False, **kwargs):
        """Mock implementation of Redis.from_url"""
        return cls(url=url, decode_responses=decode_responses, **kwargs)

    @classmethod
    def from_pool(cls, pool, **kwargs):
        """Mock implementation of Redis.from_pool"""
        return cls(**kwargs)


class MockAsyncRedis(MockRedis):
    """Async version of MockRedis"""

    async def set(self, key, value):
        self.store[key] = value
        return True

    async def get(self, key):
        return self.store.get(key, None)

    async def exists(self, key):
        return key in self.store

    async def delete(self, key):
        return self.store.pop(key, None) is not None

    async def close(self):
        pass

    @classmethod
    def from_url(cls, url, decode_responses=False, **kwargs):
        """Mock implementation of Redis.from_url"""
        return cls(url=url, decode_responses=decode_responses, **kwargs)

    @classmethod
    def from_pool(cls, pool, **kwargs):
        """Mock implementation of Redis.from_pool"""
        return cls(**kwargs)


class MockRedisSentinel:
    def __init__(self, hosts_and_ports, socket_timeout=None, **kwargs):
        self.hosts_and_ports = hosts_and_ports
        self.socket_timeout = socket_timeout
        # Create a shared store but separate instances for master/slave
        self.shared_store = {}
        self.master_redis = MockRedis()
        self.slave_redis = MockRedis()
        # Share the store between master and slave to simulate Redis Sentinel behavior
        self.master_redis.store = self.shared_store
        self.slave_redis.store = self.shared_store

    def master_for(
        self, service_name, socket_timeout=None, username=None, password=None, **kwargs
    ):
        return self.master_redis

    def slave_for(
        self, service_name, socket_timeout=None, username=None, password=None, **kwargs
    ):
        return self.slave_redis


@dataclass
class LMCacheServerProcess:
    server_url: str
    server_process: object


@pytest.fixture(scope="function", autouse=True)
def mock_redis():
    with (
        patch("redis.Redis", MockRedis) as mock_redis_class,
        patch("redis.from_url", MockRedis.from_url),
        patch("redis.asyncio.Redis", MockAsyncRedis),
        patch("redis.asyncio.from_url", MockAsyncRedis.from_url),
        patch("redis.asyncio.ConnectionPool.from_url", lambda url, **kwargs: None),
        patch("redis.asyncio.Redis.from_pool", MockAsyncRedis.from_pool),
    ):
        yield mock_redis_class


@pytest.fixture(scope="function", autouse=True)
def mock_redis_sentinel():
    with (
        patch("redis.Sentinel", MockRedisSentinel) as mock,
        patch("redis.asyncio.Sentinel", MockRedisSentinel),
    ):
        yield mock


@pytest.fixture(scope="module")
def lmserver_v1_process(request):
    def ensure_connection(host, port):
        retries = 10
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        successful = False
        while retries > 0:
            retries -= 1
            try:
                print("Probing connection, remaining retries: ", retries)
                client_socket.connect((host, port))
                successful = True
                break
            except ConnectionRefusedError:
                time.sleep(1)
                print("Connection refused!")
                continue
            except Exception as e:
                print(f"other Exception: {e}")
                continue

        client_socket.close()
        return successful

    # Specify remote device
    device = request.param

    # Start the process
    max_retries = 5
    while max_retries > 0:
        max_retries -= 1
        port_number = random.randint(10000, 65500)
        print("Starting the lmcache v1 server process on port")
        proc = subprocess.Popen(
            shlex.split(
                f"python3 -m lmcache.v1.server localhost {port_number} {device}"
            )
        )

        # Wait for lmcache process to start
        time.sleep(5)

        successful = False
        if proc.poll() is not None:
            successful = True
        else:
            successful = ensure_connection("localhost", port_number)

        if not successful:
            proc.terminate()
            proc.wait()
        else:
            break

    # Yield control back to the test until it finishes
    server_url = f"lm://localhost:{port_number}"
    yield LMCacheServerProcess(server_url, proc)

    # Terminate the process
    proc.terminate()
    proc.wait()

    # Destroy remote disk path
    if device not in ["cpu"]:
        subprocess.run(shlex.split(f"rm -rf {device}"))


@pytest.fixture(scope="module")
def lmserver_process(request):
    def ensure_connection(host, port):
        retries = 10
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        successful = False
        while retries > 0:
            retries -= 1
            try:
                print("Probing connection, remaining retries: ", retries)
                client_socket.connect((host, port))
                successful = True
                break
            except ConnectionRefusedError:
                time.sleep(1)
                print("Connection refused!")
                continue
            except Exception as e:
                print(f"other Exception: {e}")
                continue

        client_socket.close()
        return successful

    # Specify remote device
    device = request.param

    # Start the process
    max_retries = 5
    while max_retries > 0:
        max_retries -= 1
        port_number = random.randint(10000, 65500)
        print("Starting the lmcache server process on port")
        proc = subprocess.Popen(
            shlex.split(f"python3 -m lmcache.server localhost {port_number} {device}")
        )

        # Wait for lmcache process to start
        time.sleep(5)

        successful = False
        if proc.poll() is not None:
            successful = True
        else:
            successful = ensure_connection("localhost", port_number)

        if not successful:
            proc.terminate()
            proc.wait()
        else:
            break

    # Yield control back to the test until it finishes
    server_url = f"lm://localhost:{port_number}"
    yield LMCacheServerProcess(server_url, proc)

    # Terminate the process
    proc.terminate()
    proc.wait()

    # Destroy remote disk path
    if device not in ["cpu"]:
        subprocess.run(shlex.split(f"rm -rf {device}"))


@pytest.fixture(scope="function")
def autorelease(request):
    objects = []

    def _factory(obj):
        objects.append(obj)
        return obj

    yield _factory

    # Cleanup all objects created by the factory
    for obj in objects:
        obj.close()


@pytest.fixture(scope="function")
def autorelease_v1(request):
    objects = []

    def _factory(obj):
        objects.append(obj)
        return obj

    yield _factory

    LMCacheEngineBuilder.destroy("test")

    # Cleanup all objects created by the factory
    # for obj in objects:
    #    obj.close()


@pytest.fixture(scope="session")
def memory_allocator():
    """One MixedMemoryAllocator (5GB) for the whole test session;
    .close() is a no-op per-test."""
    # init torch npu
    _ = torch.randn((10,10), device="npu")
    numa_mapping = NUMADetector._read_from_sys()

    _real = MixedMemoryAllocator(5 * 1024 * 1024 * 1024, numa_mapping=numa_mapping)  # 5GB

    class _NoCloseWrapper:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            # No-op so per-test close() calls don't shut down the shared allocator
            pass

    try:
        yield _NoCloseWrapper(_real)
    finally:
        # Actually close once when the session ends
        _real.close()


@pytest.fixture(autouse=True)  # function-scoped by default
def use_shared_allocator(request, monkeypatch, memory_allocator):
    """Default: patch. Opt out with @pytest.mark.no_shared_allocator."""
    if request.node.get_closest_marker("no_shared_allocator"):
        # do NOT patch for this test
        yield
        return

    def _create_shared_allocator(config, metadata, numa_mapping):
        return memory_allocator

    monkeypatch.setattr(
        LMCacheEngineBuilder,
        "_Create_memory_allocator",
        _create_shared_allocator,
    )
    yield
