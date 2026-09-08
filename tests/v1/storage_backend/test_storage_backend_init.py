# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest


def _mixed_sender_config(max_local_cpu_size: int = 16):
    return SimpleNamespace(
        enable_pd=True,
        enable_p2p=True,
        pd_role="sender",
        enable_async_loading=False,
        local_cpu=False,
        local_disk=False,
        max_local_cpu_size=max_local_cpu_size,
        max_local_disk_size=0,
        enable_controller=True,
        controller_pull_url="localhost:9800",
        controller_reply_url="localhost:9900",
        lmcache_worker_ports=[9950],
        p2p_host="localhost",
        p2p_init_ports=[9960],
        p2p_lookup_ports=[9962],
        transfer_channel="hccl",
        remote_storage_plugins=None,
        remote_url=None,
        use_layerwise=False,
        extra_config=None,
    )


def _worker_metadata():
    return SimpleNamespace(role="worker", worker_id=0)


def _patch_backend_constructors(monkeypatch):
    # First Party
    import lmcache_ascend.v1.storage_backend as storage_backend_module
    import lmcache_ascend.v1.storage_backend.pd as pd_module

    class _FakeLocalCPUBackend:
        def __init__(self, config, metadata, dst_device, lmcache_worker):
            self.config = config
            self.metadata = metadata
            self.dst_device = dst_device
            self.lmcache_worker = lmcache_worker

        def __str__(self):
            return "LocalCPUBackend"

    class _FakePDBackend:
        def __init__(self, config, metadata):
            self.config = config
            self.metadata = metadata

    class _FakeP2PBackend:
        def __init__(self, config, metadata, loop, local_cpu_backend, lmcache_worker):
            self.config = config
            self.metadata = metadata
            self.loop = loop
            self.local_cpu_backend = local_cpu_backend
            self.lmcache_worker = lmcache_worker

        def __str__(self):
            return "P2PBackend"

    monkeypatch.setattr(storage_backend_module, "is_npu_worker", lambda metadata: False)
    monkeypatch.setattr(storage_backend_module, "LocalCPUBackend", _FakeLocalCPUBackend)
    monkeypatch.setattr(storage_backend_module, "AscendP2PBackend", _FakeP2PBackend)
    monkeypatch.setattr(pd_module, "AscendPDBackend", _FakePDBackend)
    monkeypatch.setattr(storage_backend_module, "storage_plugin_launcher", MagicMock())

    return storage_backend_module


def test_mixed_pd_p2p_sender_provisions_local_cpu_backend(monkeypatch):
    """P2P construction gets a LocalCPUBackend even when local_cpu is disabled."""
    storage_backend_module = _patch_backend_constructors(monkeypatch)
    config = _mixed_sender_config(max_local_cpu_size=16)

    backends = storage_backend_module.CreateStorageBackends(
        config,
        _worker_metadata(),
        MagicMock(),
        lmcache_worker=MagicMock(),
    )

    assert list(backends.keys()) == ["PDBackend", "LocalCPUBackend", "P2PBackend"]
    assert backends["P2PBackend"].local_cpu_backend is backends["LocalCPUBackend"]


def test_mixed_pd_p2p_sender_requires_local_cpu_capacity(monkeypatch):
    """P2P construction reports missing LocalCPUBackend capacity explicitly."""
    storage_backend_module = _patch_backend_constructors(monkeypatch)
    config = _mixed_sender_config(max_local_cpu_size=0)

    with pytest.raises(ValueError, match="max_local_cpu_size"):
        storage_backend_module.CreateStorageBackends(
            config,
            _worker_metadata(),
            MagicMock(),
            lmcache_worker=MagicMock(),
        )
