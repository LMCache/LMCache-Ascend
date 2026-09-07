# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache.v1.config import LMCacheEngineConfig
import pytest


def _mixed_sender_config():
    config = LMCacheEngineConfig.from_defaults(
        **{
            "enable_pd": True,
            "enable_p2p": True,
            "pd_role": "sender",
            "pd_buffer_size": 1024,
            "pd_buffer_device": "cpu",
            "enable_controller": True,
            "lmcache_instance_id": "mixed-sender",
            "controller_pull_url": "localhost:9800",
            "controller_reply_url": "localhost:9900",
            "lmcache_worker_ports": [9950],
            "p2p_host": "localhost",
            "p2p_init_ports": [9960],
            "p2p_lookup_ports": [9962],
            "transfer_channel": "hccl",
            "enable_async_loading": False,
        }
    )
    return config


def test_mixed_sender_keeps_upstream_pd_validation():
    config = _mixed_sender_config()
    config.save_unfull_chunk = False
    assert config.validate() is config
    assert config.enable_p2p is True
    assert config.save_unfull_chunk is True
    assert config.ascend_flatten_multi_spec is True
    assert config.ascend_bundle_multi_spec is True
    assert config.ascend_skip_state_groups is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pd_role", "receiver"),
        ("enable_async_loading", True),
        ("enable_controller", False),
        ("controller_pull_url", None),
        ("controller_reply_url", None),
        ("lmcache_worker_ports", []),
        ("p2p_host", None),
        ("p2p_init_ports", None),
        ("p2p_lookup_ports", None),
        ("transfer_channel", None),
    ],
)
def test_mixed_sender_rejects_invalid_p2p_config(field, value):
    config = _mixed_sender_config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        config.validate()
    assert config.enable_p2p is True


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("pd_backend_mode", "invalid", ValueError),
        ("pd_buffer_size", None, AssertionError),
        ("lmcache_instance_id", None, ValueError),
    ],
)
def test_mixed_sender_restores_p2p_after_upstream_validation_error(field, value, error):
    config = _mixed_sender_config()
    setattr(config, field, value)
    with pytest.raises(error):
        config.validate()
    assert config.enable_p2p is True
