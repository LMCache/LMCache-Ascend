# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _validate_config(self):
    """Validate configuration"""
    # auto-adjust save_unfull_chunk for async loading to prevent CPU fragmentation
    if self.enable_async_loading:
        logger.warning(
            "Automatically setting save_unfull_chunk=False because "
            "enable_async_loading=True or use_layerwise=True to prevent "
            "CPU memory fragmentation"
        )
        self.save_unfull_chunk = False

    if self.enable_blending:
        if not self.save_unfull_chunk:
            logger.warning(
                "Automatically setting save_unfull_chunk=True because "
                "enable_blending=True"
            )
            self.save_unfull_chunk = True

    if self.enable_p2p:
        assert self.enable_controller
        assert self.controller_pull_url is not None
        assert self.controller_reply_url is not None
        assert self.lmcache_worker_ports is not None
        assert self.p2p_host is not None
        assert self.p2p_init_ports is not None
        assert self.p2p_lookup_ports is not None
        assert self.transfer_channel is not None

    enable_nixl_storage = self.extra_config is not None and self.extra_config.get(
        "enable_nixl_storage"
    )
    if self.enable_pd:
        assert self.pd_role is not None
        assert self.pd_buffer_size is not None
        assert self.pd_buffer_device is not None

        assert self.remote_url is None, "PD only supports remote_url=None"
        assert self.save_decode_cache is False, (
            "PD only supports save_decode_cache=False"
        )
        assert self.enable_p2p is False, "PD only supports enable_p2p=False"

    if enable_nixl_storage:
        assert self.extra_config.get("nixl_backend") is not None
        assert self.extra_config.get("nixl_path") is not None
        assert self.extra_config.get("nixl_file_pool_size") is not None
        assert self.nixl_buffer_size is not None
        assert self.nixl_buffer_device is not None

    return self
