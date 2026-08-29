# SPDX-License-Identifier: Apache-2.0
"""``LMCacheAscendConnectorV1Impl`` must reject CacheBlend + vLLM prefix cache.

The two are documented as mutually exclusive in ``examples/blending/README.md``.
When vLLM's prefix cache already covers part of a request, LMCache masks those
tokens out of the retrieval, so the blend buffer holds only the remaining
suffix and blending would run against a shifted KV window. vLLM enables prefix
caching by default, so the combination has to be refused before
``LMCacheManager`` is built and its services are started.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest


def _adapter_mod():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    return pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")


def _apply(*, enable_blending=False, extra_config=None, enable_prefix_caching):
    """Run the hook the base ``__init__`` calls before creating any service."""
    # Third Party
    from lmcache.v1.config import LMCacheEngineConfig

    adapter_mod = _adapter_mod()
    config = LMCacheEngineConfig.from_legacy()
    config.enable_blending = enable_blending
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra_config),
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
    )
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter._apply_extra_config(config, vllm_config)
    return config


def test_blending_with_prefix_caching_is_rejected():
    with pytest.raises(ValueError, match="prefix cache"):
        _apply(enable_blending=True, enable_prefix_caching=True)


def test_blending_from_extra_config_is_also_rejected():
    """The check must run after the ``lmcache.`` extra-config keys are folded in."""
    with pytest.raises(ValueError, match="prefix cache"):
        _apply(
            enable_blending=False,
            extra_config={"lmcache.enable_blending": True},
            enable_prefix_caching=True,
        )


@pytest.mark.parametrize(
    "enable_blending,enable_prefix_caching",
    [(True, False), (False, True), (False, False)],
)
def test_other_combinations_are_accepted(enable_blending, enable_prefix_caching):
    config = _apply(
        enable_blending=enable_blending,
        enable_prefix_caching=enable_prefix_caching,
    )
    assert config.enable_blending is enable_blending
