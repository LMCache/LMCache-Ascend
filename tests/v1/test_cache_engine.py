# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401

# TODO (gingfung): once we supported NPUDirectFS,
# re-enable test_multi_device_backends

# Third Party
from lmcache_tests.v1.test_cache_engine import (
    test_builder,
    test_builder_destroy,
    test_builder_destroy_multiple_instances,
    test_force_store_wait,
    test_paged_hierarchy_retrieve,
    test_paged_mem_leak,
    test_paged_mixed_retrieve,
    test_paged_prefetch_retrieve,
    test_paged_retrieve_after_eviction,
)
from lmcache_tests.v1.test_cache_engine import (
    test_paged_retrieve_prefix as original_paged_retrieve_prefix,
)
from lmcache_tests.v1.test_cache_engine import (
    test_paged_same_retrieve_store,
    test_paged_store_kv_tensors_mask,
    test_paged_store_offset,
)
import pytest


# TODO (gingfung): removed cachegen test until ready
@pytest.mark.parametrize("fmt", ["vllm"])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("backend", ["cpu", "local_disk", "remote"])
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
def test_paged_retrieve_prefix_patched(
    fmt, chunk_size, backend, lmserver_v1_process, autorelease_v1
):
    original_paged_retrieve_prefix(
        fmt, chunk_size, backend, lmserver_v1_process, autorelease_v1
    )
