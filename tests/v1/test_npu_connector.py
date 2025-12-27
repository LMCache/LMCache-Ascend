# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
from unittest.mock import patch

# First Party
from lmcache_tests.v1.test_gpu_connector import (
    test_vllm_paged_connector_v2_with_gpu_and_mla as original_test_vllm_paged_connector_v2_with_gpu_and_mla,
    test_layerwise_vllm_paged_connector_with_gpu as original_test_layerwise_vllm_paged_connector_with_gpu,
    test_batched_layerwise_vllm_paged_connector_with_gpu as original_test_batched_layerwise_vllm_paged_connector_with_gpu,
    test_vllm_paged_connector_v2_to_gpu_bench as original_test_vllm_paged_connector_v2_to_gpu_bench
    # TODO (gingfung): once we have sglang kernel, re-enable this
    #  test_sglang_connector_with_gpu_and_mla 
)

from lmcache_ascend.v1.npu_connector import (
    VLLMPagedMemNPUConnectorV2,
    VLLMPagedMemLayerwiseNPUConnector,
)

@pytest.mark.parametrize("use_npu", [True, False])
@pytest.mark.parametrize("use_mla", [True, False])
def test_vllm_paged_connector_v2_with_npu_and_mla(use_npu, use_mla):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemGPUConnectorV2"
    
    with patch(target_patch, new=VLLMPagedMemNPUConnectorV2):
        original_test_vllm_paged_connector_v2_with_gpu_and_mla(use_npu, use_mla)

@pytest.mark.parametrize("use_npu", [True])
def test_layerwise_vllm_paged_connector_with_npu(use_npu):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    
    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_layerwise_vllm_paged_connector_with_gpu(use_npu)


@pytest.mark.parametrize("use_npu", [True])
def test_batched_layerwise_vllm_paged_connector_with_npu(use_npu):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    
    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_batched_layerwise_vllm_paged_connector_with_gpu(use_npu)

def test_vllm_paged_connector_v2_to_npu_bench(benchmark):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemGPUConnectorV2"

    with patch(target_patch, new=VLLMPagedMemNPUConnectorV2):
        original_test_vllm_paged_connector_v2_to_gpu_bench(benchmark)