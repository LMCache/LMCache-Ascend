# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache_ascend.v1.npu_connector import VLLMPagedMemNPUConnectorV2

def create_npu_connector(hidden_dim, num_layers):
    return VLLMPagedMemNPUConnectorV2(hidden_dim, num_layers)