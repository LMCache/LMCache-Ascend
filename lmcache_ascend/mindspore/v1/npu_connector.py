# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import List

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
import numpy as np
import torch

# First Party
from lmcache_ascend.v1.npu_connector import KVCacheFormat, is_310p
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)


class VLLMPagedMemNPUConnectorV2(VLLMPagedMemGPUConnectorV2):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)

        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED
        self.is_310p = is_310p()

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.kv_format = KVCacheFormat.detect(kv_caches)

        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError(
                "Undefined KV cache format detected. "
                "Unable to determine the format of input kv_caches."
            )

        if self.kv_format.is_separate_format():
            self.kvcaches_device = kv_caches[0][0].device
        else:
            self.kvcaches_device = kv_caches[0].device

        assert self.kvcaches_device.type == "Ascend", "The device should be Ascend NPU."
        idx = self.kvcaches_device.index

        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]

        if self.kv_format == KVCacheFormat.SEPARATE_KV:
            self.kv_size = 2
            pointers_list = []
            for k, v in kv_caches:
                pointers_list.append(k.data_ptr())
                pointers_list.append(v.data_ptr())

            self.kv_cache_pointers = torch.empty(
                self.num_layers * self.kv_size, dtype=torch.int64, device="cpu"
            )
        else:
            self.kv_size = 1
            pointers_list = [t.data_ptr() for t in kv_caches]

            self.kv_cache_pointers = torch.empty(
                self.num_layers, dtype=torch.int64, device="cpu"
            )

        self.kv_cache_pointers.numpy()[:] = pointers_list

        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.kv_cache_pointers.shape, dtype=torch.int64, device=self.kvcaches_device
        )

        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        first_tensor = (
            kv_caches[0][0] if self.kv_format.is_separate_format() else kv_caches[0]
        )

        if self.use_mla:
            # kv_caches[0].shape: [num_pages, page_size, head_size]
            # kv_caches[0].shape: [1, num_pages, page_size, head_size] (vllm-Ascend)
            self.page_buffer_size = kv_caches[0].shape[-3] * kv_caches[0].shape[-2]
        else:
            # vLLM v0.9.2 ...
            # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
            # 310P: [2, num_blocks, num_kv_heads * head_size // 16, block_size, 16]
            # 910B: [2, num_blocks, block_size, num_kv_heads, head_size]
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                # kv_caches[0]: [tuple(k,v)，tuple(k,v)]
                assert first_tensor.dim() >= 2
                self.page_buffer_size = first_tensor.shape[0] * first_tensor.shape[1]
            else:
                assert first_tensor.dim() == 5
                self.page_buffer_size = first_tensor.shape[1] * first_tensor.shape[2]

        return self.kv_cache_pointers_on_gpu[idx]

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemNPUConnector."
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format "
                    "in order to be processed by VLLMPagedMemNPUConnector."
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        if self.is_310p:
            # memory_obj -> tmp_gpu_buffer -> kvcaches
            self.gpu_buffer.zero_()
            target_gpu_buffer = self.gpu_buffer[:, :, : end - start, :].contiguous()
            target_gpu_buffer.copy_(torch.from_numpy(memory_obj.tensor))
            self.transfer_func = lmc_ops.multi_layer_kv_transfer_ms
        else:
            target_gpu_buffer = memory_obj.tensor
            self.transfer_func = lmc_ops.multi_layer_kv_transfer

        self.transfer_func(
            target_gpu_buffer,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.page_buffer_size,
            False,
            self.use_mla,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError("KV cache format is not initialized!")

        with torch.cuda.stream(self.store_stream):
            use_tmp_buf = self.is_310p or (
                self.gpu_buffer is not None and end - start != self.gpu_buffer.shape[2]
            )
            if use_tmp_buf:
                if self.is_310p:
                    self.gpu_buffer.zero_()
                target_buffer = self.gpu_buffer[:, :, : end - start, :].contiguous()
                self.transfer_func = lmc_ops.multi_layer_kv_transfer_ms
            else:
                target_buffer = memory_obj.tensor
                self.transfer_func = lmc_ops.multi_layer_kv_transfer

            self.transfer_func(
                target_buffer,
                kv_cache_pointers,
                slot_mapping[start:end],
                self.page_buffer_size,
                True,
                self.use_mla,
                self.kv_format.value,
            )

            if use_tmp_buf:
                np.copyto(memory_obj.tensor, target_buffer.cpu().numpy())

        # if not memory_obj.tensor.is_cuda:
        # Force a synchronize if the target buffer is NOT CUDA device
        # NOTE: for better performance, we may not want to sync for every
        # memory object
        self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])
