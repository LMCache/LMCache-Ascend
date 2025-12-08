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
from typing import List, Optional, Union, Tuple
from enum import Enum, auto

# Third Party
import torch

# First Party
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.gpu_connector import (
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
)
import lmcache_ascend.mindspore.c_ops as lmc_ops
from lmcache.logging import init_logger

logger = init_logger(__name__)

import mindspore

class KVCacheFormat(Enum):
    """
    The storage format enumeration of KV cache is used to distinguish 
    the KV cache data structures of different versions of vLLM.
    
    The order of enum values MUST match the KVCacheFormat 
    definition in kernels/types.h to ensure correct interoperability 
    between Python and C++ code.
    """

    UNDEFINED = 0

    MERGED_KV = auto()
    """merge format (eg: vLLM 0.9.2 ...)
    layer: [num_kv, num_blocks, block_size, num_heads, head_dim]
    """

    SEPARATE_KV = auto()
    """Separation format (eg: vLLM 0.11.0+ ...)
    layer: tuple: (K_tensor, V_tensor)
    - K_tensor.shape = [num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [num_blocks, block_size, num_heads, head_dim]

    eg: kvcaches[0] = (K, V)
    """
    
    def is_separate_format(self) -> bool:
        return self == KVCacheFormat.SEPARATE_KV
    
    def is_merged_format(self) -> bool:
        return self == KVCacheFormat.MERGED_KV
    
    @staticmethod
    def detect(
        kvcaches: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
    ) -> 'KVCacheFormat':
        if not kvcaches:
            return KVCacheFormat.UNDEFINED
        
        first_cache = kvcaches[0]
        
        if isinstance(first_cache, tuple):
            return KVCacheFormat.SEPARATE_KV
        elif isinstance(first_cache, torch.Tensor):
            if first_cache.shape[0] == 2:
                return KVCacheFormat.MERGED_KV
        
        return KVCacheFormat.UNDEFINED

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

        first_tensor = kv_caches[0][0] if self.kv_format.is_separate_format() else kv_caches[0]

        if self.use_mla:
            # kv_caches[0].shape: [num_pages, page_size, head_size]
            # kv_caches[0].shape: [1, num_pages, page_size, head_size] (vllm-Ascend)
            self.page_buffer_size = kv_caches[0].shape[-3] * kv_caches[0].shape[-2]
        else:
            # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size] vllm 0.9.2 ...
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
                    " order to be processed by VLLMPagedMemNPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemNPUConnector"
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        # lmc_ops.multi_layer_kv_transfer(
        #     memory_obj.tensor,
        #     kv_cache_pointers,
        #     slot_mapping[start:end],
        #     self.kvcaches_device,
        #     self.page_buffer_size,
        #     False,
        #     self.use_mla,
        #     self.kv_format.value # 1:MERGED_KV / 2:SEPARATE_KV
        # )

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.page_buffer_size,
            False,
            self.use_mla,
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
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                # lmc_ops.multi_layer_kv_transfer(
                #     memory_obj.tensor, 
                #     kv_cache_pointers, 
                #     slot_mapping[start:end], 
                #     self.kvcaches_device,
                #     self.page_buffer_size,
                #     True,
                #     self.use_mla,
                #     self.kv_format.value # 1:MERGED_KV / 2:SEPARATE_KV
                # )
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                )
            else:
                assert self.gpu_buffer.device == self.kvcaches_device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                # lmc_ops.multi_layer_kv_transfer(
                #     tmp_gpu_buffer,
                #     kv_cache_pointers,
                #     slot_mapping[start:end],
                #     self.kvcaches_device,
                #     self.page_buffer_size,
                #     True,
                #     self.use_mla,
                #     self.kv_format.value # 1:MERGED_KV / 2:SEPARATE_KV
                # )
                lmc_ops.multi_layer_kv_transfer(
                    tmp_gpu_buffer,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                )
                memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

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

class VLLMPagedMemLayerwiseNPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.load_stream = mindspore.runtime.Stream()
        self.store_stream = mindspore.runtime.Stream()

        if use_gpu:
            # Unsupported, quitely ignore the arg and copy directly
            logger.debug("Layerwise connector is ignoring `use_gpu` and copying directly")
            pass

        logger.debug("Initialized Mindspore paged layerwise NPU connector for caching to/from Ascend")

    def _lazy_initialize_buffer(self, kv_caches):
        # The base LMCache implementation is a no-op when use_gpu is False (otherwise it sets up an intermediary buffer). 
        # This class mirrors that functionality, but currently doesn't support use_gpu, so no-ops.
        return kv_caches

    @_lmcache_nvtx_annotate
    def batched_to_gpu(
        self,
        starts: List[int],
        ends: List[int],
        **kwargs):

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        kvcaches = self._lazy_initialize_buffer(kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])
        slot_mapping_full = mindspore.ops.concat(slot_mapping_chunks, axis=0)

        current_stream = mindspore.runtime.current_stream()

        logger.debug(f"Starting the layerwise {'synchronous' if sync else 'asynchronous'} load of {self.num_layers} layers")
        for layer_id in range(self.num_layers):
            if sync:
                current_stream.wait_stream(self.load_stream)

                if layer_id > 0:
                    logger.debug(f"Synchornised the loading of layer {layer_id - 1} (total layers: {self.num_layers})")

            memory_objs_layer = yield
            with mindspore.runtime.StreamCtx(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                for chunk_memory_obj, slot_mapping_chunk in zip(memory_objs_layer, slot_mapping_chunks):
                    lmc_ops.single_layer_kv_transfer(
                        chunk_memory_obj.tensor,
                        kvcaches[layer_id][0],
                        kvcaches[layer_id][1],
                        slot_mapping_chunk,
                        False,
                        True,
                    )

        yield

        if sync:
            self.store_stream.synchronize()
            logger.debug(f"Finished loading final layer {self.num_layers - 1}")

        logger.debug(f"Complete the layerwise {'synchronous' if sync else 'asynchronous'} load of {self.num_layers} layers")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(self, memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]], starts: List[int], ends: List[int], **kwargs,):
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        kvcaches = self._lazy_initialize_buffer(kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])
        slot_mapping_full = mindspore.ops.concat(slot_mapping_chunks, axis=0)

        num_tokens = len(slot_mapping_full)

        current_stream = mindspore.runtime.current_stream()

        logger.debug(f"Starting the layerwise {'synchronous' if sync else 'asynchronous'} store of {self.num_layers} layers")
        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]

            with mindspore.runtime.StreamCtx(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                for chunk_memory_obj, slot_mapping_chunk in zip(memory_objs_layer, slot_mapping_chunks):
                    lmc_ops.single_layer_kv_transfer(
                        chunk_memory_obj.tensor,
                        kvcaches[layer_id][0],
                        kvcaches[layer_id][1],
                        slot_mapping_chunk,
                        True,
                        True,
                    )

            yield

            if sync:
                self.store_stream.synchronize()
                logger.debug(f"Synchornised the storing of layer {layer_id} (total layers: {self.num_layers})")

        logger.debug(f"Completed the layerwise {'synchronous' if sync else 'asynchronous'} store of {self.num_layers} layers")
        yield
