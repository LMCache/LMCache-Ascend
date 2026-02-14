# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from typing import Any, List, Optional, Tuple, Union

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.gpu_connector import (
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
    SGLangGPUConnector,
    SGLangLayerwiseGPUConnector,
)
from lmcache.v1.memory_management import GPUMemoryAllocator, MemoryFormat, MemoryObj
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)

_IS_310P = None


def is_310p():
    global _IS_310P
    if _IS_310P is None:
        # First Party
        from lmcache_ascend import _build_info

        _IS_310P = _build_info.__soc_version__.lower().startswith("ascend310p")
    return _IS_310P


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

    SGLang NPU Layer-Concatenated
    kvcaches = [K_all_layers, V_all_layers]
    - K_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    """

    def is_separate_format(self) -> bool:
        return self == KVCacheFormat.SEPARATE_KV

    def is_merged_format(self) -> bool:
        return self == KVCacheFormat.MERGED_KV

    @staticmethod
    def detect(
        kvcaches: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
        use_mla: bool = False,
    ) -> "KVCacheFormat":
        if not kvcaches:
            return KVCacheFormat.UNDEFINED

        first_cache = kvcaches[0]

        # SGLang NPU: kvcaches = [K_tensor, V_tensor]
        if isinstance(kvcaches, list) and len(kvcaches) == 2:
            if isinstance(first_cache, torch.Tensor) and first_cache.ndim == 5:
                return KVCacheFormat.SEPARATE_KV

        if isinstance(first_cache, tuple):
            return KVCacheFormat.SEPARATE_KV
        elif isinstance(first_cache, torch.Tensor):
            ndim = first_cache.ndim
            shape = first_cache.shape

            # MLA detect
            # MLA Shape: [num_blocks, block_size, head_size] (3D)
            #         or: [1, num_blocks, block_size, head_size] (4D with first dim = 1)
            is_mla_shape = (ndim == 3) or (ndim == 4 and shape[0] == 1)
            if use_mla or is_mla_shape:
                return KVCacheFormat.MERGED_KV

            # Flash Attention：[2, num_blocks, block_size, num_heads, head_size]
            if ndim == 5 and shape[0] == 2:
                return KVCacheFormat.MERGED_KV

            # Flash Infer：[num_blocks, 2, block_size, num_heads, head_size]
            if ndim == 5 and shape[1] == 2:
                return KVCacheFormat.MERGED_KV

        return KVCacheFormat.UNDEFINED


class VLLMBufferLayerwiseNPUConnector(VLLMBufferLayerwiseGPUConnector):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        super().__init__(
            hidden_dim_size, num_layers, use_gpu, use_double_buffer, **kwargs
        )
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED
        self.use_mla = bool(kwargs.get("use_mla", False))
        self.fused_rotary_emb: Any = None

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            self.kv_format = KVCacheFormat.detect(kv_caches)
            if self.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError("Could not detect KV cache format.")

            ref_tensor = (
                kv_caches[0][0] if self.kv_format.is_separate_format() else kv_caches[0]
            )
            self.kv_device = ref_tensor.device

            first_layer_cache = kv_caches[0]

            # flash attention: [num_layers, 2, num_blocks,
            # block_size, num_heads, head_size]
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]

                assert key_tensor.shape == value_tensor.shape, (
                    f"Key and Value tensors must have identical shapes, "
                    f"got key={key_tensor.shape}, value={value_tensor.shape}"
                )

                k_cache_shape_per_layer = key_tensor.shape

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                assert (
                    first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2
                ), (
                    "MERGED_KV format should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                    f"Got shape: {first_layer_cache.shape}"
                )

                # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
                k_cache_shape_per_layer = first_layer_cache[0].shape
            else:
                raise ValueError(f"Unsupported KV cache format: {self.kv_format}")

            self.vllm_two_major = True

            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}\n"
                f"  - gpu_buffer_size: {gpu_buffer_size / (1024 * 1024)} MB"
            )

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def _prepare_transfer_context(self, kwargs) -> torch.Tensor:
        """
        Initialize context for KV cache transfer, validate required
        parameters and lazy init buffer.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise ValueError("kvcaches should be provided in kwargs or initialized.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        self._lazy_initialize_buffer(self.kvcaches)
        return kwargs["slot_mapping"]

    def _get_full_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        starts: List[int],
        ends: List[int],
        mode: str = "slice",
    ) -> tuple[torch.Tensor, int]:
        """
        Generate full continuous slot mapping tensor and calculate total token count.
        Supports two modes for different transfer directions (to/from GPU).
        """
        if mode == "slice":
            slot_mapping_full = slot_mapping[starts[0] : ends[-1]]
        elif mode == "concat":
            slot_mapping_chunks = [
                slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
            ]
            slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        else:
            raise ValueError(
                f"Unsupported slot mapping mode: {mode}, only 'slice'/'concat' allowed"
            )

        num_tokens = len(slot_mapping_full)
        return slot_mapping_full, num_tokens

    def _allocate_gpu_buffers(
        self, num_tokens: int, count: int = 1
    ) -> Union[object, list[object]]:
        """
        Allocate specified number of GPU buffers for KV cache with shape
        calculated by token count. Performs strict assertion checks for
        valid buffer allocation.
        """
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None, (
            "GPU buffer allocator not initialized"
        )
        buffers = []
        for _ in range(count):
            buf_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_2TD
            )
            assert buf_obj is not None, "Failed to allocate GPU buffer in GPUConnector"
            assert buf_obj.tensor is not None, "GPU buffer object has no valid tensor"
            buffers.append(buf_obj)
        return buffers[0] if count == 1 else buffers

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """
        slot_mapping = self._prepare_transfer_context(kwargs)

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        slot_mapping_full, num_all_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="slice"
        )

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]
        load_gpu_buffer_obj: Any = None
        compute_gpu_buffer_obj: Any = None
        compute_gpu_buffer_obj, load_gpu_buffer_obj = self._allocate_gpu_buffers(
            num_all_tokens, count=2
        )

        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=self.kv_device
            )
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kv_device
            )

        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                lmc_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    False,
                    self.kv_format.value,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )
                del self.buffer_mapping[layer_id - 2]

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done
                torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )

                # gap zeroing after RoPE
                if self.current_gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                with torch.cuda.stream(self.load_stream):
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions

            elif layer_id == self.num_layers:
                yield

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        slot_mapping = self._prepare_transfer_context(kwargs)

        buf_start = 0
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(start, end, device=self.kv_device, dtype=torch.int64)
                )

        slot_mapping_full, num_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="concat"
        )

        tmp_gpu_buffer_obj = self._allocate_gpu_buffers(num_tokens, count=1)

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    True,
                    self.kv_format.value,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )

                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield


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

        if is_310p():
            assert "num_kv_head" in kwargs, ("num_kv_head should be provided in 310p",)
            assert "head_size" in kwargs, ("head_size should be provided in 310p",)
            self.num_kv_head = kwargs["num_kv_head"]
            self.head_size = kwargs["head_size"]
            self.dtype = kwargs["dtype"]
            self.device = kwargs["device"]

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheEngineMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemGPUConnectorV2":
        """Create a connector from LMCacheEngineMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemGPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            num_kv_head=num_kv_head,
            head_size=head_size,
        )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)

        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError(
                "Undefined KV cache format detected. "
                "Unable to determine the format of input kv_caches."
            )

        if self.kv_format.is_separate_format():
            self.kvcaches_device = kv_caches[0][0].device
        else:
            self.kvcaches_device = kv_caches[0].device

        assert self.kvcaches_device.type == "npu", "The device should be Ascend NPU."
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
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                # kv_caches[0]: [tuple(k,v)]
                # 310P: [num_blocks, num_kv_heads * head_size // 16, block_size, 16]
                # 910B: [num_blocks, block_size, num_kv_heads, head_size]
                assert first_tensor.dim() >= 2
                if is_310p():
                    self.block_size = first_tensor.shape[-2]
                    self.page_buffer_size = first_tensor.shape[0] * self.block_size
                else:
                    self.page_buffer_size = (
                        first_tensor.shape[0] * first_tensor.shape[1]
                    )

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
                # 310P: [2, num_blocks, num_kv_heads * head_size // 16, block_size, 16]
                # 910B: [2, num_blocks, block_size, num_kv_heads, head_size]
                assert first_tensor.dim() == 5
                if is_310p():
                    self.block_size = first_tensor.shape[-2]
                    self.page_buffer_size = first_tensor.shape[1] * self.block_size
                else:
                    self.page_buffer_size = (
                        first_tensor.shape[1] * first_tensor.shape[2]
                    )

        return self.kv_cache_pointers_on_gpu[idx]

    def to_gpu_310p(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
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

        tmp_gpu_buffer = torch.empty(
            memory_obj.tensor.size(), dtype=self.dtype, device=self.device
        )

        tmp_gpu_buffer.copy_(memory_obj.tensor)

        lmc_ops.multi_layer_kv_transfer_310p(
            tmp_gpu_buffer,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            False,
            self.use_mla,
            self.num_kv_head,
            self.head_size,
            self.block_size,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

    def from_gpu_310p(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
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

        assert self.gpu_buffer.device == self.kvcaches_device

        tmp_gpu_buffer = torch.empty(
            memory_obj.tensor.size(), dtype=self.dtype, device=self.device
        )

        lmc_ops.multi_layer_kv_transfer_310p(
            tmp_gpu_buffer,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            True,
            self.use_mla,
            self.num_kv_head,
            self.head_size,
            self.block_size,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

        memory_obj.tensor.copy_(tmp_gpu_buffer)
        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

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
                    "The memory object should be in KV_2LTD format in "
                    " order to be processed by VLLMPagedMemNPUConnector."
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
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
            # No staging buffer or token count mismatch
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches_device,
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                    self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                )
            else:
                assert self.gpu_buffer.device == self.kvcaches_device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                lmc_ops.fused_multi_layer_kv_transfer(
                    memory_obj.tensor,  # dst: CPU buffer
                    tmp_gpu_buffer,  # staging cache
                    kv_cache_pointers,  # src: paged KV cache
                    slot_mapping[start:end],
                    self.kvcaches_device,
                    self.page_buffer_size,
                    True,  # from_gpu
                    self.use_mla,
                    self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                )

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                if is_310p():
                    self.to_gpu_310p(memory_obj, start, end, **kwargs)
                else:
                    self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            if is_310p():
                self.from_gpu_310p(memory_obj, start, end, **kwargs)
            else:
                self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMPagedMemLayerwiseNPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)

        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

        # layerwise mode currently does not support MLA
        self.use_mla = kwargs.get("use_mla", False)

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.

        Supports both legacy formats and new SEPARATE_KV format:
        - Legacy MERGED_KV: [2, num_blocks, block_size, num_heads, head_size]
        - New SEPARATE_KV: tuple(key_tensor, value_tensor) where each is
          [num_blocks, block_size, num_heads, head_size]
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")

            self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)

            if self.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError(
                    "Undefined KV cache format detected. "
                    "Unable to determine the format of input kv_caches."
                )

            logger.info(f"Detected KV cache format: {self.kv_format.name}")

            first_layer_cache = kv_caches[0]

            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]

                assert key_tensor.shape == value_tensor.shape, (
                    f"Key and Value tensors must have identical shapes, "
                    f"got key={key_tensor.shape}, value={value_tensor.shape}"
                )

                k_cache_shape_per_layer = key_tensor.shape
                self.vllm_two_major = False

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                assert (
                    first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2
                ), (
                    "MERGED_KV format should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                    f"Got shape: {first_layer_cache.shape}"
                )

                self.vllm_two_major = first_layer_cache.shape[0] == 2

                if self.vllm_two_major:
                    # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
                    k_cache_shape_per_layer = first_layer_cache[0].shape
                else:
                    # Flash Infer: [num_blocks, 2, block_size, num_heads, head_size]
                    k_cache_shape_per_layer = first_layer_cache[:, 0].shape
            else:
                raise ValueError(f"Unsupported KV cache format: {self.kv_format}")

            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}"
            )

            num_elements_key = k_cache_shape_per_layer.numel()
            num_elements = num_elements_key * 2
            gpu_buffer_size = num_elements * self.element_size

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate NPU buffer in NPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")
            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                if self.use_gpu:
                    cpu_tensors = []
                    for memory_obj in memory_objs_layer:
                        assert memory_obj.tensor is not None
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                        cpu_tensors.append(memory_obj.tensor)

                    # Fused transfer: N H2D memcpy + 1 scatter kernel
                    lmc_ops.batched_fused_single_layer_kv_transfer(
                        cpu_tensors,  # CPU memory objects
                        tmp_gpu_buffer_obj.tensor,  # GPU staging buffer
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        chunk_offsets,  # offset for each chunk
                        chunk_sizes,  # size for each chunk
                        False,  # to_gpu
                        self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                        True,  # token_major
                        self.vllm_two_major,
                    )

                else:
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.tensor is not None

                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            False,
                            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                            True,
                            self.vllm_two_major,
                        )

        yield

        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate NPU buffer in NPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                if self.use_gpu:
                    cpu_tensors = []
                    for memory_obj in memory_objs_layer:
                        assert memory_obj.tensor is not None
                        cpu_tensors.append(memory_obj.tensor)

                    # Fused transfer: 1 scatter kernel + N D2H memcpy
                    lmc_ops.batched_fused_single_layer_kv_transfer(
                        cpu_tensors,
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        chunk_offsets,
                        chunk_sizes,
                        True,  # from_gpu
                        self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                        True,  # token_major
                        self.vllm_two_major,
                    )
                else:
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.tensor is not None

                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            True,
                            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                            True,
                            self.vllm_two_major,
                        )
            yield

            if sync:
                self.store_stream.synchronize()

            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()
        yield


class SGLangNPUConnector(SGLangGPUConnector):
    pass


class SGLangLayerwiseNPUConnector(SGLangLayerwiseGPUConnector):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [num_blocks, block_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        # [2, self.layer_num, self.size // self.page_size + 1,
        # self.page_size, self.head_num, self.head_dim,]
        self.kv_format = KVCacheFormat.detect(kv_caches)
        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError("Could not detect KV cache format.")

        if self.use_gpu and self.gpu_buffer_allocator is None:
            k_cache_shape_per_layer = kv_caches[0][0].shape
            max_tokens = k_cache_shape_per_layer[0]
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}\n"
                f"  - gpu_buffer_size: {gpu_buffer_size / (1024 * 1024)} MB"
            )

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            current_layer_kv = (self.kvcaches[0][layer_id], self.kvcaches[1][layer_id])

            # memobj -> gpu_buffer -> kvcaches
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                if self.use_gpu:
                    tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                        memory_obj.tensor, non_blocking=True
                    )
                else:
                    lmc_ops.single_layer_kv_transfer(
                        memory_obj.tensor,
                        current_layer_kv,
                        slot_mapping[start:end],
                        False,
                        self.kv_format.value,
                        True,
                        True,
                    )

            if self.use_gpu:
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    current_layer_kv,
                    slot_mapping_full,
                    False,
                    self.kv_format.value,
                    True,
                    True,
                )

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            current_layer_kv = (self.kvcaches[0][layer_id], self.kvcaches[1][layer_id])

            if self.use_gpu:
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    current_layer_kv,
                    slot_mapping_full,
                    True,
                    self.kv_format.value,
                    True,
                    True,
                )

            start_idx = 0

            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.tensor is not None

                if self.use_gpu:
                    chunk_len = memory_obj.tensor.shape[0]
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start_idx : start_idx + chunk_len],
                        non_blocking=True,
                    )
                    start_idx += chunk_len
                else:
                    lmc_ops.single_layer_kv_transfer(
                        memory_obj.tensor,
                        current_layer_kv,
                        slot_mapping[start:end],
                        True,
                        self.kv_format.value,
                        True,
                        True,
                    )

            yield
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
