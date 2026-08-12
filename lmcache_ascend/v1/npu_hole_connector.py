# SPDX-License-Identifier: Apache-2.0
"""
Hole-aware layerwise NPU connector for assembling non-contiguous cached K/V.

This module provides the retrieval and buffer-assembly layer used by the
hole-mode blender. It extends the standard layerwise NPU connector with the
ability to load cached K/V for multiple hit ranges, place them into one dense
buffer covering the full `covered_tokens` span, and leave explicit gaps for the
hole positions that will be filled by fresh computation during layer-0 forward.

See `docs/hole-feature-overview.md` for the maintainer-facing overview of how
this connector participates in the worker load path.
"""

# Standard
from typing import Any, List, Union, cast
import os
import time

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.trace_utils import (
    mask_to_string,
    summarize_kv_tensor_stats,
    summarize_slot_mapping,
    tensor_to_list,
    trace_flow,
    trace_flow_enabled,
    trace_layer_enabled,
)
import torch

# First Party
from lmcache_ascend.v1.npu_connector import VLLMBufferLayerwiseNPUConnector
from lmcache_ascend.v1.timer import emit_timer
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)


def _env_enabled(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() not in {"", "0", "false", "no", "off"}


class VLLMBufferLayerwiseNPULegacyHoleConnector(VLLMBufferLayerwiseNPUConnector):
    """Legacy CacheBlend connector used only by hole-mode fallback requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fine_grained_sync_enabled = _env_enabled(
            "LMCACHE_HOLE_FINE_GRAINED_SYNC",
            default=True,
        )
        self._fine_grained_sync_runtime_disabled = False
        self._reuse_timer_force_sync_enabled = _env_enabled(
            "LMCACHE_REUSE_TIMER_FORCE_SYNC",
            default=False,
        )
        self._buffer_load_events: dict[int, object] = {}
        self._buffer_flush_events: dict[int, object] = {}

    def _emit_reuse_timer(
        self,
        bucket: str,
        *,
        req_id: str | int | None,
        layer_id: int,
        duration_ms: float,
        path: str | None,
        load_mode: str | None,
    ) -> None:
        emit_timer(
            bucket,
            req_id=req_id,
            layer_id=layer_id,
            duration_ms=duration_ms,
            path=path,
            load_mode=load_mode,
        )

    def _maybe_force_sync_reuse_timer(self, *, use_load_stream: bool = False) -> None:
        if not self._reuse_timer_force_sync_enabled:
            return
        if use_load_stream:
            self.load_stream.synchronize()
            return
        torch.cuda.synchronize()

    def _wait_for_loaded_buffer(
        self,
        *,
        req_id: str | int | None,
        layer_id: int,
        buffer_obj: Any | None = None,
        path: str | None,
        load_mode: str | None,
    ) -> None:
        use_fine_grained = (
            self._fine_grained_sync_enabled
            and not self._fine_grained_sync_runtime_disabled
        )
        if use_fine_grained:
            try:
                sync_start = time.perf_counter()
                current_stream = torch.cuda.current_stream()
                load_event = (
                    None
                    if buffer_obj is None
                    else self._buffer_load_events.pop(id(buffer_obj), None)
                )
                if load_event is not None:
                    current_stream.wait_event(load_event)
                    strategy = "wait_event"
                else:
                    current_stream.wait_stream(self.load_stream)
                    strategy = "wait_stream"
                if self._reuse_timer_force_sync_enabled:
                    current_stream.synchronize()
                self._emit_reuse_timer(
                    "reuse_sync",
                    req_id=req_id,
                    layer_id=layer_id,
                    duration_ms=(time.perf_counter() - sync_start) * 1000.0,
                    path=path,
                    load_mode=load_mode,
                )
                if trace_flow_enabled():
                    trace_flow(
                        "npu_connector.hole_legacy",
                        "sync_loaded_buffer",
                        layer_id=layer_id,
                        strategy=strategy,
                    )
                return
            except Exception as exc:
                self._fine_grained_sync_runtime_disabled = True
                logger.exception(
                    "Legacy hole fine-grained load sync failed at layer %d; "
                    "falling back to global synchronize for the rest of the run.",
                    layer_id,
                )
                if trace_flow_enabled():
                    trace_flow(
                        "npu_connector.hole_legacy",
                        "sync_loaded_buffer_failure",
                        layer_id=layer_id,
                        strategy="wait_event_or_wait_stream",
                        fallback="global_synchronize",
                        error=repr(exc),
                    )

        sync_start = time.perf_counter()
        torch.cuda.synchronize()
        self._emit_reuse_timer(
            "reuse_sync",
            req_id=req_id,
            layer_id=layer_id,
            duration_ms=(time.perf_counter() - sync_start) * 1000.0,
            path=path,
            load_mode=load_mode,
        )
        if trace_flow_enabled():
            trace_flow(
                "npu_connector.hole_legacy",
                "sync_loaded_buffer",
                layer_id=layer_id,
                strategy="global_synchronize",
                runtime_disabled=self._fine_grained_sync_runtime_disabled,
            )

    def _record_load_buffer_ready(self, *, buffer_obj: Any) -> None:
        if (
            not self._fine_grained_sync_enabled
            or self._fine_grained_sync_runtime_disabled
        ):
            return
        load_event = torch.npu.Event()
        load_event.record(self.load_stream)
        self._buffer_load_events[id(buffer_obj)] = load_event

    def _record_flush_buffer_busy(self, *, buffer_obj: Any) -> None:
        if (
            not self._fine_grained_sync_enabled
            or self._fine_grained_sync_runtime_disabled
        ):
            return
        flush_event = torch.npu.Event()
        flush_event.record(torch.cuda.current_stream())
        self._buffer_flush_events[id(buffer_obj)] = flush_event

    def _wait_buffer_reusable(
        self,
        *,
        buffer_obj: Any,
        req_id: str | int | None,
        layer_id: int,
        path: str | None,
        load_mode: str | None,
    ) -> None:
        if (
            not self._fine_grained_sync_enabled
            or self._fine_grained_sync_runtime_disabled
        ):
            return

        flush_event = self._buffer_flush_events.pop(id(buffer_obj), None)
        if flush_event is None:
            return

        wait_start = time.perf_counter()
        self.load_stream.wait_event(flush_event)
        if self._reuse_timer_force_sync_enabled:
            self.load_stream.synchronize()
        self._emit_reuse_timer(
            "reuse_flush_reuse_wait",
            req_id=req_id,
            layer_id=layer_id,
            duration_ms=(time.perf_counter() - wait_start) * 1000.0,
            path=path,
            load_mode=load_mode,
        )

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """Load legacy fallback chunks with hole-safe buffer synchronization."""
        slot_mapping = self._prepare_transfer_context(kwargs)
        debug_chunk_tags = list(kwargs.get("debug_chunk_tags", []) or [])

        if self.fused_rotary_emb is None and self.cache_positions:
            raise ValueError("fused_rotary_emb must be set before legacy loading.")

        timer_req_id = kwargs.get("req_id")
        timer_path = str(kwargs.get("timer_path", "hole"))
        timer_load_mode = kwargs.get("timer_load_mode", "legacy")

        slot_mapping_full, num_all_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="slice"
        )

        gap_mask_start = time.perf_counter()
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]
        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]
        self._emit_reuse_timer(
            "reuse_gap_mask",
            req_id=timer_req_id,
            layer_id=0,
            duration_ms=(time.perf_counter() - gap_mask_start) * 1000.0,
            path=timer_path,
            load_mode=timer_load_mode,
        )
        if trace_flow_enabled():
            trace_flow(
                "npu_connector.hole_legacy",
                "batched_to_gpu_start",
                starts=starts,
                ends=ends,
                slot_mapping=summarize_slot_mapping(slot_mapping_full),
                num_all_tokens=num_all_tokens,
                gap_mask=mask_to_string(gap_mask),
                gap_positions=self.current_gap_positions.tolist(),
            )

        buffer_alloc_start = time.perf_counter()
        allocated_buffers = cast(
            list[MemoryObj],
            self._allocate_gpu_buffers(num_all_tokens, count=2),
        )
        compute_gpu_buffer_obj, load_gpu_buffer_obj = allocated_buffers
        self._emit_reuse_timer(
            "reuse_buffer_alloc",
            req_id=timer_req_id,
            layer_id=0,
            duration_ms=(time.perf_counter() - buffer_alloc_start) * 1000.0,
            path=timer_path,
            load_mode=timer_load_mode,
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
                flush_start = time.perf_counter()
                flushed_buffer_obj = self.buffer_mapping[layer_id - 2]
                lmc_ops.single_layer_kv_transfer(
                    flushed_buffer_obj.tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    False,
                    self.kv_format.value,
                    False,
                    self.vllm_two_major,
                )
                self._maybe_force_sync_reuse_timer()
                self._emit_reuse_timer(
                    "reuse_flush",
                    req_id=timer_req_id,
                    layer_id=layer_id - 2,
                    duration_ms=(time.perf_counter() - flush_start) * 1000.0,
                    path=timer_path,
                    load_mode=timer_load_mode,
                )
                self._record_flush_buffer_busy(buffer_obj=flushed_buffer_obj)
                del self.buffer_mapping[layer_id - 2]

            if 0 < layer_id <= self.num_layers:
                self._wait_for_loaded_buffer(
                    req_id=timer_req_id,
                    layer_id=layer_id - 1,
                    buffer_obj=load_gpu_buffer_obj,
                    path=timer_path,
                    load_mode=timer_load_mode,
                )
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None
                    rope_start = time.perf_counter()
                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )
                    self._maybe_force_sync_reuse_timer()
                    self._emit_reuse_timer(
                        "reuse_rope",
                        req_id=timer_req_id,
                        layer_id=layer_id - 1,
                        duration_ms=(time.perf_counter() - rope_start) * 1000.0,
                        path=timer_path,
                        load_mode=timer_load_mode,
                    )

                if self.current_gap_positions.numel():
                    zero_start = time.perf_counter()
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0
                    self._maybe_force_sync_reuse_timer()
                    self._emit_reuse_timer(
                        "reuse_zero_gap",
                        req_id=timer_req_id,
                        layer_id=layer_id - 1,
                        duration_ms=(time.perf_counter() - zero_start) * 1000.0,
                        path=timer_path,
                        load_mode=timer_load_mode,
                    )

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

            if layer_id < self.num_layers:
                memory_objs_layer = yield
                with torch.cuda.stream(self.load_stream):
                    self._wait_buffer_reusable(
                        buffer_obj=load_gpu_buffer_obj,
                        req_id=timer_req_id,
                        layer_id=layer_id,
                        path=timer_path,
                        load_mode=timer_load_mode,
                    )
                    copy_start = time.perf_counter()
                    for chunk_idx, (start, end, memory_obj) in enumerate(
                        zip(starts, ends, memory_objs_layer, strict=False)
                    ):
                        chunk_tag = (
                            debug_chunk_tags[chunk_idx]
                            if chunk_idx < len(debug_chunk_tags)
                            else None
                        )
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)
                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)
                        if trace_flow_enabled() and trace_layer_enabled(layer_id):
                            trace_flow(
                                "npu_connector.hole_legacy",
                                "load_chunk_kv_stats",
                                layer_id=layer_id,
                                start=start,
                                end=end,
                                chunk_tag=chunk_tag,
                                cached_positions=tensor_to_list(
                                    memory_obj.metadata.cached_positions,
                                    dtype=torch.long,
                                ),
                                k_stats=summarize_kv_tensor_stats(memory_obj.tensor[0]),
                                v_stats=summarize_kv_tensor_stats(memory_obj.tensor[1]),
                            )
                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions

                    self._record_load_buffer_ready(buffer_obj=load_gpu_buffer_obj)
                    self._maybe_force_sync_reuse_timer(use_load_stream=True)
                    self._emit_reuse_timer(
                        "reuse_copy_enqueue",
                        req_id=timer_req_id,
                        layer_id=layer_id,
                        duration_ms=(time.perf_counter() - copy_start) * 1000.0,
                        path=timer_path,
                        load_mode=timer_load_mode,
                    )
            elif layer_id == self.num_layers:
                yield

        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()
        self._buffer_load_events.clear()
        self._buffer_flush_events.clear()

        assert not self.buffer_mapping, (
            "There are still layers in the buffer mapping after releasing buffers."
        )
        yield


class VLLMBufferLayerwiseNPUHoleConnector(VLLMBufferLayerwiseNPUConnector):
    """
    Hole-aware layerwise buffer connector for non-contiguous K/V reuse.

    This class extends `VLLMBufferLayerwiseNPUConnector` to handle hole-aware
    load specifications instead of a single contiguous prefix. Its main
    responsibilities are to retrieve cached K/V for each hit range, assemble
    those ranges into a dense buffer spanning `covered_tokens`, and expose hit
    and gap positions to the blender so layer 0 can fill hole positions and the
    check layer can restrict topK candidate selection to hit positions.

    See `docs/hole-feature-overview.md` for the higher-level request flow and
    how this connector fits into the worker-side load path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_hit_positions = torch.empty((0,), dtype=torch.long)
        self.current_prefix_start = 0
        self.current_prefix_end = 0
        self._gap_buffer_semantics = "hole"
        self._gap_buffer_req_id: str | None = None
        self._gap_buffer_load_mode: str | None = None
        self._hole_fine_grained_sync_enabled = _env_enabled(
            "LMCACHE_HOLE_FINE_GRAINED_SYNC",
            default=True,
        )
        self._hole_skip_gap_zeroing_enabled = _env_enabled(
            "LMCACHE_HOLE_SKIP_GAP_ZEROING",
            default=True,
        )
        self._hole_assert_gaps_enabled = _env_enabled(
            "LMCACHE_HOLE_ASSERT_GAPS",
            default=False,
        )
        self._reuse_timer_force_sync_enabled = _env_enabled(
            "LMCACHE_REUSE_TIMER_FORCE_SYNC",
            default=False,
        )
        self._hole_fine_grained_sync_runtime_disabled = False
        self.flush_stream = torch.cuda.Stream()
        self.rope_stream = torch.cuda.Stream()
        self._buffer_flush_events: dict[int, object] = {}
        self._buffer_load_events: dict[int, object] = {}
        self._buffer_rope_events: dict[int, object] = {}
        self._buffer_scatter_events: dict[int, object] = {}
        logger.info(
            "Hole connector config: fine_grained_sync=%s skip_gap_zeroing=%s "
            "assert_gaps=%s reuse_timer_force_sync=%s",
            self._hole_fine_grained_sync_enabled,
            self._hole_skip_gap_zeroing_enabled,
            self._hole_assert_gaps_enabled,
            self._reuse_timer_force_sync_enabled,
        )

    def set_gap_buffer_semantics(
        self,
        mode: str,
        req_id: str | None = None,
        load_mode: str | None = None,
    ) -> None:
        if mode not in {"hole", "legacy"}:
            raise ValueError(f"Unsupported hole gap-buffer semantics mode: {mode}")
        self._gap_buffer_semantics = mode
        self._gap_buffer_req_id = req_id
        self._gap_buffer_load_mode = load_mode
        if trace_flow_enabled():
            trace_flow(
                "npu_connector.hole",
                "set_gap_buffer_semantics",
                mode=mode,
                req_id=req_id,
                load_mode=load_mode,
            )

    def _emit_hole_reuse_timer(
        self,
        bucket: str,
        *,
        layer_id: int,
        duration_ms: float,
    ) -> None:
        emit_timer(
            bucket,
            req_id=self._gap_buffer_req_id,
            layer_id=layer_id,
            duration_ms=duration_ms,
            path="hole",
            load_mode=self._gap_buffer_load_mode,
        )

    def _maybe_force_sync_reuse_timer(self, *, use_load_stream: bool = False) -> None:
        if not self._reuse_timer_force_sync_enabled:
            return
        if use_load_stream:
            self.load_stream.synchronize()
            return
        torch.cuda.synchronize()

    def _assert_gap_buffer_ready(
        self,
        *,
        layer_id: int,
        buffer_tensor: torch.Tensor | None,
        stage: str,
    ) -> None:
        assert_start = time.perf_counter()
        if (
            not self._hole_assert_gaps_enabled
            or buffer_tensor is None
            or self.current_gap_positions.numel() == 0
        ):
            return

        gap_k = buffer_tensor[0][self.current_gap_positions]
        gap_v = buffer_tensor[1][self.current_gap_positions]
        gap_k_all_zero = bool((gap_k == 0).all().item())
        gap_v_all_zero = bool((gap_v == 0).all().item())
        gap_k_all_finite = bool(torch.isfinite(gap_k).all().item())
        gap_v_all_finite = bool(torch.isfinite(gap_v).all().item())

        if (
            gap_k_all_finite
            and gap_v_all_finite
            and not (gap_k_all_zero and gap_v_all_zero)
        ):
            self._emit_hole_reuse_timer(
                "reuse_gap_assert",
                layer_id=layer_id,
                duration_ms=(time.perf_counter() - assert_start) * 1000.0,
            )
            return

        semantics = self._gap_buffer_semantics

        event = (
            "gap_buffer_assert_failure"
            if semantics == "hole"
            else "gap_buffer_legacy_warning"
        )
        if trace_flow_enabled():
            trace_flow(
                "npu_connector.hole",
                event,
                layer_id=layer_id,
                stage=stage,
                semantics=semantics,
                req_id=self._gap_buffer_req_id,
                gap_positions=self.current_gap_positions.tolist(),
                gap_k_stats=summarize_kv_tensor_stats(gap_k),
                gap_v_stats=summarize_kv_tensor_stats(gap_v),
                gap_k_all_zero=gap_k_all_zero,
                gap_v_all_zero=gap_v_all_zero,
                gap_k_all_finite=gap_k_all_finite,
                gap_v_all_finite=gap_v_all_finite,
            )
        if semantics == "legacy":
            self._emit_hole_reuse_timer(
                "reuse_gap_assert",
                layer_id=layer_id,
                duration_ms=(time.perf_counter() - assert_start) * 1000.0,
            )
            logger.error(
                "Legacy hole-connector gap slice stayed invalid before paged-KV "
                "handoff: req_id=%s layer=%d stage=%s gap_k_all_zero=%s "
                "gap_v_all_zero=%s gap_k_all_finite=%s gap_v_all_finite=%s "
                "gap_positions=%s",
                self._gap_buffer_req_id,
                layer_id,
                stage,
                gap_k_all_zero,
                gap_v_all_zero,
                gap_k_all_finite,
                gap_v_all_finite,
                self.current_gap_positions.tolist(),
            )
            return

        self._emit_hole_reuse_timer(
            "reuse_gap_assert",
            layer_id=layer_id,
            duration_ms=(time.perf_counter() - assert_start) * 1000.0,
        )
        raise AssertionError(
            "Hole buffer gap slice is invalid before paged-KV handoff: "
            f"req_id={self._gap_buffer_req_id} layer={layer_id} stage={stage} "
            f"gap_k_all_zero={gap_k_all_zero} gap_v_all_zero={gap_v_all_zero} "
            f"gap_k_all_finite={gap_k_all_finite} "
            f"gap_v_all_finite={gap_v_all_finite}"
        )

    def _wait_for_hole_loaded_buffer(self, layer_id: int) -> None:
        use_fine_grained = (
            self._hole_fine_grained_sync_enabled
            and not self._hole_fine_grained_sync_runtime_disabled
        )
        if use_fine_grained:
            try:
                sync_start = time.perf_counter()
                current_stream = torch.cuda.current_stream()
                current_stream.wait_stream(self.load_stream)
                if self._reuse_timer_force_sync_enabled:
                    current_stream.synchronize()
                self._emit_hole_reuse_timer(
                    "reuse_sync",
                    layer_id=layer_id,
                    duration_ms=(time.perf_counter() - sync_start) * 1000.0,
                )
                if trace_flow_enabled():
                    trace_flow(
                        "npu_connector.hole",
                        "sync_loaded_buffer",
                        layer_id=layer_id,
                        strategy="wait_stream",
                    )
                return
            except Exception as exc:
                self._hole_fine_grained_sync_runtime_disabled = True
                logger.exception(
                    "Hole fine-grained load sync failed at layer %d; "
                    "falling back to global synchronize for the rest of the run.",
                    layer_id,
                )
                if trace_flow_enabled():
                    trace_flow(
                        "npu_connector.hole",
                        "sync_loaded_buffer_failure",
                        layer_id=layer_id,
                        strategy="wait_stream",
                        fallback="global_synchronize",
                        error=repr(exc),
                    )

        sync_start = time.perf_counter()
        torch.cuda.synchronize()
        self._emit_hole_reuse_timer(
            "reuse_sync",
            layer_id=layer_id,
            duration_ms=(time.perf_counter() - sync_start) * 1000.0,
        )
        if trace_flow_enabled():
            trace_flow(
                "npu_connector.hole",
                "sync_loaded_buffer",
                layer_id=layer_id,
                strategy="global_synchronize",
                runtime_disabled=self._hole_fine_grained_sync_runtime_disabled,
            )

    def _record_load_buffer_ready(self, *, buffer_obj) -> None:
        if (
            not self._hole_fine_grained_sync_enabled
            or self._hole_fine_grained_sync_runtime_disabled
        ):
            return
        load_event = torch.npu.Event()
        load_event.record(self.load_stream)
        self._buffer_load_events[id(buffer_obj)] = load_event

    def _enqueue_flush_buffer(
        self,
        *,
        layer_id: int,
        buffer_obj,
        slot_mapping_full: torch.Tensor,
    ) -> None:
        self._assert_gap_buffer_ready(
            layer_id=layer_id,
            buffer_tensor=buffer_obj.tensor,
            stage="before_paged_kv_flush",
        )
        flush_start = time.perf_counter()
        compute_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.flush_stream):
            scatter_event = self._buffer_scatter_events.pop(layer_id, None)
            if scatter_event is not None:
                self.flush_stream.wait_event(scatter_event)
            else:
                # Empty-query layers do not run attention and therefore have no
                # completion event. Preserve ordering with the compute stream.
                self.flush_stream.wait_stream(compute_stream)
            lmc_ops.single_layer_kv_transfer(
                buffer_obj.tensor,
                self.kvcaches[layer_id],
                slot_mapping_full,
                False,
                self.kv_format.value,
                False,
                self.vllm_two_major,
            )
            flush_event = torch.npu.Event()
            flush_event.record(self.flush_stream)
        self._emit_hole_reuse_timer(
            "reuse_flush",
            layer_id=layer_id,
            duration_ms=(time.perf_counter() - flush_start) * 1000.0,
        )
        self._buffer_flush_events[id(buffer_obj)] = flush_event
        logger.debug("Enqueued loading hole layer %d into paged memory", layer_id)

    def _wait_hole_buffer_reusable(self, *, layer_id: int, buffer_obj) -> None:
        flush_event = self._buffer_flush_events.pop(id(buffer_obj), None)
        if flush_event is None:
            return
        wait_start = time.perf_counter()
        self.load_stream.wait_event(flush_event)
        if self._reuse_timer_force_sync_enabled:
            self.load_stream.synchronize()
        self._emit_hole_reuse_timer(
            "reuse_flush_reuse_wait",
            layer_id=layer_id,
            duration_ms=(time.perf_counter() - wait_start) * 1000.0,
        )

    def get_gap_positions(self) -> torch.Tensor:
        if self.current_gap_positions is None:
            return torch.empty((0,), dtype=torch.long)
        return self.current_gap_positions

    def get_hit_positions(self) -> torch.Tensor:
        return self.current_hit_positions

    def get_kv(self, layer_id: int):
        """Override: lazily wait for rope_stream to finish RoPE + gap zeroing
        before returning the buffer.  This lets model layernorm + qkv_proj
        overlap with rope/zero work on the rope_stream."""
        if layer_id not in self.buffer_mapping:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")
        buffer_obj = self.buffer_mapping[layer_id]
        rope_event = self._buffer_rope_events.pop(id(buffer_obj), None)
        if rope_event is not None:
            torch.cuda.current_stream().wait_event(rope_event)
        gpu_buffer = buffer_obj.tensor
        assert gpu_buffer is not None
        return gpu_buffer[0], gpu_buffer[1]

    def record_scatter_done(self, layer_id: int) -> None:
        """Record completion of attention reads from the layer buffer."""
        if layer_id not in self.buffer_mapping:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")
        event = torch.npu.Event()
        event.record(torch.cuda.current_stream())
        self._buffer_scatter_events[layer_id] = event

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """Store hole ranges while preserving their global token positions."""
        slot_mapping = self._prepare_transfer_context(kwargs)
        position_offset = int(kwargs.get("position_offset", 0))
        debug_chunk_tags = list(kwargs.get("debug_chunk_tags", []) or [])

        buf_start = 0
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(
                        start + position_offset,
                        end + position_offset,
                        device=self.kv_device,
                        dtype=torch.int64,
                    )
                )

        slot_mapping_full, num_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="concat"
        )
        if trace_flow_enabled():
            trace_flow(
                "npu_connector.hole",
                "batched_from_gpu_start",
                starts=starts,
                ends=ends,
                slot_mapping=summarize_slot_mapping(slot_mapping_full),
                num_tokens=num_tokens,
                position_offset=position_offset,
                cached_position_chunks=[
                    tensor_to_list(chunk, dtype=torch.long)
                    for chunk in old_positions_chunks
                ],
            )

        tmp_gpu_buffer_obj = self._allocate_gpu_buffers(num_tokens, count=1)
        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    True,
                    self.kv_format.value,
                    False,
                    self.vllm_two_major,
                )

                for chunk_idx, (
                    (chunk_start, chunk_end),
                    memory_obj,
                    old_positions,
                ) in enumerate(
                    zip(
                        buf_starts_ends,
                        memory_objs_layer,
                        old_positions_chunks,
                        strict=False,
                    )
                ):
                    chunk_tag = (
                        debug_chunk_tags[chunk_idx]
                        if chunk_idx < len(debug_chunk_tags)
                        else None
                    )
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][chunk_start:chunk_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][chunk_start:chunk_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions
                    if trace_flow_enabled() and trace_layer_enabled(layer_id):
                        trace_flow(
                            "npu_connector.hole",
                            "save_chunk_from_buffer",
                            layer_id=layer_id,
                            chunk_range=[chunk_start, chunk_end],
                            chunk_tag=chunk_tag,
                            cached_positions=tensor_to_list(
                                old_positions,
                                dtype=torch.long,
                            ),
                            k_stats=summarize_kv_tensor_stats(memory_obj.tensor[0]),
                            v_stats=summarize_kv_tensor_stats(memory_obj.tensor[1]),
                        )

            yield
            self.store_stream.synchronize()
            logger.debug("Finished offloading hole layer %d", layer_id)

        tmp_gpu_buffer_obj.ref_count_down()
        yield

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        slot_mapping = self._prepare_transfer_context(kwargs)
        debug_chunk_tags = list(kwargs.get("debug_chunk_tags", []) or [])

        self._buffer_flush_events.clear()
        self._buffer_load_events.clear()
        self._buffer_rope_events.clear()
        self._buffer_scatter_events.clear()

        prefix_start = kwargs.get("prefix_start", 0)
        prefix_end = kwargs["prefix_end"]

        if self.fused_rotary_emb is None and self.cache_positions:
            raise ValueError("fused_rotary_emb must be set before hole loading.")

        self.current_prefix_start = prefix_start
        self.current_prefix_end = prefix_end

        num_all_tokens = prefix_end - prefix_start
        slot_mapping_full = slot_mapping[prefix_start:prefix_end]

        gap_mask_start = time.perf_counter()
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        for start, end in zip(starts, ends, strict=False):
            local_start = start - prefix_start
            local_end = end - prefix_start
            gap_mask[local_start:local_end] = False

        self.current_gap_positions = torch.where(gap_mask)[0]
        self.current_hit_positions = torch.where(~gap_mask)[0]
        self._emit_hole_reuse_timer(
            "reuse_gap_mask",
            layer_id=0,
            duration_ms=(time.perf_counter() - gap_mask_start) * 1000.0,
        )
        skip_gap_zeroing_for_request = (
            self._hole_skip_gap_zeroing_enabled and self._gap_buffer_semantics == "hole"
        )
        if trace_flow_enabled():
            trace_flow(
                "npu_connector.hole",
                "batched_to_gpu_start",
                prefix_start=prefix_start,
                prefix_end=prefix_end,
                starts=starts,
                ends=ends,
                slot_mapping=summarize_slot_mapping(slot_mapping_full),
                gap_mask=mask_to_string(gap_mask),
                gap_positions=self.current_gap_positions.tolist(),
                hit_positions=self.current_hit_positions.tolist(),
                fine_grained_sync_enabled=self._hole_fine_grained_sync_enabled,
                skip_gap_zeroing_enabled=self._hole_skip_gap_zeroing_enabled,
                skip_gap_zeroing_for_request=skip_gap_zeroing_for_request,
                gap_semantics=self._gap_buffer_semantics,
                req_id=self._gap_buffer_req_id,
            )

        buffer_alloc_start = time.perf_counter()
        allocated_buffers = cast(
            list[MemoryObj],
            self._allocate_gpu_buffers(num_all_tokens, count=2),
        )
        compute_gpu_buffer_obj, load_gpu_buffer_obj = allocated_buffers
        self._emit_hole_reuse_timer(
            "reuse_buffer_alloc",
            layer_id=0,
            duration_ms=(time.perf_counter() - buffer_alloc_start) * 1000.0,
        )

        if self.cache_positions:
            new_positions_full = torch.arange(
                prefix_start, prefix_end, dtype=torch.int64, device=self.kv_device
            )
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kv_device
            )

        send_step_start: float | None = None
        send_step_timed_ms = 0.0
        send_step_layer_id: int | None = None

        def finalize_send_other() -> None:
            nonlocal send_step_start, send_step_timed_ms, send_step_layer_id
            if send_step_start is None or send_step_layer_id is None:
                return
            total_ms = (time.perf_counter() - send_step_start) * 1000.0
            other_ms = max(total_ms - send_step_timed_ms, 0.0)
            self._emit_hole_reuse_timer(
                "reuse_send_other",
                layer_id=send_step_layer_id,
                duration_ms=other_ms,
            )
            send_step_start = None
            send_step_timed_ms = 0.0
            send_step_layer_id = None

        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                flush_step_start = time.perf_counter()
                self._enqueue_flush_buffer(
                    layer_id=layer_id - 2,
                    buffer_obj=self.buffer_mapping[layer_id - 2],
                    slot_mapping_full=slot_mapping_full,
                )
                send_step_timed_ms += (time.perf_counter() - flush_step_start) * 1000.0
                del self.buffer_mapping[layer_id - 2]

            if layer_id > 0 and layer_id <= self.num_layers:
                # ── Phase 1: extract load dependency ──────────────────────
                # Instead of making current_stream wait for the load, we
                # defer the dependency to rope_stream so that model compute
                # (layernorm + qkv_proj) can start on current_stream
                # immediately after this send() returns.
                sync_step_start = time.perf_counter()
                load_event = None
                use_fine_grained = (
                    self._hole_fine_grained_sync_enabled
                    and not self._hole_fine_grained_sync_runtime_disabled
                )
                if use_fine_grained:
                    try:
                        load_event = self._buffer_load_events.pop(
                            id(load_gpu_buffer_obj), None
                        )
                        sync_start = time.perf_counter()
                        self._emit_hole_reuse_timer(
                            "reuse_sync",
                            layer_id=layer_id - 1,
                            duration_ms=(time.perf_counter() - sync_start) * 1000.0,
                        )
                        if trace_flow_enabled():
                            trace_flow(
                                "npu_connector.hole",
                                "sync_loaded_buffer",
                                layer_id=layer_id - 1,
                                strategy="deferred_to_rope_stream",
                                has_load_event=load_event is not None,
                            )
                    except Exception as exc:
                        self._hole_fine_grained_sync_runtime_disabled = True
                        use_fine_grained = False
                        load_event = None
                        logger.exception(
                            "Hole fine-grained load event sync failed at layer %d; "
                            "falling back to global synchronize for the rest "
                            "of the run.",
                            layer_id - 1,
                        )
                        if trace_flow_enabled():
                            trace_flow(
                                "npu_connector.hole",
                                "sync_loaded_buffer_failure",
                                layer_id=layer_id - 1,
                                strategy="deferred_to_rope_stream",
                                fallback="global_synchronize",
                                error=repr(exc),
                            )

                if not use_fine_grained:
                    # Fallback: global sync blocks host until all GPU work
                    # finishes; rope_stream needs no explicit wait afterwards.
                    sync_start = time.perf_counter()
                    torch.cuda.synchronize()
                    self._emit_hole_reuse_timer(
                        "reuse_sync",
                        layer_id=layer_id - 1,
                        duration_ms=(time.perf_counter() - sync_start) * 1000.0,
                    )
                    if trace_flow_enabled():
                        trace_flow(
                            "npu_connector.hole",
                            "sync_loaded_buffer",
                            layer_id=layer_id - 1,
                            strategy="global_synchronize",
                            runtime_disabled=self._hole_fine_grained_sync_runtime_disabled,
                        )

                send_step_timed_ms += (time.perf_counter() - sync_step_start) * 1000.0

                # ── Phase 2: swap buffers (Python reference swap only) ────
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                # ── Phase 3: RoPE + gap zeroing on rope_stream ───────────
                # These ops run on a dedicated stream so they overlap with
                # model compute on the default stream.  get_kv() will
                # lazily wait on the rope_ready event.
                rope_start = time.perf_counter()
                with torch.cuda.stream(self.rope_stream):
                    if load_event is not None:
                        self.rope_stream.wait_event(load_event)
                    elif use_fine_grained:
                        # Fine-grained enabled but no event — wait on stream
                        self.rope_stream.wait_stream(self.load_stream)
                    # else: global sync already ensured all streams idle

                    if self.cache_positions:
                        assert compute_gpu_buffer_obj.tensor is not None
                        compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                            old_positions_full,
                            new_positions_full,
                            compute_gpu_buffer_obj.tensor[0],
                        )

                    if (
                        self.current_gap_positions.numel()
                        and not skip_gap_zeroing_for_request
                    ):
                        compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = (
                            0.0
                        )

                    rope_ready = torch.npu.Event()
                    rope_ready.record(self.rope_stream)

                self._buffer_rope_events[id(compute_gpu_buffer_obj)] = rope_ready

                if self._reuse_timer_force_sync_enabled:
                    self.rope_stream.synchronize()
                rope_zero_ms = (time.perf_counter() - rope_start) * 1000.0
                if self.cache_positions:
                    self._emit_hole_reuse_timer(
                        "reuse_rope",
                        layer_id=layer_id - 1,
                        duration_ms=rope_zero_ms,
                    )
                if (
                    self.current_gap_positions.numel()
                    and not skip_gap_zeroing_for_request
                ):
                    self._emit_hole_reuse_timer(
                        "reuse_zero_gap",
                        layer_id=layer_id - 1,
                        duration_ms=rope_zero_ms,
                    )
                    if trace_flow_enabled():
                        trace_flow(
                            "npu_connector.hole",
                            "zero_gap_positions",
                            layer_id=layer_id - 1,
                            zeroed_positions=self.current_gap_positions.tolist(),
                            gap_semantics=self._gap_buffer_semantics,
                            req_id=self._gap_buffer_req_id,
                        )
                elif self.current_gap_positions.numel():
                    if trace_flow_enabled():
                        trace_flow(
                            "npu_connector.hole",
                            "skip_zero_gap_positions",
                            layer_id=layer_id - 1,
                            gap_positions=self.current_gap_positions.tolist(),
                            gap_semantics=self._gap_buffer_semantics,
                            req_id=self._gap_buffer_req_id,
                        )
                send_step_timed_ms += rope_zero_ms

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

            # RoPE is complete on the compute buffer; preload the next buffer.
            if layer_id < self.num_layers:
                finalize_send_other()
                memory_objs_layer = yield
                send_step_start = time.perf_counter()
                send_step_timed_ms = 0.0
                send_step_layer_id = layer_id
                reuse_wait_start = time.perf_counter()
                self._wait_hole_buffer_reusable(
                    layer_id=layer_id,
                    buffer_obj=load_gpu_buffer_obj,
                )
                send_step_timed_ms += (time.perf_counter() - reuse_wait_start) * 1000.0
                with torch.cuda.stream(self.load_stream):
                    copy_start = time.perf_counter()
                    for chunk_idx, (start, end, memory_obj) in enumerate(
                        zip(starts, ends, memory_objs_layer, strict=False)
                    ):
                        chunk_tag = (
                            debug_chunk_tags[chunk_idx]
                            if chunk_idx < len(debug_chunk_tags)
                            else None
                        )
                        local_start = start - prefix_start
                        local_end = end - prefix_start
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][local_start:local_end].copy_(
                            memory_obj.tensor[0], non_blocking=True
                        )
                        load_gpu_buffer_obj.tensor[1][local_start:local_end].copy_(
                            memory_obj.tensor[1], non_blocking=True
                        )
                        if trace_layer_enabled(layer_id):
                            trace_flow(
                                "npu_connector.hole",
                                "load_chunk_kv_stats",
                                layer_id=layer_id,
                                start=start,
                                end=end,
                                chunk_tag=chunk_tag,
                                cached_positions=tensor_to_list(
                                    memory_obj.metadata.cached_positions,
                                    dtype=torch.long,
                                ),
                                k_stats=summarize_kv_tensor_stats(memory_obj.tensor[0]),
                                v_stats=summarize_kv_tensor_stats(memory_obj.tensor[1]),
                            )
                        if self.cache_positions and layer_id == 0:
                            old_positions_full[local_start:local_end] = (
                                memory_obj.metadata.cached_positions
                            )
                            if trace_layer_enabled(layer_id):
                                trace_flow(
                                    "npu_connector.hole",
                                    "load_chunk_into_buffer",
                                    layer_id=layer_id,
                                    start=start,
                                    end=end,
                                    cached_positions=tensor_to_list(
                                        memory_obj.metadata.cached_positions,
                                        dtype=torch.long,
                                    ),
                                )
                    self._record_load_buffer_ready(buffer_obj=load_gpu_buffer_obj)
                    self._maybe_force_sync_reuse_timer(use_load_stream=True)
                    copy_ms = (time.perf_counter() - copy_start) * 1000.0
                    self._emit_hole_reuse_timer(
                        "reuse_copy_enqueue",
                        layer_id=layer_id,
                        duration_ms=copy_ms,
                    )
                    send_step_timed_ms += copy_ms
            elif layer_id == self.num_layers:
                finalize_send_other()
                yield

        self.rope_stream.synchronize()
        self.flush_stream.synchronize()
        self._buffer_flush_events.clear()
        self._buffer_load_events.clear()
        self._buffer_rope_events.clear()
        self._buffer_scatter_events.clear()
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()
        assert len(self.buffer_mapping) == 0
        yield
