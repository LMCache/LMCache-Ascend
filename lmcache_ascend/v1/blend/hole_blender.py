# SPDX-License-Identifier: Apache-2.0
"""
Hole-aware CacheBlend blender for non-contiguous prefix reuse.

This module is the companion to `lmcache_ascend/v1/blend/blender.py`, which
implements the standard contiguous-prefix CacheBlend path. The hole blender
operates on a covered region that can span cached segments before and after an
uncached gap, materializes fresh K/V for hole positions into the same dense
buffer as cached hit positions, and carries the resulting recompute set through
later layers.

See `docs/hole-feature-overview.md` for the maintainer-facing overview of how
this blender fits into the connector and worker flow.
"""

# Standard
from dataclasses import dataclass
from typing import Optional, Union
import os
import time

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.trace_utils import (
    mask_to_string,
    summarize_key,
    summarize_kv_tensor_stats,
    summarize_ranges,
    tensor_to_list,
    trace_flow,
    trace_flow_enabled,
)
import torch

# First Party
from lmcache_ascend.v1.blend.models.hole import infer_hole_model_from_vllm
from lmcache_ascend.v1.hole_segment_utils import HoleSegmentHelper
from lmcache_ascend.v1.npu_hole_connector import VLLMBufferLayerwiseNPUHoleConnector
from lmcache_ascend.v1.timer import emit_timer as emit_trace_timer

logger = init_logger(__name__)


def _env_enabled(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_enabled_any(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        if name in os.environ:
            return _env_enabled(name, default)
    return default


@dataclass
class HoleBlendMetadata:
    imp_indices: Optional[torch.Tensor] = None
    buffer_indices: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None

    def clean(self):
        self.imp_indices = None
        self.buffer_indices = None
        self.positions = None


class LMCBlenderHole:
    """
    Algorithmic core of hole-mode CacheBlend.

    At layer 0, this blender processes the full covered region (`R1 + H + R2`)
    and materializes fresh K/V for hole positions into a dense working buffer.
    At the check layer, it computes refreshed `diff_k` only on hit positions,
    because hole positions do not have cached `old_k` to compare against. The
    recompute set is then formed as `topK ∪ hole positions`, and later layers
    continue with that sparse selection while unaffected hit positions keep
    reusing cached K/V.

    The primary entry points are `blend()`, `blend_layer()`, and
    `process_qkv()`. See `docs/hole-feature-overview.md` section 4 for the
    request-flow description and the relationship to the standard nohole
    blender.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector: VLLMBufferLayerwiseNPUHoleConnector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector
        self.layerwise_model = infer_hole_model_from_vllm(vllm_model, self)
        self.num_layers = len(vllm_model.model.layers)
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )
        self.metadata = HoleBlendMetadata()
        self.segment_helper = HoleSegmentHelper(config, cache_engine.metadata)
        self.gpu_connector.fused_rotary_emb = self.layerwise_model.fused_rotary_emb
        trace_value = str(os.environ.get("LMCACHE_TRACE_BLEND", "0")).strip().lower()
        self.trace_blend = trace_value not in {"", "0", "false", "no", "off"}
        test_hole_value = str(os.environ.get("TEST_HOLE", "0")).strip().lower()
        self.test_hole = test_hole_value not in {"", "0", "false", "no", "off"}
        self._trace_tokens_cpu: Optional[torch.Tensor] = None
        self._trace_req_id: Optional[str] = None
        self._active_timer_req_id: Optional[str] = None
        self._active_timer_path: str = "hole"
        self._active_timer_load_mode: Optional[str] = None
        self._layer_topk_ms: dict[int, float] = {}
        self._layer_blend_accounted_ms: dict[int, float] = {}
        self._active_num_falses: Optional[int] = None
        self._active_test_hole_miss_start: Optional[int] = None
        self._active_test_hole_prefix_end: Optional[int] = None
        self._active_adaptive_topk_sections_abs: Optional[list[tuple[int, int]]] = None
        self._active_trace_gap_positions_local: Optional[list[Union[int, str]]] = None
        self._active_trace_gap_positions_abs: Optional[list[Union[int, str]]] = None
        self._active_trace_hit_positions_local: Optional[list[Union[int, str]]] = None
        self._active_trace_hit_positions_abs: Optional[list[Union[int, str]]] = None
        self._hole_assert_gaps_enabled = _env_enabled(
            "LMCACHE_HOLE_ASSERT_GAPS",
            default=False,
        )
        self._adaptive_topk_enabled = _env_enabled_any(
            ("adaptive_topk", "ADAPTIVE_TOPK"),
            default=False,
        )

    def get_last_topk_ms(self, layer_id: int) -> float:
        return float(self._layer_topk_ms.get(layer_id, 0.0))

    def get_accounted_blend_ms(self, layer_id: int) -> float:
        return float(self._layer_blend_accounted_ms.get(layer_id, 0.0))

    def emit_timer(
        self,
        bucket: str,
        layer_id: int,
        duration_ms: float,
    ) -> None:
        duration_ms = float(duration_ms)
        if bucket == "blend":
            duration_ms = max(
                duration_ms - self.get_accounted_blend_ms(layer_id),
                0.0,
            )
        elif bucket.startswith("blend_"):
            self._layer_blend_accounted_ms[layer_id] = (
                self._layer_blend_accounted_ms.get(layer_id, 0.0) + duration_ms
            )
        emit_trace_timer(
            bucket,
            req_id=self._active_timer_req_id,
            layer_id=layer_id,
            duration_ms=duration_ms,
            path=self._active_timer_path,
            load_mode=self._active_timer_load_mode,
        )

    def _count_num_falses(self, mask: Optional[torch.Tensor]) -> int:
        if mask is None:
            return 0
        return int(mask.numel()) - int(mask.sum().item())

    def _trace_position_list(
        self,
        positions: Optional[torch.Tensor],
        *,
        offset: int = 0,
        max_items: int = 128,
    ) -> Optional[list[Union[int, str]]]:
        if positions is None:
            return None
        if not isinstance(positions, torch.Tensor):
            return tensor_to_list(positions, dtype=torch.long, max_items=max_items)

        flat_positions = positions.reshape(-1)
        prefix_positions = (
            flat_positions[:max_items]
            .detach()
            .to(
                device="cpu",
                dtype=torch.long,
            )
        )
        if offset:
            prefix_positions = prefix_positions + int(offset)
        values: list[Union[int, str]] = prefix_positions.tolist()
        if flat_positions.numel() > max_items:
            values.append("...")
        return values

    def _cache_active_trace_positions(
        self,
        *,
        gap_positions: Optional[torch.Tensor] = None,
        hit_positions: Optional[torch.Tensor] = None,
        num_falses: int = 0,
    ) -> None:
        if not (self.trace_blend or trace_flow_enabled()):
            return
        if gap_positions is not None and self._active_trace_gap_positions_local is None:
            self._active_trace_gap_positions_local = self._trace_position_list(
                gap_positions
            )
            self._active_trace_gap_positions_abs = self._trace_position_list(
                gap_positions,
                offset=num_falses,
            )
        if hit_positions is not None and self._active_trace_hit_positions_local is None:
            self._active_trace_hit_positions_local = self._trace_position_list(
                hit_positions
            )
            self._active_trace_hit_positions_abs = self._trace_position_list(
                hit_positions,
                offset=num_falses,
            )

    def _get_test_hole_miss_start(self, load_spec) -> Optional[int]:
        prefix_miss_ranges = list(getattr(load_spec, "prefix_miss_ranges", []) or [])
        if not self.test_hole or len(prefix_miss_ranges) != 1:
            return None
        miss_start, _miss_end = prefix_miss_ranges[0]
        return int(miss_start)

    def _get_test_hole_prefix_end(self, load_spec) -> Optional[int]:
        miss_start = self._get_test_hole_miss_start(load_spec)
        if miss_start is None:
            return None

        hit_ranges = list(getattr(load_spec, "hit_ranges", []) or [])
        prefix_hit_ends = [
            int(end)
            for start, end in hit_ranges
            if int(start) < miss_start and int(end) <= miss_start
        ]
        if not prefix_hit_ends:
            return 0
        return max(prefix_hit_ends)

    def _get_effective_hit_ranges(
        self, load_spec
    ) -> tuple[list[tuple[int, int]], bool]:
        hit_ranges = list(getattr(load_spec, "hit_ranges", []) or [])
        return hit_ranges, self._get_test_hole_miss_start(load_spec) is not None

    def _get_adaptive_topk_sections(self, load_spec) -> list[tuple[int, int]]:
        covered_tokens = int(getattr(load_spec, "covered_tokens", 0) or 0)
        if covered_tokens <= 0:
            return []

        sections: list[tuple[int, int]] = []
        cursor = 0
        for miss_start, miss_end in list(
            getattr(load_spec, "prefix_miss_ranges", []) or []
        ):
            miss_start = int(miss_start)
            miss_end = int(miss_end)
            if cursor < miss_start:
                sections.append((cursor, miss_start))
            cursor = max(cursor, miss_end)

        if cursor < covered_tokens:
            sections.append((cursor, covered_tokens))
        return sections

    def _select_adaptive_top_positions(
        self,
        *,
        layer_id: int,
        candidate_positions: torch.Tensor,
        diff_k: torch.Tensor,
        ratio: float,
        num_falses: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int, list[dict[str, int]]]:
        section_summaries: list[dict[str, int]] = []
        if candidate_positions.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                0,
                section_summaries,
            )

        if (
            not self._adaptive_topk_enabled
            or not self._active_adaptive_topk_sections_abs
        ):
            topk_prepare_start = time.perf_counter()
            topk_num = int(candidate_positions.numel() * ratio)
            topk_num = max(topk_num, 1)
            topk_num = min(topk_num, candidate_positions.numel())
            self.emit_timer(
                "blend_topk_prepare",
                layer_id,
                (time.perf_counter() - topk_prepare_start) * 1000.0,
            )
            topk_kernel_start = time.perf_counter()
            rel_top = torch.topk(diff_k, k=topk_num).indices
            self.emit_timer(
                "blend_topk_kernel",
                layer_id,
                (time.perf_counter() - topk_kernel_start) * 1000.0,
            )
            topk_gather_start = time.perf_counter()
            top_positions = candidate_positions[rel_top]
            self.emit_timer(
                "blend_topk_gather",
                layer_id,
                (time.perf_counter() - topk_gather_start) * 1000.0,
            )
            topk_sort_start = time.perf_counter()
            top_positions, _ = torch.sort(top_positions)
            self.emit_timer(
                "blend_topk_sort",
                layer_id,
                (time.perf_counter() - topk_sort_start) * 1000.0,
            )
            return top_positions, int(topk_num), section_summaries

        selected_sections: list[torch.Tensor] = []
        total_topk = 0
        for abs_start, abs_end in self._active_adaptive_topk_sections_abs:
            local_start = max(int(abs_start) - int(num_falses), 0)
            local_end = max(int(abs_end) - int(num_falses), 0)
            if local_end <= local_start:
                continue

            topk_mask_start = time.perf_counter()
            section_lo = int(
                torch.searchsorted(
                    candidate_positions,
                    local_start,
                    right=False,
                ).item()
            )
            section_hi = int(
                torch.searchsorted(
                    candidate_positions,
                    local_end,
                    right=False,
                ).item()
            )
            self.emit_timer(
                "blend_topk_section_mask",
                layer_id,
                (time.perf_counter() - topk_mask_start) * 1000.0,
            )
            topk_gather_start = time.perf_counter()
            section_candidates = candidate_positions[section_lo:section_hi]
            self.emit_timer(
                "blend_topk_gather",
                layer_id,
                (time.perf_counter() - topk_gather_start) * 1000.0,
            )
            if section_candidates.numel() == 0:
                section_summaries.append(
                    {
                        "abs_start": int(abs_start),
                        "abs_end": int(abs_end),
                        "local_start": int(local_start),
                        "local_end": int(local_end),
                        "candidate_count": 0,
                        "topk_count": 0,
                    }
                )
                continue

            topk_gather_start = time.perf_counter()
            section_diff = diff_k[section_lo:section_hi]
            self.emit_timer(
                "blend_topk_gather",
                layer_id,
                (time.perf_counter() - topk_gather_start) * 1000.0,
            )
            topk_prepare_start = time.perf_counter()
            section_topk_num = int(section_candidates.numel() * ratio)
            section_topk_num = max(section_topk_num, 1)
            section_topk_num = min(section_topk_num, section_candidates.numel())
            self.emit_timer(
                "blend_topk_prepare",
                layer_id,
                (time.perf_counter() - topk_prepare_start) * 1000.0,
            )
            topk_kernel_start = time.perf_counter()
            rel_top = torch.topk(section_diff, k=section_topk_num).indices
            self.emit_timer(
                "blend_topk_kernel",
                layer_id,
                (time.perf_counter() - topk_kernel_start) * 1000.0,
            )
            topk_gather_start = time.perf_counter()
            section_top = section_candidates[rel_top]
            self.emit_timer(
                "blend_topk_gather",
                layer_id,
                (time.perf_counter() - topk_gather_start) * 1000.0,
            )
            selected_sections.append(section_top)
            total_topk += int(section_topk_num)
            section_summaries.append(
                {
                    "abs_start": int(abs_start),
                    "abs_end": int(abs_end),
                    "local_start": int(local_start),
                    "local_end": int(local_end),
                    "candidate_count": int(section_candidates.numel()),
                    "topk_count": int(section_topk_num),
                }
            )

        if not selected_sections:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                0,
                section_summaries,
            )

        top_positions = torch.cat(selected_sections)
        if top_positions.numel() > 1:
            topk_sort_start = time.perf_counter()
            top_positions, _ = torch.sort(top_positions)
            self.emit_timer(
                "blend_topk_sort",
                layer_id,
                (time.perf_counter() - topk_sort_start) * 1000.0,
            )
        return top_positions, int(total_topk), section_summaries

    def _get_test_hole_forced_gap_positions(
        self,
        gap_positions: torch.Tensor,
        num_falses: int,
    ) -> torch.Tensor:
        prefix_end = self._active_test_hole_prefix_end
        if prefix_end is None:
            return gap_positions
        local_prefix_end = max(int(prefix_end) - int(num_falses), 0)
        forced_gap_start = local_prefix_end
        if self._active_test_hole_miss_start is not None:
            # In TEST_HOLE, match legacy/nohole more closely at the boundary by
            # not force-recomputing the separator right after the reusable
            # prefix. Those separator slots remain whatever the connector made
            # them, which is closer to the legacy contiguous-prefix behavior.
            forced_gap_start = min(
                local_prefix_end + int(self.segment_helper.sep_len),
                gap_positions.numel() + local_prefix_end,
            )
        return gap_positions[gap_positions >= forced_gap_start]

    def _log_recomputed_tokens(
        self,
        layer_id: int,
        recompute_positions: torch.Tensor,
        hit_positions: torch.Tensor,
        gap_positions: torch.Tensor,
        top_hit_positions: torch.Tensor,
    ) -> None:
        if not self.trace_blend:
            return
        if self._trace_tokens_cpu is None:
            return
        recompute_trace = self._trace_position_list(recompute_positions)
        hit_trace = (
            self._active_trace_hit_positions_abs
            if self._active_trace_hit_positions_abs is not None
            else self._trace_position_list(hit_positions)
        )
        gap_trace = (
            self._active_trace_gap_positions_abs
            if self._active_trace_gap_positions_abs is not None
            else self._trace_position_list(gap_positions)
        )
        top_hit_trace = self._trace_position_list(top_hit_positions)
        token_ids = [
            int(self._trace_tokens_cpu[pos].item())
            for pos in (recompute_trace or [])
            if pos != "..."
            if 0 <= int(pos) < int(self._trace_tokens_cpu.shape[0])
        ]
        logger.debug(
            "Hole blend recompute layer=%d req_id=%s hit_positions=%s "
            "gap_positions=%s top_hit_positions=%s recomputed_positions=%s "
            "recomputed_token_ids=%s",
            layer_id,
            self._trace_req_id if self._trace_req_id is not None else "unknown",
            hit_trace,
            gap_trace,
            top_hit_trace,
            recompute_trace,
            token_ids[:50],
        )
        logger.debug(
            "#recomp_ids=%d #hit_positions=%d #gap_positions=%d",
            int(recompute_positions.numel()),
            int(hit_positions.numel()),
            int(gap_positions.numel()),
        )
        if trace_flow_enabled():
            trace_flow(
                "blender.hole",
                "recompute_positions",
                layer_id=layer_id,
                req_id=self._trace_req_id,
                hit_positions=hit_trace,
                gap_positions=gap_trace,
                top_hit_positions=top_hit_trace,
                recomputed_positions=recompute_trace,
                recomputed_token_ids=token_ids,
            )

    def _materialize_gap_positions(
        self,
        layer_id: int,
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gap_positions: torch.Tensor,
        num_falses: int,
        req_id: Optional[Union[str, int]],
    ) -> None:
        if gap_positions.numel() == 0:
            return

        local_k = k[num_falses:]
        local_v = v[num_falses:]
        if old_k.shape[0] != local_k.shape[0] or old_v.shape[0] != local_v.shape[0]:
            raise ValueError(
                "Hole buffer shape mismatch while materializing gap positions: "
                f"old_k={tuple(old_k.shape)} local_k={tuple(local_k.shape)} "
                f"old_v={tuple(old_v.shape)} local_v={tuple(local_v.shape)}"
            )

        old_k[gap_positions] = local_k[gap_positions]
        old_v[gap_positions] = local_v[gap_positions]
        if trace_flow_enabled():
            gap_positions_trace = (
                self._active_trace_gap_positions_local
                if self._active_trace_gap_positions_local is not None
                else self._trace_position_list(gap_positions)
            )
            trace_flow(
                "blender.hole",
                "materialize_gap_positions",
                layer_id=layer_id,
                req_id=req_id,
                num_falses=num_falses,
                gap_positions=gap_positions_trace,
            )
        self._assert_gap_materialization_matches_full_source(
            stage="materialize_gap_positions",
            layer_id=layer_id,
            req_id=req_id,
            old_k=old_k,
            old_v=old_v,
            local_k=local_k,
            local_v=local_v,
            gap_positions=gap_positions,
            num_falses=num_falses,
        )

    def _raise_gap_assertion(
        self,
        *,
        stage: str,
        layer_id: int,
        req_id: Optional[Union[str, int]],
        message: str,
        **fields,
    ) -> None:
        trace_flow(
            "blender.hole",
            "gap_assert_failure",
            stage=stage,
            layer_id=layer_id,
            req_id=req_id,
            **fields,
        )
        raise AssertionError(
            f"{message} at stage={stage} layer={layer_id} req_id={req_id}"
        )

    def _assert_gap_materialization_matches_full_source(
        self,
        *,
        stage: str,
        layer_id: int,
        req_id: Optional[Union[str, int]],
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        gap_positions: torch.Tensor,
        num_falses: int,
    ) -> None:
        if not self._hole_assert_gaps_enabled or gap_positions.numel() == 0:
            return

        actual_k = old_k[gap_positions]
        expected_k = local_k[gap_positions]
        actual_v = old_v[gap_positions]
        expected_v = local_v[gap_positions]
        if torch.equal(actual_k, expected_k) and torch.equal(actual_v, expected_v):
            return

        mismatch_mask = (actual_k != expected_k).reshape(actual_k.shape[0], -1).any(
            dim=1
        ) | (actual_v != expected_v).reshape(actual_v.shape[0], -1).any(dim=1)
        mismatch_local = gap_positions[mismatch_mask]
        mismatch_abs = mismatch_local + num_falses
        self._raise_gap_assertion(
            stage=stage,
            layer_id=layer_id,
            req_id=req_id,
            message="Hole gap materialization does not match full-source KV",
            gap_positions=tensor_to_list(gap_positions, dtype=torch.long),
            mismatch_local=tensor_to_list(mismatch_local, dtype=torch.long),
            mismatch_abs=tensor_to_list(mismatch_abs, dtype=torch.long),
            actual_gap_k_stats=summarize_kv_tensor_stats(actual_k),
            expected_gap_k_stats=summarize_kv_tensor_stats(expected_k),
            actual_gap_v_stats=summarize_kv_tensor_stats(actual_v),
            expected_gap_v_stats=summarize_kv_tensor_stats(expected_v),
        )

    def _assert_gap_positions_covered(
        self,
        *,
        stage: str,
        layer_id: int,
        req_id: Optional[Union[str, int]],
        gap_positions: torch.Tensor,
        buffer_indices: torch.Tensor | None,
        num_falses: int,
    ) -> None:
        if (
            not self._hole_assert_gaps_enabled
            or buffer_indices is None
            or gap_positions.numel() == 0
        ):
            return

        missing_mask = ~torch.isin(gap_positions, buffer_indices)
        if not missing_mask.any():
            return

        missing_local = gap_positions[missing_mask]
        missing_abs = missing_local + num_falses
        self._raise_gap_assertion(
            stage=stage,
            layer_id=layer_id,
            req_id=req_id,
            message="Hole dense-buffer path is missing gap positions in buffer_indices",
            buffer_indices=tensor_to_list(buffer_indices, dtype=torch.long),
            gap_positions=tensor_to_list(gap_positions, dtype=torch.long),
            missing_local=tensor_to_list(missing_local, dtype=torch.long),
            missing_abs=tensor_to_list(missing_abs, dtype=torch.long),
        )

    def _assert_gap_materialization_matches_sparse_source(
        self,
        *,
        stage: str,
        layer_id: int,
        req_id: Optional[Union[str, int]],
        old_k: torch.Tensor,
        old_v: torch.Tensor,
        sparse_k: torch.Tensor,
        sparse_v: torch.Tensor,
        gap_positions: torch.Tensor,
        buffer_indices: torch.Tensor,
        num_falses: int,
    ) -> None:
        if not self._hole_assert_gaps_enabled or gap_positions.numel() == 0:
            return

        gap_sparse_mask = torch.isin(buffer_indices, gap_positions)
        if not gap_sparse_mask.any():
            self._raise_gap_assertion(
                stage=stage,
                layer_id=layer_id,
                req_id=req_id,
                message="Hole dense-buffer path has no sparse rows for gap positions",
                gap_positions=tensor_to_list(gap_positions, dtype=torch.long),
                buffer_indices=tensor_to_list(buffer_indices, dtype=torch.long),
            )

        dense_gap_positions = buffer_indices[gap_sparse_mask]
        actual_k = old_k[dense_gap_positions]
        expected_k = sparse_k[gap_sparse_mask]
        actual_v = old_v[dense_gap_positions]
        expected_v = sparse_v[gap_sparse_mask]
        if torch.equal(actual_k, expected_k) and torch.equal(actual_v, expected_v):
            return

        mismatch_mask = (actual_k != expected_k).reshape(actual_k.shape[0], -1).any(
            dim=1
        ) | (actual_v != expected_v).reshape(actual_v.shape[0], -1).any(dim=1)
        mismatch_local = dense_gap_positions[mismatch_mask]
        mismatch_abs = mismatch_local + num_falses
        self._raise_gap_assertion(
            stage=stage,
            layer_id=layer_id,
            req_id=req_id,
            message="Hole dense-buffer gap KV does not match sparse recompute source",
            gap_positions=tensor_to_list(gap_positions, dtype=torch.long),
            buffer_indices=tensor_to_list(buffer_indices, dtype=torch.long),
            mismatch_local=tensor_to_list(mismatch_local, dtype=torch.long),
            mismatch_abs=tensor_to_list(mismatch_abs, dtype=torch.long),
            actual_gap_k_stats=summarize_kv_tensor_stats(actual_k),
            expected_gap_k_stats=summarize_kv_tensor_stats(expected_k),
            actual_gap_v_stats=summarize_kv_tensor_stats(actual_v),
            expected_gap_v_stats=summarize_kv_tensor_stats(expected_v),
        )

    def _assert_skip_zero_safe(
        self,
        layer_id: int,
        gap_positions: torch.Tensor,
        num_falses: int,
        req_id: Optional[Union[str, int]],
    ) -> None:
        if not getattr(self.gpu_connector, "_hole_skip_gap_zeroing_enabled", False):
            return
        self._assert_gap_positions_covered(
            stage="skip_zero_safe",
            layer_id=layer_id,
            req_id=req_id,
            gap_positions=gap_positions,
            buffer_indices=self.metadata.buffer_indices,
            num_falses=num_falses,
        )

    def _sparse_retrieve_layer(
        self,
        tokens: torch.Tensor,
        load_spec,
        **kwargs,
    ):
        starts = []
        ends = []
        keys = []
        chunk_tags = []
        source_offsets = []
        request_configs = kwargs.get("request_configs")
        prefix_start = kwargs.get("prefix_start", 0)
        effective_hit_ranges = list(
            kwargs.get("effective_hit_ranges") or load_spec.hit_ranges
        )
        plan_lookup_start = time.perf_counter()

        if self.cache_engine.storage_manager is None:
            raise ValueError("storage_manager is required for hole retrieve")

        for start, end in effective_hit_ranges:
            if end <= prefix_start:
                continue
            adj_start = max(start, prefix_start)
            adj_end = end
            if adj_end <= adj_start:
                continue
            key = self.segment_helper.make_cache_key(
                tokens,
                (start, end),
                request_configs,
            )
            trace_flow(
                "blender.hole",
                "sparse_hit_range",
                req_id=kwargs.get("req_id"),
                original_range=[start, end],
                adjusted_range=[adj_start, adj_end],
                source_offset=adj_start - start,
                prefix_start=prefix_start,
                key=summarize_key(key),
            )
            starts.append(adj_start)
            ends.append(adj_end)
            keys.append(key.split_layers(self.cache_engine.num_layers))
            chunk_tags.append(summarize_key(key))
            source_offsets.append(adj_start - start)

        if not keys:
            self.emit_timer(
                "reuse_plan_lookup",
                0,
                (time.perf_counter() - plan_lookup_start) * 1000.0,
            )
            trace_flow(
                "blender.hole",
                "sparse_retrieve_empty",
                req_id=kwargs.get("req_id"),
                covered_tokens=load_spec.covered_tokens,
                hit_ranges=summarize_ranges(effective_hit_ranges),
            )
            for _ in range(self.num_layers):
                yield None
            yield None
            yield torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
            return

        location = getattr(load_spec, "location", None)
        if location is None:
            location = self.cache_engine.storage_manager.contains(keys[0][0])
        if location is None:
            raise ValueError("Unable to resolve storage location for hole retrieval.")
        self.emit_timer(
            "reuse_plan_lookup",
            0,
            (time.perf_counter() - plan_lookup_start) * 1000.0,
        )
        keys_layer_major = [list(row) for row in zip(*keys, strict=False)]
        get_generator = self.cache_engine.storage_manager.layerwise_batched_get(
            keys_layer_major,
            location=location,
        )

        mem_obj_consumer = self.gpu_connector.batched_to_gpu(
            starts,
            ends,
            prefix_end=load_spec.covered_tokens,
            debug_chunk_tags=chunk_tags,
            source_offsets=source_offsets,
            **kwargs,
        )
        consumer_prime_start = time.perf_counter()
        next(mem_obj_consumer)
        self.emit_timer(
            "reuse_consumer_prime",
            0,
            (time.perf_counter() - consumer_prime_start) * 1000.0,
        )

        to_count_down = []
        for _layer_id in range(self.num_layers):
            task_next_start = time.perf_counter()
            task = next(get_generator)
            self.emit_timer(
                "reuse_task_next",
                _layer_id,
                (time.perf_counter() - task_next_start) * 1000.0,
            )
            yield None
            storage_wait_start = time.perf_counter()
            mem_objs_layer = task.result()
            self.emit_timer(
                "reuse_storage_wait",
                _layer_id,
                (time.perf_counter() - storage_wait_start) * 1000.0,
            )
            consumer_send_start = time.perf_counter()
            mem_obj_consumer.send(mem_objs_layer)
            self.emit_timer(
                "reuse_consumer_send",
                _layer_id,
                (time.perf_counter() - consumer_send_start) * 1000.0,
            )
            to_count_down.extend(mem_objs_layer)

        ref_count_down_start = time.perf_counter()
        for mem_obj in to_count_down:
            mem_obj.ref_count_down()
        self.emit_timer(
            "reuse_ref_count_down",
            self.num_layers - 1,
            (time.perf_counter() - ref_count_down_start) * 1000.0,
        )

        yield None
        consumer_finalize_start = time.perf_counter()
        next(mem_obj_consumer)
        self.emit_timer(
            "reuse_consumer_finalize",
            self.num_layers - 1,
            (time.perf_counter() - consumer_finalize_start) * 1000.0,
        )
        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
        for start, end in effective_hit_ranges:
            ret_mask[start:end] = True
        trace_flow(
            "blender.hole",
            "sparse_retrieve_finish",
            req_id=kwargs.get("req_id"),
            ranges=summarize_ranges(effective_hit_ranges),
            ret_mask=mask_to_string(ret_mask),
        )
        yield ret_mask

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._layer_topk_ms[layer_id] = 0.0
        self._layer_blend_accounted_ms[layer_id] = 0.0
        kv_fetch_start = time.perf_counter()
        old_k, old_v = self.gpu_connector.get_kv(layer_id)
        self.emit_timer(
            "blend_kv_fetch",
            layer_id,
            (time.perf_counter() - kv_fetch_start) * 1000.0,
        )
        num_falses = (
            self._active_num_falses
            if self._active_num_falses is not None
            else self._count_num_falses(mask)
        )

        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        positions_init_start = time.perf_counter()
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        self.emit_timer(
            "blend_positions_init",
            layer_id,
            (time.perf_counter() - positions_init_start) * 1000.0,
        )

        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        qk_post_start = time.perf_counter()
        if "qk_post_processing" in kwargs:
            q, k = kwargs["qk_post_processing"](
                q, k, attn_layer, self.metadata.positions
            )
        else:
            q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)
        self.emit_timer(
            "blend_qk_post",
            layer_id,
            (time.perf_counter() - qk_post_start) * 1000.0,
        )

        gap_positions_start = time.perf_counter()
        gap_positions = self.gpu_connector.get_gap_positions().to(q.device)
        forced_gap_positions = self._get_test_hole_forced_gap_positions(
            gap_positions,
            num_falses,
        )
        self._cache_active_trace_positions(
            gap_positions=gap_positions,
            num_falses=num_falses,
        )
        self.emit_timer(
            "blend_gap_positions",
            layer_id,
            (time.perf_counter() - gap_positions_start) * 1000.0,
        )

        if layer_id in self.common_metadata.check_layers:
            hit_positions_start = time.perf_counter()
            hit_positions = self.gpu_connector.get_hit_positions().to(q.device)
            self._cache_active_trace_positions(
                hit_positions=hit_positions,
                num_falses=num_falses,
            )
            self.emit_timer(
                "blend_hit_positions",
                layer_id,
                (time.perf_counter() - hit_positions_start) * 1000.0,
            )
            topk_candidate_positions = hit_positions
            test_hole_miss_start = self._active_test_hole_miss_start
            if test_hole_miss_start is not None:
                prefix_end = self._active_test_hole_prefix_end
                if prefix_end is None:
                    topk_candidate_positions = torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=q.device,
                    )
                else:
                    local_prefix_end = max(int(prefix_end) - int(num_falses), 0)
                    topk_candidate_positions = torch.arange(
                        local_prefix_end,
                        device=q.device,
                        dtype=torch.long,
                    )

            top_positions = torch.empty((0,), dtype=torch.long, device=q.device)
            ratio = 0.0
            topk_num = 0
            adaptive_topk_sections: list[dict[str, int]] = []
            if self.common_metadata.recomp_ratios:
                ratio = float(self.common_metadata.recomp_ratios[0])
            if ratio > 0 and topk_candidate_positions.numel() > 0:
                topk_start = time.perf_counter()
                topk_diff_start = time.perf_counter()
                diff_k = torch.sum(
                    (
                        k[num_falses:][topk_candidate_positions].to(torch.float32)
                        - old_k[topk_candidate_positions].to(torch.float32)
                    )
                    ** 2,
                    dim=[1],
                )
                self.emit_timer(
                    "blend_topk_diff",
                    layer_id,
                    (time.perf_counter() - topk_diff_start) * 1000.0,
                )
                topk_select_start = time.perf_counter()
                (
                    top_positions,
                    topk_num,
                    adaptive_topk_sections,
                ) = self._select_adaptive_top_positions(
                    layer_id=layer_id,
                    candidate_positions=topk_candidate_positions,
                    diff_k=diff_k,
                    ratio=ratio,
                    num_falses=num_falses,
                    device=q.device,
                )
                self.emit_timer(
                    "blend_topk_select",
                    layer_id,
                    (time.perf_counter() - topk_select_start) * 1000.0,
                )
                topk_ms = (time.perf_counter() - topk_start) * 1000.0
                self._layer_topk_ms[layer_id] = topk_ms
                self.emit_timer("topk_l1", layer_id, topk_ms)

            logger.debug(
                "old_k.shape = %s num_falses (F mask) = %d hits = %d "
                "topk_candidates = %d "
                "forced_gaps = %d total_gaps = %d ratio=%f top_k_in_prefix %d "
                "test_hole_miss_start=%s test_hole_prefix_end=%s adaptive_topk=%s "
                "adaptive_sections=%s",
                old_k.shape,
                num_falses,
                hit_positions.numel(),
                topk_candidate_positions.numel(),
                forced_gap_positions.numel(),
                gap_positions.numel(),
                ratio,
                topk_num,
                test_hole_miss_start,
                self._active_test_hole_prefix_end,
                self._adaptive_topk_enabled,
                adaptive_topk_sections,
            )
            recompute_local = torch.cat((top_positions, forced_gap_positions))
            if recompute_local.numel() > 1:
                recompute_sort_start = time.perf_counter()
                # In normal hole mode, top_positions is chosen from hit_positions,
                # and in TEST_HOLE exact-prefix mode it is chosen from the legacy
                # contiguous prefix [0, prefix_end). forced_gap_positions is always
                # taken from the suffix starting at prefix_end, so the two sets are
                # disjoint. We still need the final recompute list to be globally
                # sorted for downstream query ordering, but we do not need the
                # dedup work from torch.unique(..., sorted=True).
                recompute_local, _ = torch.sort(recompute_local)
                self.emit_timer(
                    "blend_recompute_sort",
                    layer_id,
                    (time.perf_counter() - recompute_sort_start) * 1000.0,
                )

            recompute_abs = recompute_local + num_falses
            self._log_recomputed_tokens(
                layer_id,
                recompute_abs,
                hit_positions + num_falses,
                forced_gap_positions + num_falses,
                top_positions + num_falses,
            )
            if trace_flow_enabled():
                hit_positions_trace = (
                    self._active_trace_hit_positions_local
                    if self._active_trace_hit_positions_local is not None
                    else self._trace_position_list(hit_positions)
                )
                gap_positions_trace = (
                    self._active_trace_gap_positions_local
                    if self._active_trace_gap_positions_local is not None
                    else self._trace_position_list(gap_positions)
                )
                topk_candidate_positions_trace = (
                    hit_positions_trace
                    if self._active_test_hole_miss_start is None
                    else self._trace_position_list(topk_candidate_positions)
                )
                forced_gap_positions_trace = (
                    gap_positions_trace
                    if self._active_test_hole_prefix_end is None
                    else self._trace_position_list(forced_gap_positions)
                )
                top_positions_trace = self._trace_position_list(top_positions)
                recompute_local_trace = self._trace_position_list(recompute_local)
                recompute_abs_trace = self._trace_position_list(recompute_abs)
                trace_flow(
                    "blender.hole",
                    "process_qkv",
                    layer_id=layer_id,
                    req_id=kwargs.get("req_id", self._trace_req_id),
                    num_falses=num_falses,
                    hit_positions=hit_positions_trace,
                    topk_candidate_positions=topk_candidate_positions_trace,
                    topk_candidate_hit_positions=topk_candidate_positions_trace,
                    gap_positions=gap_positions_trace,
                    forced_gap_positions=forced_gap_positions_trace,
                    top_prefix_positions=top_positions_trace,
                    top_hit_positions=top_positions_trace,
                    recompute_local=recompute_local_trace,
                    recompute_abs=recompute_abs_trace,
                    test_hole_miss_start=test_hole_miss_start,
                    test_hole_prefix_end=self._active_test_hole_prefix_end,
                    adaptive_topk_enabled=self._adaptive_topk_enabled,
                    adaptive_topk_sections=adaptive_topk_sections,
                )

            self.metadata.buffer_indices = recompute_local
            self.metadata.imp_indices = recompute_abs
            self.metadata.positions = recompute_abs

            recompute_gather_start = time.perf_counter()
            q = q[recompute_abs]
            k = k[recompute_abs]
            v = v[recompute_abs]
            residual = residual[recompute_abs]
            attn_output = attn_output[: len(recompute_abs)]
            self.emit_timer(
                "blend_recompute_gather",
                layer_id,
                (time.perf_counter() - recompute_gather_start) * 1000.0,
            )
            attn_metadata_update_start = time.perf_counter()
            attn_metadata.update_from_top_indices(recompute_abs)
            if hasattr(attn_metadata, "max_query_len"):
                attn_metadata.max_query_len = len(recompute_abs)
            if hasattr(attn_metadata, "query_start_loc"):
                attn_metadata.query_start_loc = torch.tensor(
                    [0, len(recompute_abs)],
                    dtype=torch.int32,
                    device=q.device,
                )
            self.emit_timer(
                "blend_attn_metadata",
                layer_id,
                (time.perf_counter() - attn_metadata_update_start) * 1000.0,
            )

        if self.metadata.buffer_indices is None:
            gap_materialize_start = time.perf_counter()
            self._materialize_gap_positions(
                layer_id=layer_id,
                old_k=old_k,
                old_v=old_v,
                k=k,
                v=v,
                gap_positions=forced_gap_positions,
                num_falses=num_falses,
                req_id=kwargs.get("req_id", self._trace_req_id),
            )
            self.emit_timer(
                "blend_gap_materialize",
                layer_id,
                (time.perf_counter() - gap_materialize_start) * 1000.0,
            )

        if self.metadata.buffer_indices is not None:
            self._assert_gap_positions_covered(
                stage="dense_buffer_before_scatter",
                layer_id=layer_id,
                req_id=kwargs.get("req_id", self._trace_req_id),
                gap_positions=forced_gap_positions,
                buffer_indices=self.metadata.buffer_indices,
                num_falses=num_falses,
            )
            self._assert_skip_zero_safe(
                layer_id=layer_id,
                gap_positions=forced_gap_positions,
                num_falses=num_falses,
                req_id=kwargs.get("req_id", self._trace_req_id),
            )
            gap_scatter_start = time.perf_counter()
            old_k[self.metadata.buffer_indices] = k
            old_v[self.metadata.buffer_indices] = v
            self.emit_timer(
                "blend_gap_scatter",
                layer_id,
                (time.perf_counter() - gap_scatter_start) * 1000.0,
            )
            self._assert_gap_materialization_matches_sparse_source(
                stage="dense_buffer_after_scatter",
                layer_id=layer_id,
                req_id=kwargs.get("req_id", self._trace_req_id),
                old_k=old_k,
                old_v=old_v,
                sparse_k=k,
                sparse_v=v,
                gap_positions=forced_gap_positions,
                buffer_indices=self.metadata.buffer_indices,
                num_falses=num_falses,
            )
            return q, old_k, old_v, residual, attn_output, attn_metadata
        return q, k, v, residual, attn_output, attn_metadata

    def blend_layer(
        self,
        tokens: torch.Tensor,
        load_spec,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        layerwise_model_executor = self.layerwise_model.compute_layer(
            tokens,
            mask,
        )
        layerwise_retriever = self._sparse_retrieve_layer(
            tokens,
            load_spec,
            **kwargs,
        )

        next(layerwise_retriever)
        yield

        for _ in range(self.num_layers):
            layer_id = _
            wait_start = time.perf_counter()
            next(layerwise_retriever)
            self.emit_timer(
                "wait_reuse",
                layer_id,
                (time.perf_counter() - wait_start) * 1000.0,
            )
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)
        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        load_spec,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).npu()
        effective_hit_ranges, test_hole_active = self._get_effective_hit_ranges(
            load_spec
        )
        test_hole_miss_start = self._get_test_hole_miss_start(load_spec)
        test_hole_prefix_end = self._get_test_hole_prefix_end(load_spec)
        self._active_test_hole_miss_start = test_hole_miss_start
        self._active_test_hole_prefix_end = test_hole_prefix_end
        self._active_adaptive_topk_sections_abs = self._get_adaptive_topk_sections(
            load_spec
        )
        if test_hole_active:
            logger.warning(
                "TEST_HOLE active for req_id=%s: single prefix miss=%s, "
                "keeping hole retrieval unchanged but matching legacy prefix "
                "behavior before miss_start=%s using legacy_prefix_end=%s; "
                "top-k candidates are contiguous prefix positions [0,%s) and "
                "forced gaps start at that boundary; hit_ranges=%s "
                "(covered_tokens=%d)",
                kwargs.get("req_id"),
                getattr(load_spec, "prefix_miss_ranges", None),
                test_hole_miss_start,
                test_hole_prefix_end,
                test_hole_prefix_end,
                getattr(load_spec, "hit_ranges", None),
                getattr(load_spec, "covered_tokens", None),
            )
            trace_flow(
                "blender.hole",
                "test_hole_single_miss",
                req_id=kwargs.get("req_id"),
                covered_tokens=getattr(load_spec, "covered_tokens", None),
                original_hit_ranges=summarize_ranges(
                    getattr(load_spec, "hit_ranges", None)
                ),
                effective_hit_ranges=summarize_ranges(effective_hit_ranges),
                test_hole_miss_start=test_hole_miss_start,
                test_hole_prefix_end=test_hole_prefix_end,
                prefix_miss_ranges=summarize_ranges(
                    getattr(load_spec, "prefix_miss_ranges", None)
                ),
            )
        trace_flow(
            "blender.hole",
            "blend_start",
            req_id=kwargs.get("req_id"),
            token_count=len(tokens),
            mask=mask_to_string(mask),
            token_ids=tensor_to_list(tokens, dtype=torch.long),
            mode=getattr(load_spec, "mode", None),
            covered_tokens=getattr(load_spec, "covered_tokens", None),
            hit_ranges=summarize_ranges(getattr(load_spec, "hit_ranges", None)),
            effective_hit_ranges=summarize_ranges(effective_hit_ranges),
            test_hole_active=test_hole_active,
            test_hole_miss_start=test_hole_miss_start,
            test_hole_prefix_end=test_hole_prefix_end,
            adaptive_topk_enabled=self._adaptive_topk_enabled,
            adaptive_topk_sections=summarize_ranges(
                self._active_adaptive_topk_sections_abs
            ),
            prefix_miss_ranges=summarize_ranges(
                getattr(load_spec, "prefix_miss_ranges", None)
            ),
        )
        if self.trace_blend:
            self._trace_tokens_cpu = tokens.detach().to(device="cpu", dtype=torch.long)
            req_id = kwargs.get("req_id")
            self._trace_req_id = None if req_id is None else str(req_id)
        req_id = kwargs.get("req_id")
        self._active_timer_req_id = None if req_id is None else str(req_id)
        self._active_timer_path = str(kwargs.get("timer_path", "hole"))
        timer_load_mode = kwargs.get(
            "timer_load_mode", getattr(load_spec, "mode", None)
        )
        self._active_timer_load_mode = (
            None if timer_load_mode is None else str(timer_load_mode)
        )
        self._active_num_falses = self._count_num_falses(mask)
        layerwise_blender = self.blend_layer(
            tokens,
            load_spec,
            mask,
            effective_hit_ranges=effective_hit_ranges,
            **kwargs,
        )
        try:
            for _ in range(self.num_layers + 2):
                next(layerwise_blender)
        finally:
            self._trace_tokens_cpu = None
            self._trace_req_id = None
            self._active_timer_req_id = None
            self._active_timer_path = "hole"
            self._active_timer_load_mode = None
            self._active_num_falses = None
            self._active_test_hole_miss_start = None
            self._active_test_hole_prefix_end = None
            self._active_adaptive_topk_sections_abs = None
            self._active_trace_gap_positions_local = None
            self._active_trace_gap_positions_abs = None
            self._active_trace_hit_positions_local = None
            self._active_trace_hit_positions_abs = None
            self._layer_topk_ms.clear()
            self._layer_blend_accounted_ms.clear()
