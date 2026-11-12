# SPDX-License-Identifier: Apache-2.0
"""Qwen3 MP selective blending before the standard scheduler's tail forward."""

# Standard
from typing import Any
import json

# Third Party
from lmcache.integration.vllm.mp_blend_client import (
    BlendCoverageTask,
    LMCacheMPSelectiveConnector,
)
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.token_codec import unpack_token_ids
from vllm.distributed import get_tp_group
import torch

# First Party
from lmcache_ascend.v1.blend.mp_selective import SelectiveConfig, selective_prefill

logger = init_logger(__name__)


class MPSelectiveBlendConnector(LMCacheMPSelectiveConnector):
    """Complete loaded coverage with selective Qwen3 computation.

    Uses the standard vLLM scheduler and forward context. Requires synchronous,
    eager execution, one request, no APC/LoRA/quantization, and NHD KV storage.
    Ordinary forward starts at the tail; no KV-only worker extension is needed.
    """

    def __init__(
        self, vllm_config: Any, role: Any, kv_cache_config: Any = None
    ) -> None:
        cfg = vllm_config
        if not cfg.model_config.enforce_eager:
            raise ValueError("MP selective blending requires enforce_eager=True")
        if cfg.cache_config.enable_prefix_caching:
            raise ValueError("MP selective blending requires prefix caching disabled")
        if cfg.scheduler_config.max_num_seqs != 1:
            raise ValueError("MP selective blending requires max_num_seqs=1")
        if cfg.lora_config is not None or cfg.model_config.quantization is not None:
            raise ValueError("MP selective blending does not support LoRA/quantization")
        if cfg.model_config.hf_config.model_type != "qwen3":
            raise ValueError("MP selective blending requires dense Qwen3")
        get_config = cfg.kv_transfer_config.get_from_extra_config
        self.selective_config = SelectiveConfig(
            ratio=float(get_config("lmcache.mp.blend_recompute_ratio", 0.15)),
            check_layer=get_config("lmcache.mp.blend_check_layer", 1),
            selection_metric=get_config("lmcache.mp.blend_selection_metric", "k"),
            refresh_check_layer=get_config(
                "lmcache.mp.blend_refresh_check_layer", False
            ),
        )
        if cfg.speculative_config is not None:
            raise ValueError(
                "MP selective blending does not support speculative decoding"
            )
        parallel = cfg.parallel_config
        if parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1:
            raise ValueError("MP selective blending requires one DP and PP rank")
        if str(cfg.cache_config.cache_dtype).startswith("fp8"):
            raise ValueError("MP selective blending requires an unquantized KV cache")
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("MP selective blending requires one KV cache group")
        self.selective_caches: dict[str, Any] = {}
        super().__init__(vllm_config, role, kv_cache_config)
        if self.lazy_offload:
            raise ValueError("MP selective blending requires eager offload")

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Capture worker-local paged KV, rejecting packed Ascend formats."""
        # Third Party
        import torch_npu

        for pair in kv_caches.values():
            for tensor in pair:
                if torch_npu.get_npu_format(tensor) not in (0, 2):
                    raise ValueError(
                        "MP selective blending requires NPU KV format 0 or 2"
                    )
        super().register_kv_caches(kv_caches)
        self.selective_caches = kv_caches

    def complete_blend(
        self, task: BlendCoverageTask, loaded: bool, forward_context: Any
    ) -> None:
        """Agree on load success, then compute every outstanding position.

        A confirmed failure on any TP rank discards all hits for this step.
        Successful ranks have already waited for their writes before entering
        this collective, so full recomputation cannot race a late scatter.
        """
        group = get_tp_group()
        first_cache = next(iter(self.selective_caches.values()))[0]
        local_success = torch.tensor(
            [int(loaded)], device=first_cache.device, dtype=torch.int32
        )
        success_count = int(group.all_reduce(local_success).item())
        all_loaded = success_count == group.world_size
        ready = task.computed_prefix_tokens
        if all_loaded:
            ready = max(ready, min(task.prefix_hit_tokens, task.coverage_end))
        stats = {"computed_end": task.coverage_end, "retained_prefix_tokens": ready}
        if ready < task.coverage_end:
            stats = selective_prefill(
                forward_context.model_instance,
                self.selective_caches,
                unpack_token_ids(task.key.token_bytes)[: task.coverage_end],
                task.block_ids[0],
                [(m.cur_st, m.cur_ed) for m in task.matches] if all_loaded else [],
                self.selective_config,
                reduce_scores=group.all_reduce,
                broadcast=group.broadcast,
                computed_prefix_tokens=ready,
            )
        logger.info(
            "MP_BLEND_EVENT %s",
            json.dumps(
                {
                    "event": "selective_prefill_complete",
                    "request_id": task.key.request_id,
                    "worker_id": self.worker_adapter.worker_id,
                    "load_success_count": success_count,
                    "full_recompute_fallback": not all_loaded,
                    **stats,
                }
            ),
        )

    def unregister_kv_caches(self) -> None:
        """Release captured views with the base connector's IPC registration."""
        super().unregister_kv_caches()
        self.selective_caches = {}

    def restore_kv_caches(self) -> None:
        """Reject sleep/wake until refreshed paged views are supported."""
        raise NotImplementedError("MP selective blending does not support sleep/wake")
