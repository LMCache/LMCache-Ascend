# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List

# Third Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from sglang.srt.configs.model_config import ModelConfig
import torch
import torch.distributed as dist

logger = init_logger(__name__)


def LMCacheConnector__init__(
    self,
    sgl_config: ModelConfig,
    tp_size: int,
    rank: int,
    k_pool: List[torch.Tensor],
    v_pool: List[torch.Tensor],
):
    # Third Party
    from lmcache.integration.sglang.sglang_adapter import init_lmcache_engine

    # NOTE(niming): Ensure k_pool is non-empty before accessing its elements.
    # Explicitly check length to avoid ambiguity with Tensor-like objects.
    if k_pool is None or len(k_pool) == 0:
        raise ValueError("k_pool cannot be empty during initialization.")
    kv_dtype = k_pool[0].dtype
    if k_pool[0].is_cuda and k_pool[0].device.index is not None:
        local_rank = k_pool[0].device.index
    else:
        # Fallback for CPU / odd cases
        local_rank = rank

    # rank is the global tensor parallel rank (tp_rank) from SGLang
    # local_rank is the local GPU device index
    self.lmcache_engine = init_lmcache_engine(
        sgl_config,
        tp_size,
        local_rank,
        rank,  # global_rank (tp_rank) for metadata
        kv_dtype,
    )
    self.sgl_config = sgl_config
    self.tp_size = tp_size
    self.rank = local_rank  # Use local_rank for torch.device() calls

    # NOTE(niming): NPU expects kvcaches = [K_tensor, V_tensor]
    # where K_tensor.shape = [layer_num, num_blocks, block_size, head_num, head_dim]
    # and V_tensor.shape = [layer_num, num_blocks, block_size, head_num, head_dim]
    self.kvcaches = [k_pool, v_pool]

    self.num_layer = sgl_config.num_hidden_layers

    self.lmcache_engine.post_init(kvcaches=self.kvcaches)


@torch.no_grad()
def LMCacheLayerwiseConnector_global_min_tokens(
    self, local_tokens: int, tp_group: dist.ProcessGroup, device: torch.device
):
    """Synchronize min tokens across TP ranks, ensuring NPU stability under load."""
    # If tensor parallel size is 1, no need for all_reduce
    if self.tp_size == 1:
        return local_tokens

    t = torch.tensor([local_tokens], dtype=torch.int32, device=device)
    # NOTE(niming):Mandatory synchronization for TP > 1 on NPU/HCCL.
    # Under high request loads, the NPU task manager may experience race conditions
    # between compute kernels and HCCL all_reduce, leading to a permanent deadlock.
    torch_dev.synchronize()

    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=tp_group)

    return int(t.item())
