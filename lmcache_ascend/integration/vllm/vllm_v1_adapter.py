# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
)
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from vllm.distributed.parallel_state import get_pp_group
import torch

logger = init_logger(__name__)


# Patching wait_for_save to remove the PD disagg_spec skip_leading_tokens
# override. The upstream code does:
#   if self.kv_role == "kv_producer" and request.disagg_spec:
#       skip_leading_tokens = min(skip_leading_tokens,
#                                 request.disagg_spec.num_transferred_tokens)
# save_spec.skip_leading_tokens is already aligned with the number of tokens
# that have been saved, in chunk prefills and delay pull mode, this can cause
# redundant full re-saves when there is an existing cache hit.
# In push mode, this is not a problem, because the skip leading tokens
# already aligns with the number of tokens that have been saved.
@_lmcache_nvtx_annotate
def wait_for_save(self):
    """Blocking until the KV cache is saved to the connector buffer."""

    connector_metadata = self._parent._get_connector_metadata()
    assert isinstance(connector_metadata, LMCacheConnectorMetadata)

    if self.kv_role == "kv_consumer":
        return

    if self.use_layerwise:
        for request in connector_metadata.requests:
            layerwise_storer = self._layerwise_save_storers.pop(request.req_id, None)
            if layerwise_storer is not None:
                next(layerwise_storer)
            self.lmcache_engine.lookup_unpin(request.req_id)
        return

    assert len(self.kv_caches) > 0
    kvcaches = list(self.kv_caches.values())

    assert self.lmcache_engine is not None

    for request in connector_metadata.requests:
        self.lmcache_engine.lookup_unpin(request.req_id)

        save_spec = request.save_spec
        if (
            save_spec is None or not save_spec.can_save
        ) and self.kv_role != "kv_producer":
            continue

        token_ids = request.token_ids

        slot_mapping = request.slot_mapping
        assert isinstance(slot_mapping, torch.Tensor)
        assert len(slot_mapping) == len(token_ids)

        slot_mapping = slot_mapping.to(self.device)

        skip_leading_tokens = save_spec.skip_leading_tokens

        if skip_leading_tokens == len(token_ids):
            continue
        skip_leading_tokens = (
            skip_leading_tokens // self._lmcache_chunk_size * self._lmcache_chunk_size
        )

        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:skip_leading_tokens] = False

        logger.info(
            "Storing KV cache for %d out of %d tokens "
            "(skip_leading_tokens=%d) for request %s",
            len(token_ids) - skip_leading_tokens,
            len(token_ids),
            skip_leading_tokens,
            request.req_id,
        )

        is_last_prefill = request.is_last_prefill
        if is_last_prefill:
            if request.disagg_spec:
                request.disagg_spec.is_last_prefill = True
        else:
            if not self.enable_blending:
                token_len = len(token_ids)
                aligned_token_len = (
                    token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                )
                token_ids = token_ids[:aligned_token_len]
                store_mask = store_mask[:aligned_token_len]
                slot_mapping = slot_mapping[:aligned_token_len]

        self.lmcache_engine.store(
            token_ids,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            transfer_spec=request.disagg_spec,
            request_configs=request.request_configs,
            req_id=request.req_id,
        )

        if get_pp_group().is_last_rank:
            save_spec.skip_leading_tokens = len(token_ids)
            if request.disagg_spec:
                request.disagg_spec.num_transferred_tokens = len(token_ids)
