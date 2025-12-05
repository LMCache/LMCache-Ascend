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
from typing import TYPE_CHECKING

# Third Party
import torch
import numpy as np

from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata

if TYPE_CHECKING:
    # Third Party
    from vllm.forward_context import ForwardContext

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
def LMCacheConnectorV1Impl_start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
    """Start loading the KV cache from the connector buffer to vLLM's
    paged KV buffer.

    Args:
        forward_context (ForwardContext): the forward context.
        **kwargs: additional arguments for the load operation

    Note:
        The number of elements in kv_caches and layer_names should be
        the same.
    """
    self.current_layer = 0

    if len(self.kv_caches) == 0:
        self._init_kv_caches_from_forward_context(forward_context)
    
    metadata = self._parent._get_connector_metadata()
    assert isinstance(metadata, LMCacheConnectorMetadata)

    assert len(self.kv_caches) > 0
    kvcaches = list(self.kv_caches.values())

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        logger.warning("In connector.start_load_kv, but the attn_metadata is None")
        return

    assert self.lmcache_engine is not None

    for idx, request in enumerate(metadata.requests):
        if request.load_spec is None:
            continue
        last_idx = idx

    self.layerwise_retrievers = []
    for idx, request in enumerate(metadata.requests):
        if request.load_spec is None:
            continue

        tokens = request.token_ids.asnumpy()
        # TODO: have a pre-allocated buffer to hold the slot_mappings
        slot_mapping = request.slot_mapping.cuda()
        assert len(tokens) == len(slot_mapping)

        token_mask = np.ones_like(tokens, dtype=np.bool_)
        masked_token_count = (
            request.load_spec.vllm_cached_tokens
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )
        token_mask[:masked_token_count] = False

        if self.skip_last_n_tokens > 0:
            tokens = tokens[: -self.skip_last_n_tokens]
            token_mask = token_mask[: -self.skip_last_n_tokens]

        lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
        if self.use_layerwise:
            if idx == last_idx:
                sync = True
            else:
                sync = False
            # NOTE(Jiayi): Perform blending before layerwise prefix caching
            if self.enable_blending:
                # TODO(Jiayi): Need to make prefix caching and blending compatible
                self.blender.blend(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                )
            else:
                layerwise_retriever = self.lmcache_engine.retrieve_layer(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    sync=sync,
                )
                # NOTE: retrieve for two layers at the first layer
                next(layerwise_retriever)
                next(layerwise_retriever)
                self.layerwise_retrievers.append(layerwise_retriever)
        else:
            ret_token_mask = self.lmcache_engine.retrieve(
                tokens[:lmcache_cached_tokens],
                token_mask[:lmcache_cached_tokens],
                kvcaches=kvcaches,
                slot_mapping=slot_mapping[:lmcache_cached_tokens],
            )

            # Check the result
            num_retrieved_tokens = ret_token_mask.sum().item()
            num_expected_tokens = (
                lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
            )
            if num_retrieved_tokens < num_expected_tokens:
                logger.error(
                    "The number of retrieved tokens is less than the "
                    "expected number of tokens! This should not happen!"
                )
                logger.error(
                    "Num retrieved tokens: %d, num expected tokens: %d",
                    num_retrieved_tokens,
                    num_expected_tokens,
                )


@_lmcache_nvtx_annotate
def LMCacheConnectorV1Impl_wait_for_save(self):
    """Blocking until the KV cache is saved to the connector buffer."""
    if self.kv_role == "kv_consumer":
        # Don't do save if the role is kv_consumer
        return

    if self.use_layerwise:
        for layerwise_storer in self.layerwise_storers:
            next(layerwise_storer)
        return

    connector_metadata = self._parent._get_connector_metadata()
    assert isinstance(connector_metadata, LMCacheConnectorMetadata)

    assert len(self.kv_caches) > 0
    kvcaches = list(self.kv_caches.values())

    assert self.lmcache_engine is not None

    for request in connector_metadata.requests:
        save_spec = request.save_spec
        if save_spec is None or not save_spec.can_save:
            continue

        token_ids = request.token_ids
        assert isinstance(token_ids, torch.Tensor)
        
        # NOTE: Mindspore TorchAdapter is_cpu is always false
        # assert token_ids.is_cpu
        token_ids_np = token_ids.asnumpy()

        slot_mapping = request.slot_mapping
        assert isinstance(slot_mapping, torch.Tensor)
        assert len(slot_mapping) == len(token_ids)

        # TODO: have a pre-allocated buffer to hold the slot_mappings
        slot_mapping = slot_mapping.cuda()
        # NOTE: In PD setting, lmcache_engine.lookup() will always return
        # 0 if there is no local storage configured. In this case, we
        # should rely on the slip_leading_tokens in save_spec to avoid
        # transmit the already saved tokens again.
        # skip_leading_tokens = max(
        #    self.lmcache_engine.lookup(token_ids),
        #    save_spec.skip_leading_tokens,
        # )
        skip_leading_tokens = save_spec.skip_leading_tokens

        if skip_leading_tokens == len(token_ids_np):
            continue  # skip this request
        # Align to lmcache chunk size
        skip_leading_tokens = (
            skip_leading_tokens
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )

        store_mask = np.ones_like(token_ids_np, dtype=np.bool_)
        store_mask[:skip_leading_tokens] = False
        logger.info(
            "Storing KV cache for %d out of %d tokens "
            "(skip_leading_tokens=%d) for request %s",
            len(token_ids_np) - skip_leading_tokens,
            len(token_ids_np),
            skip_leading_tokens,
            request.req_id,
        )
        self.lmcache_engine.store(
            token_ids_np,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
        )

        # NOTE(Jiayi): We assume all tokens are saved
        save_spec.skip_leading_tokens = len(token_ids)
