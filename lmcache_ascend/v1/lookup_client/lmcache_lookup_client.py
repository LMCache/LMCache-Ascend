# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)


def LMCacheLookupClient_lookup(
    self,
    token_ids: Union[torch.Tensor, list[int]],
    lookup_id: str,
    request_configs: Optional[dict] = None,
) -> Optional[int]:
    # NOTE(niming): Ensure token_ids is a list; vLLM may pass
    # custom types like ConstantList that are not directly
    # serializable by the transport layer.
    if not isinstance(token_ids, list):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        else:
            token_ids = list(token_ids)

    # Delegate to the upstream lookup implementation with
    # normalized token_ids
    # Standard
    import json

    request_configs_str = ""
    if request_configs is not None and len(request_configs) != 0:
        request_configs_str = json.dumps(request_configs)

    if not self.enable_blending:
        hashes = []
        offsets = []

        for start, end, key in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(key)
            offsets.append(end - start)

        if not hashes:
            return 0

        msg_buf = [
            hashes,
            offsets,
            lookup_id,
            request_configs_str,
        ]
    else:
        msg_buf = [
            token_ids,
            lookup_id,
            request_configs_str,
        ]

    responses = self.transport.send_and_recv_all(msg_buf)

    if not responses:
        return 0

    results = [int.from_bytes(resp, "big") for resp in responses]

    assert len(results) == self.transport.world_size
    if len(set(results)) > 1:
        logger.warning(
            "Lookup results (number of hit tokens) differ "
            "across (TP and PP) ranks: %s.",
            results,
        )
    num_hit_toks = min(results)
    self.reqs_status[lookup_id] = num_hit_toks

    return num_hit_toks
