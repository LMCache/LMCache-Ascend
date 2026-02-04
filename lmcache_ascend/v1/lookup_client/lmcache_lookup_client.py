# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import json

# Third Party
from lmcache.logging import init_logger
import torch
import zmq

logger = init_logger(__name__)


def lookup(
    self,
    token_ids: Union[torch.Tensor, list[int]],
    lookup_id: str,
    request_configs: Optional[dict] = None,
) -> Optional[int]:
    # NOTE:Ensure token_ids is a list; vLLM may pass
    # custom types like ConstantList
    if not isinstance(token_ids, list):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        else:
            token_ids = list(token_ids)

    lookup_id_buf = lookup_id.encode("utf-8")
    request_configs_str = ""
    if request_configs is not None and len(request_configs) != 0:
        request_configs_str = json.dumps(request_configs)
    request_configs_buf = request_configs_str.encode("utf-8")

    # NOTE(Jiayi): We cannot only send hashes when blending enabled
    # because the blender need the input embedding.
    if not self.enable_blending:
        hashes = []
        offsets = []

        # We already have hashes here so we can skip the chunks that are already
        # in GPU cache. Don't pass num_computed_tokens to lookup server.

        for start, end, key in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(key)
            offsets.append(end - start)

        # if the token database returns no hashes, return 0
        if not hashes:
            return 0

        hash_buf = self.encoder.encode(hashes)
        offset_buf = self.encoder.encode(offsets)
        msg_buf = [
            hash_buf,
            offset_buf,
            lookup_id_buf,
            request_configs_buf,
        ]
    else:
        tokens_buf = self.encoder.encode(token_ids)
        msg_buf = [
            tokens_buf,
            lookup_id_buf,
            request_configs_buf,
        ]

    results = []
    failed_rank = -1
    try:
        for i in range(self.num_ranks):
            failed_rank = i
            self.sockets[i].send_multipart(msg_buf, copy=False)

        # TODO(Jiayi): we can use zmq poll to optimize a bit
        for i in range(self.num_ranks):
            failed_rank = i
            resp = self.sockets[i].recv()
            result = int.from_bytes(resp, "big")
            results.append(result)
    except zmq.Again as e:
        logger.error(
            "Timeout occurred for rank %s, recreating all sockets. Error: %s",
            failed_rank,
            e,
        )
        self._recreate_socket()
        return 0
    except zmq.ZMQError as e:
        logger.error(
            "ZMQ error for rank %s: %s, recreating all sockets",
            failed_rank,
            e,
        )
        self._recreate_socket()
        return 0

    assert len(results) == self.num_ranks
    if len(set(results)) > 1:
        logger.warning(
            "Lookup results (number of hit tokens) differ "
            "across (TP and PP) ranks: %s.",
            results,
        )
    # NOTE: it is possible that the number of hit tokens is different
    # across (TP and PP) ranks, so we can use the minimum value as the
    # number of hit tokens.
    num_hit_toks = min(results)
    self.reqs_status[lookup_id] = num_hit_toks

    return num_hit_toks
