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
    ranks = self.tensor_parallel_size
    if self.create_lookup_server_only_on_worker_0_for_mla:
        ranks = 1

    # NOTE(Jiayi): We cannot only send hashes when blending enabled
    # because the blender need the input embedding.
    if not self.enable_blending:
        hashes = []
        offsets = []
        for start, end, key in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(key)
            offsets.append(end - start)
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
    try:
        for i in range(ranks):
            self.sockets[i].send_multipart(msg_buf, copy=False)

        # TODO(Jiayi): we can use zmq poll to optimize a bit
        for i in range(ranks):
            resp = self.sockets[i].recv()
            result = int.from_bytes(resp, "big")
            results.append(result)
    except zmq.Again:
        logger.error(f"Timeout occurred for rank {i}")
        return 0
    except zmq.ZMQError as e:
        logger.error(f"ZMQ error for rank {i}: {str(e)}")
        return 0

    assert len(results) == ranks
    if len(set(results)) > 1:
        logger.warning(
            f"Lookup results (number of hit tokens) differ "
            f"across tensor parallel ranks: {results}."
        )
    # NOTE: it is possible that the number of hit tokens is different
    # across TP ranks, so we can use the minimum value as the
    # number of hit tokens.
    return min(results)
