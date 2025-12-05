# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union
import threading
import json

# Third Party
from vllm.utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackDecoder
import torch
import zmq
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)

def LMCacheLookupClient_lookup(
    self,
    token_ids: Union[torch.Tensor, list[int]],
    lookup_id: str,
    request_configs: Optional[dict] = None,
) -> Optional[int]:
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
        tokens_buf = self.encoder.encode(token_ids.asnumpy())
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

def LMCacheLookupServer___init__(self, lmcache_engine: LMCacheEngine, vllm_config: "VllmConfig"):
    self.decoder = MsgpackDecoder(np.ndarray)
    self.decoder = msgspec.msgpack.Decoder()
    self.ctx = zmq.Context()  # type: ignore[attr-defined]
    rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
        "lmcache_rpc_port", 0
    )
    socket_path = get_zmq_rpc_path_lmcache(
        vllm_config, "lookup", rpc_port, vllm_config.parallel_config.rank
    )
    self.socket = make_zmq_socket(
        self.ctx,
        socket_path,
        zmq.REP,  # type: ignore[attr-defined]
        bind=True,
    )

    self.lmcache_engine = lmcache_engine
    self.running = True

    self.enable_blending = lmcache_engine.config.enable_blending

    def process_request():
        while self.running:
            frames = self.socket.recv_multipart(copy=False)
            lookup_id = frames[-2].bytes.decode("utf-8")
            request_configs_str = frames[-1].bytes.decode("utf-8")
            request_configs = None
            if request_configs_str != "":
                request_configs = json.loads(request_configs_str)
            if not self.enable_blending:
                hash_frames = frames[0]
                offset_frames = frames[1]
                hashes = self.decoder.decode(hash_frames)
                offsets = self.decoder.decode(offset_frames)
                result = self.lmcache_engine.lookup(
                    hashes=hashes,
                    offsets=offsets,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                )
            else:
                token_frames = frames[0]
                tokens = self.decoder.decode(token_frames)
                result = self.lmcache_engine.lookup(
                    tokens=tokens,
                    lookup_id=lookup_id,
                    pin=True,
                    request_configs=request_configs,
                )
            response = result.to_bytes(4, "big")
            self.socket.send(response)

    logger.info(f"lmcache lookup server start on {socket_path}")
    self.thread = threading.Thread(target=process_request, daemon=True)
    self.thread.start()
