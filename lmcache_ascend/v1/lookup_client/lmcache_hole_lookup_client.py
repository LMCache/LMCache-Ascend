# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import namedtuple
from typing import Optional, Union
import json

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_rpc_path_lmcache,
    get_zmq_socket,
)
import msgspec
import torch
import zmq

# First Party
from lmcache_ascend.v1.hole_segment_utils import merge_hit_flags
from lmcache_ascend.v1.hole_types import HoleLookupResult

logger = init_logger(__name__)


class LMCacheHoleLookupClient:
    def __init__(self, vllm_config, config, metadata):
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder()
        self.ctx = get_zmq_context(use_asyncio=False)
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        assert metadata.engine_id is not None, (
            "engine_id is required for RPC communication"
        )
        self.pipeline_parallel_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_ranks = self.tensor_parallel_size * self.pipeline_parallel_size
        self.lookup_server_worker_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )

        self.sockets = []
        if len(self.lookup_server_worker_ids) > 0:
            ranks = self.lookup_server_worker_ids
            self.num_ranks = len(self.lookup_server_worker_ids)
        else:
            ranks = [i for i in range(self.num_ranks)]

        SocketParams = namedtuple("SocketParams", ["socket_path", "rank"])
        self.socket_params = [
            SocketParams(
                socket_path=get_zmq_rpc_path_lmcache(
                    metadata.engine_id, "lookup", rpc_port, rank
                ),
                rank=rank,
            )
            for rank in ranks
        ]
        self.timeout_ms = config.lookup_timeout_ms
        self.reqs_status: dict[str, HoleLookupResult] = {}

        for params in self.socket_params:
            socket = get_zmq_socket(
                self.ctx,
                params.socket_path,
                "ipc",
                zmq.REQ,
                "connect",
            )
            socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.sockets.append(socket)

    def _recreate_socket(self) -> None:
        for rank_idx in range(self.num_ranks):
            old_socket = self.sockets[rank_idx]
            if old_socket is not None:
                try:
                    old_socket.close(linger=0)
                except zmq.ZMQError:
                    pass

            params = self.socket_params[rank_idx]
            new_socket = get_zmq_socket(
                self.ctx,
                params.socket_path,
                "ipc",
                zmq.REQ,
                "connect",
            )
            new_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            new_socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.sockets[rank_idx] = new_socket

    def lookup_cache(self, lookup_id: str) -> Optional[HoleLookupResult]:
        return self.reqs_status.get(lookup_id)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> HoleLookupResult:
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

        tokens_buf = self.encoder.encode(token_ids)
        msg_buf = [tokens_buf, lookup_id_buf, request_configs_buf]

        results: list[HoleLookupResult] = []
        failed_rank = -1
        try:
            for i in range(self.num_ranks):
                failed_rank = i
                self.sockets[i].send_multipart(msg_buf, copy=False)

            for i in range(self.num_ranks):
                failed_rank = i
                resp = self.sockets[i].recv()
                payload = self.decoder.decode(resp)
                results.append(HoleLookupResult.from_wire(payload))
        except zmq.Again as error:
            logger.error(
                "hole lookup timeout on rank %s: %s",
                failed_rank,
                error,
            )
            self._recreate_socket()
            result = HoleLookupResult(mode="legacy", covered_tokens=0, tail_start=0)
            self.reqs_status[lookup_id] = result
            return result
        except zmq.ZMQError as error:
            logger.error(
                "hole lookup ZMQ error on rank %s: %s",
                failed_rank,
                error,
            )
            self._recreate_socket()
            result = HoleLookupResult(mode="legacy", covered_tokens=0, tail_start=0)
            self.reqs_status[lookup_id] = result
            return result

        result = merge_hit_flags(results)
        self.reqs_status[lookup_id] = result
        return result

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.reqs_status.pop(lookup_id, None)

    def supports_producer_reuse(self) -> bool:
        return True

    def close(self):
        for socket in self.sockets:
            try:
                socket.close(linger=0)
            except Exception:
                logger.warning("Error closing hole lookup socket")
        try:
            if self.ctx:
                self.ctx.term()
        except Exception:
            logger.warning("Error terminating hole lookup context")
