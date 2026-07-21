# SPDX-License-Identifier: Apache-2.0
# Standard
import json
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache, get_zmq_socket
from lmcache.v1.trace_utils import summarize_key, summarize_ranges, trace_flow
from lmcache_ascend.v1.hole_segment_utils import HoleSegmentHelper, derive_lookup_result
from lmcache_ascend.v1.hole_types import HoleLookupResult

logger = init_logger(__name__)


class LMCacheHoleLookupServer:
    def __init__(self, lmcache_engine, vllm_config):
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        assert lmcache_engine.metadata.engine_id is not None, (
            "engine_id is required for RPC communication"
        )
        socket_path = get_zmq_rpc_path_lmcache(
            lmcache_engine.metadata.engine_id,
            "lookup",
            rpc_port,
            lmcache_engine.metadata.worker_id,
        )
        self.socket = get_zmq_socket(
            self.ctx,
            socket_path,
            "ipc",
            zmq.REP,  # type: ignore[attr-defined]
            "bind",
        )
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)

        self.lmcache_engine = lmcache_engine
        self.segment_helper = HoleSegmentHelper(
            lmcache_engine.config, lmcache_engine.metadata
        )
        self.running = True

        def process_request():
            while self.running:
                try:
                    frames = self.socket.recv_multipart(copy=False)
                except zmq.Again:
                    continue

                lookup_id = frames[-2].bytes.decode("utf-8")
                request_configs_str = frames[-1].bytes.decode("utf-8")
                request_configs = None
                if request_configs_str != "":
                    request_configs = json.loads(request_configs_str)

                tokens = self.decoder.decode(frames[0])
                result = self._lookup_tokens(tokens, lookup_id, request_configs)
                self.socket.send(self.encoder.encode(result.to_wire()))

        logger.info("hole lookup server start on %s", socket_path)
        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def _segment_hit(self, tokens, token_range, lookup_id: str, request_configs: dict | None):
        key = self.segment_helper.make_cache_key(tokens, token_range, request_configs)
        key_all_layers = key.split_layers(self.lmcache_engine.num_layers)
        hit_chunks, block_mapping = self.lmcache_engine.storage_manager.batched_contains(
            key_all_layers,
            None,
            True,
        )
        if hit_chunks != self.lmcache_engine.num_layers or len(block_mapping) != 1:
            trace_flow(
                "hole_lookup",
                "segment_miss",
                lookup_id=lookup_id,
                token_range=list(token_range),
                key=summarize_key(key),
                hit_chunks=hit_chunks,
                locations=list(block_mapping.keys()),
            )
            return False, None

        location = next(iter(block_mapping.keys()))
        self.lmcache_engine.lookup_pins[lookup_id][location].extend(key_all_layers)
        trace_flow(
            "hole_lookup",
            "segment_hit",
            lookup_id=lookup_id,
            token_range=list(token_range),
            key=summarize_key(key),
            location=location,
        )
        return True, location

    def _lookup_tokens(
        self,
        tokens,
        lookup_id: str,
        request_configs: dict | None,
    ) -> HoleLookupResult:
        segment_ranges = self.segment_helper.split_ranges(tokens)
        trace_flow(
            "hole_lookup",
            "lookup_start",
            lookup_id=lookup_id,
            total_tokens=len(tokens),
            segment_ranges=summarize_ranges(segment_ranges),
        )
        if not segment_ranges:
            return HoleLookupResult(mode="legacy", covered_tokens=0, tail_start=0)

        hit_flags: list[bool] = []
        first_hit_location = None
        location_conflict = False
        first_miss_idx: int | None = None
        for idx, token_range in enumerate(segment_ranges):
            if location_conflict:
                hit_flags.append(False)
                continue

            is_hit, location = self._segment_hit(
                tokens,
                token_range,
                    lookup_id,
                    request_configs,
                )
            if is_hit and first_hit_location is None:
                first_hit_location = location
            elif is_hit and first_hit_location != location:
                logger.warning(
                    "hole lookup detected multi-location request for %s; "
                    "stopping coverage at the previous location",
                    lookup_id,
                )
                is_hit = False
                location_conflict = True
            hit_flags.append(is_hit)
            if not is_hit:
                first_miss_idx = idx
                break

        if location_conflict:
            hit_flags.extend([False] * (len(segment_ranges) - len(hit_flags)))
            result = derive_lookup_result(
                segment_ranges,
                hit_flags,
                location=first_hit_location,
            )
            trace_flow(
                "hole_lookup",
                "lookup_finish",
                lookup_id=lookup_id,
                hit_flags=hit_flags,
                mode=result.mode,
                covered_tokens=result.covered_tokens,
                hit_ranges=summarize_ranges(result.hit_ranges),
                prefix_miss_ranges=summarize_ranges(result.prefix_miss_ranges),
                speculative_tail_segments=0,
                speculative_fallback=False,
            )
            return result

        if first_miss_idx is not None:
            if first_miss_idx == 0:
                hit_flags.extend([False] * (len(segment_ranges) - len(hit_flags)))
                result = derive_lookup_result(
                    segment_ranges,
                    hit_flags,
                    location=first_hit_location,
                )
                trace_flow(
                    "hole_lookup",
                    "lookup_finish",
                    lookup_id=lookup_id,
                    hit_flags=hit_flags,
                    mode=result.mode,
                    covered_tokens=result.covered_tokens,
                    hit_ranges=summarize_ranges(result.hit_ranges),
                    prefix_miss_ranges=summarize_ranges(result.prefix_miss_ranges),
                    speculative_tail_segments=len(segment_ranges) - 1,
                    speculative_fallback=False,
                )
                return result

            remaining_tail_segments = len(segment_ranges) - first_miss_idx - 1
            if remaining_tail_segments < 2:
                hit_flags.extend([False] * (len(segment_ranges) - len(hit_flags)))
                result = derive_lookup_result(
                    segment_ranges,
                    hit_flags,
                    location=first_hit_location,
                )
                trace_flow(
                    "hole_lookup",
                    "lookup_finish",
                    lookup_id=lookup_id,
                    hit_flags=hit_flags,
                    mode=result.mode,
                    covered_tokens=result.covered_tokens,
                    hit_ranges=summarize_ranges(result.hit_ranges),
                    prefix_miss_ranges=summarize_ranges(result.prefix_miss_ranges),
                    speculative_tail_segments=remaining_tail_segments,
                    speculative_fallback=False,
                )
                return result

            for token_range in segment_ranges[first_miss_idx + 1 :]:
                is_hit, location = self._segment_hit(
                    tokens,
                    token_range,
                    lookup_id,
                    request_configs,
                )
                if is_hit and first_hit_location is not None and first_hit_location != location:
                    logger.warning(
                        "hole lookup detected multi-location request for %s; "
                        "stopping coverage at the previous location",
                        lookup_id,
                    )
                    is_hit = False
                hit_flags.append(is_hit)

        result = derive_lookup_result(
            segment_ranges,
            hit_flags,
            location=first_hit_location,
        )
        trace_flow(
            "hole_lookup",
            "lookup_finish",
            lookup_id=lookup_id,
            hit_flags=hit_flags,
            mode=result.mode,
            covered_tokens=result.covered_tokens,
            hit_ranges=summarize_ranges(result.hit_ranges),
            prefix_miss_ranges=summarize_ranges(result.prefix_miss_ranges),
            speculative_tail_segments=(
                0 if first_miss_idx is None else len(segment_ranges) - first_miss_idx - 1
            ),
            speculative_fallback=first_miss_idx is not None,
        )
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.socket.close(linger=0)
        try:
            self.ctx.term()
        except Exception:
            logger.warning("Error terminating hole lookup server context")
