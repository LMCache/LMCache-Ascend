# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache.logging import init_logger
from lmcache.v1.event_manager import EventStatus, EventType

logger = init_logger(__name__)


def patched_prefetch_all_done_callback(
    self,
    task,
    lookup_id,
    cum_chunk_lengths_total,
    tier_expected_chunks,
):
    assert self.async_lookup_server is not None
    self.event_manager.update_event_status(
        EventType.LOADING, lookup_id, status=EventStatus.DONE
    )

    res = task.result()

    total_retrieved_chunks = 0
    for tier_idx, tier_result in enumerate(res):
        actual_chunks = len(tier_result)
        total_retrieved_chunks += actual_chunks
        if actual_chunks < tier_expected_chunks[tier_idx]:
            for subsequent_tier in res[tier_idx + 1 :]:
                for mem_obj in subsequent_tier:
                    mem_obj.ref_count_down()
            break

    # inject hotcache start ---------------------
    if (
        self.local_cpu_backend is not None
        and self.local_cpu_backend.use_hot
        and total_retrieved_chunks > 0
    ):
        chunk_count = 0
        for tier_idx, tier_result in enumerate(res):
            tier_keys = []
            tier_objs = []
            for key, mem_obj in tier_result:
                if chunk_count >= total_retrieved_chunks:
                    break
                tier_keys.append(key)
                tier_objs.append(mem_obj)
                chunk_count += 1
            if tier_keys:
                self.local_cpu_backend.batched_submit_put_task(tier_keys, tier_objs)
            if chunk_count >= total_retrieved_chunks:
                break
    # inject hotcache end ---------------------

    retrieved_length = cum_chunk_lengths_total[total_retrieved_chunks]
    logger.info(
        f"Responding to scheduler for lookup id {lookup_id}"
        f" with retrieved length {retrieved_length}"
    )
    self.async_lookup_server.send_response_to_scheduler(lookup_id, retrieved_length)
