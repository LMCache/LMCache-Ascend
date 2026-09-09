# SPDX-License-Identifier: Apache-2.0
"""Opt-in real Qwen3.5/NPU acceptance, through the normal tests bootstrap.

Run PR2's native transfer acceptance first. Set QWEN35_TEST_MODEL and explicit
QWEN35_TEST_TP; remote model/tokenizer identifiers require immutable revisions.
This ordered flow owns both sequential LLM lifetimes and its private disk cache.
It does not certify cancellation/preemption or recovery after a partial load.
"""

# Standard
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
import gc
import json
import os
import re
import time

# Third Party
import pytest


def _worker_probe(worker, action):
    """Observe real worker operations; never substitute a lookup or a copy."""
    # Imports run on the real worker, after connector registration.
    # Standard
    from importlib.machinery import EXTENSION_SUFFIXES
    import hashlib
    import inspect

    # Third Party
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    import torch

    # First Party
    from lmcache_ascend.v1.state_checkpoint import state_checkpoint_key
    import lmcache_ascend.c_ops as native
    import lmcache_ascend.integration.vllm.vllm_v1_adapter as adapter_module
    import lmcache_ascend.v1.state_cache as cache_module

    adapter = get_kv_transfer_group()._lmcache_engine
    engine = adapter.lmcache_engine
    cpu = engine.storage_manager.storage_backends["LocalCPUBackend"]
    disk = engine.storage_manager.storage_backends["LocalDiskBackend"]
    rank = engine.metadata.worker_id

    if action == "install":
        assert not hasattr(worker, "_qwen35_e2e")
        assert adapter.state_layouts, "The model did not register GDN state"
        assert inspect.isbuiltin(native.multi_layer_gdn_state_transfer)
        assert any(str(native.__file__).endswith(s) for s in EXTENSION_SUFFIXES)
        trace = {"events": [], "undo": [], "context": {}}
        worker._qwen35_e2e = trace

        def patch(owner, name, replacement):
            trace["undo"].append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)

        original_transfer = cache_module.transfer_state
        assert original_transfer is adapter_module.transfer_state

        def transfer(operation):
            original_transfer(operation)  # PR2 native copy + stream synchronization.
            digest = hashlib.sha256()
            nbytes = 0
            for plane_index, plane in enumerate(operation.buffer.planes):
                # Snapshot payload first: an additional D2H must not hide a
                # missing synchronization in the operation being observed.
                payload = plane.contiguous().view(torch.uint8).clone()
                for layer, entry in enumerate(operation.runtime.tensors):
                    actual = (
                        entry[plane_index][operation.runtime.block_id]
                        .cpu()
                        .contiguous()
                        .view(torch.uint8)
                    )
                    assert torch.equal(payload[layer], actual), "State bytes differ"
                raw = payload.numpy().tobytes()
                digest.update(raw)
                nbytes += len(raw)
            trace["events"].append(
                dict(
                    trace["context"],
                    kind="copy",
                    direction=operation.direction,
                    R=operation.checkpoint.boundary,
                    group=operation.checkpoint.group_index,
                    block=operation.runtime.block_id,
                    key=str(state_checkpoint_key(operation.checkpoint)),
                    sha256=digest.hexdigest(),
                    payload_bytes=nbytes,
                )
            )

        patch(cache_module, "transfer_state", transfer)
        patch(adapter_module, "transfer_state", transfer)
        original_store = engine.store_state

        def store(execution, *args, **kwargs):
            trace["context"] = {
                "request": execution.req_id,
                "start": execution.start,
                "E": execution.end,
                "K": execution.attention_end,
            }
            trace["events"].append(dict(trace["context"], kind="execution"))
            try:
                return original_store(execution, *args, **kwargs)
            finally:
                trace["context"] = {}

        patch(engine, "store_state", store)
        original_load = adapter._load_hybrid_request

        def load(request, execution):
            trace["context"] = {
                "request": request.req_id,
                "C": request.load_spec.vllm_cached_tokens,
                "R": request.load_spec.lmcache_cached_tokens,
                "E": None if execution is None else execution.end,
                "K": None if execution is None else execution.attention_end,
            }
            try:
                result = original_load(request, execution)
                # True is returned only after the production completion log and
                # every real Attention/state transfer succeeds on this worker.
                trace["events"].append(
                    dict(trace["context"], kind="restore", success=result)
                )
                return result
            finally:
                trace["context"] = {}

        patch(adapter, "_load_hybrid_request", load)

        def observe_get(backend, name):
            original = backend.get_blocking

            def get(key):
                obj = original(key)
                if obj is not None:
                    trace["events"].append(
                        {
                            "kind": "read",
                            "backend": name,
                            "key": str(key),
                            "state": getattr(obj.meta, "state_checkpoint", None),
                        }
                    )
                return obj

            patch(backend, "get_blocking", get)

        observe_get(cpu, "LocalCPUBackend")
        observe_get(disk, "LocalDiskBackend")
        return {
            "rank": rank,
            "engine": id(engine),
            "native": native.__file__,
            "groups": [layout.group_index for layout in adapter.state_layouts],
            "layouts": [repr(layout) for layout in adapter.state_layouts],
        }

    trace = worker._qwen35_e2e
    if action == "drain":
        events, trace["events"] = trace["events"], []
        return {"rank": rank, "engine": id(engine), "events": events}
    if action == "evict_cpu":
        if "eviction" in trace:
            return trace["eviction"]
        # Called only with no active request. A disk contains() check is
        # insufficient: wait for published entries AND released async put refs.
        with engine._engine_state_lock, cpu.cpu_lock:
            keys = list(cpu.hot_cache)
            if not keys or any(
                not disk.contains(key) or not cpu.hot_cache[key].can_evict
                for key in keys
            ):
                return {"rank": rank, "ready": False}
            state_keys = [
                str(key)
                for key in keys
                if getattr(cpu.hot_cache[key].meta, "state_checkpoint", None)
            ]
            assert state_keys, "No actual CPU checkpoints to evict"
            for key in keys:
                # force=False neither locks nor checks pins in pinned LMCache;
                # both are handled above. Keep its eviction policy consistent.
                assert cpu.remove(key, force=False)
                cpu.cache_policy.update_on_force_evict(key)
                assert key not in cpu.hot_cache and disk.contains(key)
            result = {
                "rank": rank,
                "ready": True,
                "engine": id(engine),
                "removed": len(keys),
                "state_keys": state_keys,
                "cpu_remaining": len(cpu.hot_cache),
            }
            trace["eviction"] = result
            return result
    if action == "eviction_status":
        return trace.get("eviction", {"rank": rank, "ready": False})
    if action == "close":
        for owner, name, original in reversed(trace["undo"]):
            setattr(owner, name, original)
        del worker._qwen35_e2e
        return rank
    raise ValueError(action)


@contextmanager
def _model(args):
    # Third Party
    from vllm import LLM

    llm = LLM(**args)
    try:
        yield llm
    finally:
        # Stop workers and free the first model before constructing the second.
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()


@pytest.fixture
def qwen35_acceptance(tmp_path, monkeypatch):
    model = os.environ.get("QWEN35_TEST_MODEL")
    if not model:
        pytest.skip("Set QWEN35_TEST_MODEL for real NPU/model acceptance")
    assert "QWEN35_TEST_TP" in os.environ, "Set explicit QWEN35_TEST_TP"
    tp = int(os.environ["QWEN35_TEST_TP"])
    assert tp > 0

    def pinned(identifier, variable):
        if Path(identifier).is_dir():
            return str(Path(identifier).resolve()), None
        revision = os.environ.get(variable, "")
        assert re.fullmatch(r"[0-9a-fA-F]{40}", revision), (
            f"Set {variable} to a full immutable commit, or use a local snapshot"
        )
        return identifier, revision

    model, revision = pinned(model, "QWEN35_TEST_REVISION")
    tokenizer = os.environ.get("QWEN35_TEST_TOKENIZER", model)
    if tokenizer == model:
        tokenizer_revision = os.environ.get("QWEN35_TEST_TOKENIZER_REVISION", revision)
        if tokenizer_revision is not None:
            assert re.fullmatch(r"[0-9a-fA-F]{40}", tokenizer_revision)
    else:
        tokenizer, tokenizer_revision = pinned(
            tokenizer, "QWEN35_TEST_TOKENIZER_REVISION"
        )
    block = int(os.environ.get("QWEN35_TEST_BLOCK_SIZE", "1024"))
    assert block >= 1024 and block % 1024 == 0
    # Isolate from workstation cache settings. The normal pytest bootstrap and
    # LMCACHEPATH remain in use; workers receive only this test's cache config.
    for name in list(os.environ):
        if name.startswith("LMCACHE_"):
            monkeypatch.delenv(name)
    config = {
        "chunk_size": block,
        "local_cpu": True,
        "max_local_cpu_size": float(os.environ.get("QWEN35_TEST_CPU_GB", "8")),
        "local_disk": str(tmp_path / "disk"),
        "max_local_disk_size": float(os.environ.get("QWEN35_TEST_DISK_GB", "16")),
        "store_async": False,
        "enable_async_loading": False,
        "save_unfull_chunk": False,
    }
    config_path = tmp_path / "lmcache.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.setenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "0")

    # Third Party
    import torch

    assert hasattr(torch, "npu") and torch.npu.is_available(), "Real NPU required"
    assert torch.npu.device_count() >= tp
    args = dict(
        model=model,
        revision=revision,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        mamba_cache_mode="align",
        mamba_block_size=block,
        max_num_batched_tokens=2 * block,
        max_num_seqs=1,
        max_model_len=6 * block,
        gpu_memory_utilization=float(os.environ.get("QWEN35_TEST_MEMORY", "0.8")),
    )
    evidence = {
        "args": args,
        "cache_config": config,
        "versions": {
            name: version(name)
            for name in (
                "vllm",
                "vllm-ascend",
                "lmcache",
                "lmcache-ascend",
                "torch",
                "torch-npu",
            )
        },
        "phases": {},
        "acceptance": "incomplete",
    }
    destination = Path(
        os.environ.get("QWEN35_TEST_EVIDENCE", tmp_path / "evidence.json")
    )
    try:
        yield args, block, evidence
    finally:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"Qwen3.5 acceptance evidence: {destination.resolve()}")


def test_qwen35_external_state_acceptance(qwen35_acceptance):
    """Warm -> distinct reuse/full hit -> extension -> miss -> CPU/disk restore."""
    # Third Party
    from vllm import SamplingParams
    from vllm.config import KVTransferConfig

    args, block, evidence = qwen35_acceptance
    sampling = SamplingParams(
        temperature=0, top_p=1, seed=0, max_tokens=8, ignore_eos=True
    )
    evidence["sampling"] = {
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": 8,
        "ignore_eos": True,
    }
    baseline = {}
    with _model(args) as llm:
        tokenizer = llm.get_tokenizer()

        def tokens(text, length):
            unit = tokenizer.encode(text, add_special_tokens=False)
            assert unit
            return (unit * (length // len(unit) + 1))[:length]

        warm = tokens("The river flows past the quiet village. ", 4 * block)
        prompts = {
            "warm": warm,
            "prefix": warm[: 2 * block]
            + tokens("A different ending begins here. ", block),
            "extension": warm
            + tokens("Continue the journey beyond the village. ", block // 2),
            "cold": tokens("Zebras gather beside the mountain. ", 2 * block),
        }
        assert prompts["cold"][0] != warm[0]
        assert prompts["prefix"][2 * block] != warm[2 * block]
        evidence["prompts"] = prompts
        for name, prompt in prompts.items():
            assert llm.reset_prefix_cache(reset_connector=False)
            output = llm.generate(
                [{"prompt_token_ids": prompt}], sampling, use_tqdm=False
            )[0]
            baseline[name] = list(output.outputs[0].token_ids)
    evidence["baseline"] = baseline

    cached_args = dict(
        args,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheAscendConnector", kv_role="kv_both"
        ),
    )
    with _model(cached_args) as llm:

        def rpc(action):
            return llm.collective_rpc(_worker_probe, args=(action,), timeout=60)

        workers = rpc("install")
        evidence["workers"] = workers
        assert len(workers) == args["tensor_parallel_size"]
        ranks = {worker["rank"] for worker in workers}
        assert ranks == set(range(args["tensor_parallel_size"]))
        identities = {worker["rank"]: worker["engine"] for worker in workers}
        groups = {worker["rank"]: set(worker["groups"]) for worker in workers}
        saved = {}
        request_ids = set()

        def run(name, prompt_name, expected_r=None, disk=False):
            # Preserve the same LMCache objects while ensuring C cannot be a
            # vLLM-only prefix hit. Actual C<R is asserted from worker metadata.
            assert llm.reset_prefix_cache(reset_connector=False)
            rpc("drain")
            prompt = prompts[prompt_name]
            output = llm.generate(
                [{"prompt_token_ids": prompt}], sampling, use_tqdm=False
            )[0]
            traces = rpc("drain")
            generated = list(output.outputs[0].token_ids)
            internal_ids = {
                event["request"]
                for trace in traces
                for event in trace["events"]
                if "request" in event
            }
            evidence["phases"][name] = {
                "external_request": output.request_id,
                "internal_requests": sorted(internal_ids),
                "output": generated,
                "workers": traces,
                "suffix_tokens": len(prompt) - (expected_r or 0),
            }
            assert output.request_id not in request_ids
            request_ids.add(output.request_id)
            # Pinned vLLM appends eight UUID hex characters to worker IDs;
            # RequestOutput deliberately carries the original external ID.
            assert len(internal_ids) == 1, (name, traces)
            internal_id = next(iter(internal_ids))
            assert re.fullmatch(
                re.escape(output.request_id) + r"-[0-9a-f]{8}", internal_id
            )
            assert generated == baseline[prompt_name], name
            assert {trace["rank"] for trace in traces} == ranks
            for trace in traces:
                rank, events = trace["rank"], trace["events"]
                assert trace["engine"] == identities[rank], "LMCache was reset"
                restores = [e for e in events if e["kind"] == "restore"]
                loads = [
                    e
                    for e in events
                    if e["kind"] == "copy" and e["direction"] == "load"
                ]
                if expected_r is None:
                    assert not restores and not loads, (name, trace)
                else:
                    assert len(restores) == 1, (name, trace)
                    restore = restores[0]
                    assert restore["success"] is True
                    assert restore["request"] == internal_id
                    assert restore["C"] == 0 < restore["R"] == expected_r
                    assert {e["group"] for e in loads} == groups[rank]
                    assert len(loads) == len(groups[rank])
                    for event in loads:
                        assert event["R"] == expected_r
                        assert event["block"] > 0
                        assert event["sha256"] == saved[(rank, event["key"])]
                        if disk:
                            assert any(
                                read["kind"] == "read"
                                and read["backend"] == "LocalDiskBackend"
                                and read["key"] == event["key"]
                                and read["state"] is not None
                                for read in events
                            ), "State was not read from actual disk"
                for event in events:
                    if event["kind"] == "copy" and event["direction"] == "store":
                        assert event["R"] == event["E"]
                        assert event["block"] > 0 and event["payload_bytes"] > 0
                        saved[(rank, event["key"])] = event["sha256"]
            return traces

        try:
            warm_traces = run("first_save", "warm")
            for trace in warm_traces:
                # Actual sparse endpoints, never fabricated S(block)/S(3*block).
                assert {e["R"] for e in trace["events"] if e["kind"] == "copy"} == {
                    2 * block,
                    4 * block,
                }
            run("full_hit_earlier_checkpoint", "warm", 2 * block)
            run("distinct_request_shared_prefix", "prefix", 2 * block)
            run("extension", "extension", 4 * block)
            cold_traces = run("isolated_first_save", "cold")
            for trace in cold_traces:
                assert {e["R"] for e in trace["events"] if e["kind"] == "copy"} == {
                    2 * block
                }
            run("no_earlier_checkpoint_miss", "cold")

            # Disk publication is asynchronous even with synchronous state copy.
            # Each worker evicts at most once; polling never resets either engine.
            deadline = time.monotonic() + 60
            while True:
                status = rpc("eviction_status")
                if all(item["ready"] for item in status):
                    break
                assert time.monotonic() < deadline, (
                    "Disk publication/CPU release timed out"
                )
                rpc("evict_cpu")
                time.sleep(0.1)
            evidence["eviction"] = status
            for item in status:
                assert item["removed"] > 0 and item["cpu_remaining"] == 0
                assert item["engine"] == identities[item["rank"]]
            run("disk_after_cpu_eviction", "warm", 2 * block, disk=True)
            evidence["acceptance"] = (
                "normal-path flow passed; fault/lifecycle matrix pending"
            )
        finally:
            rpc("close")
