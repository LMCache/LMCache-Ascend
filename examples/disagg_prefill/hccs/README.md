## Example of Disaggregated Prefill over HCCS (No RoCE)

This example demonstrates how to run LMCache with disaggregated prefill on a single node **without RoCE**, using the on-chip **HCCS** interconnect instead. It is a TP=1 variant of the [`1p1d`](../1p1d) example.

> This example is validated on a single node with two NPUs connected over
> HCCS. Cross-node deployment requires a supported interconnect topology and
> transport configuration and is not covered by this example.

### Differences from the `1p1d` example

| Item | `1p1d` | This example (HCCS) |
| :--- | :--- | :--- |
| Transfer channel | `hcomm_onesided` (default) | `hixl` |
| Network | RoCE (required) | HCCS only — no RoCE NIC needed |
| Key environment variables | — | `HCCL_INTRA_ROCE_ENABLE=0`, `HCCL_INTRA_PCIE_ENABLE=0` |
| TP size | 2 | 1 |
| Peer ports | `[7300, 7301]` / `[7400, 7401]` | `[7300]` / `[7400]` |

### Prerequisites

- CANN 8.5+ (for the `hixl` channel)
- Ascend HDK 25.5.0+ drivers and firmware
- At least 2 NPUs connected over HCCS
- The following patches from `docker/` must be applied before use:
  - `docker/vllm-utils.diff` to vLLM (the ARM `sched_yield` GIL fix).
    On vLLM 0.18.x the diff does not apply cleanly (older import layout);
    apply the equivalent change manually in `vllm/distributed/utils.py`:
    add `from vllm.platforms import CpuArchEnum, Platform` and make
    `USE_SCHED_YIELD` also `and Platform.get_cpu_architecture() != CpuArchEnum.ARM`.
  - `docker/vllm-sched.diff` to vLLM-Ascend (the `num_external_computed_tokens is None`
    skip-request guard). vLLM-Ascend 0.18.x already contains this logic in both
    `scheduler_dynamic_batch.py` and `recompute_scheduler.py`; no patch needed there.

> After applying patches, reinstall the affected packages (vLLM, vLLM-Ascend, LMCache, LMCache-Ascend) for the changes to take effect.

### HCCS Environment Variables

HCCL must be told to use the on-chip HCCS fabric instead of RoCE/PCIe. Export these in **both** the prefill and decode processes:

```bash
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=0
```

> **Important:** `HCCL_INTRA_PCIE_ENABLE` must be set explicitly. Setting only `HCCL_INTRA_ROCE_ENABLE=0` (or leaving the PCIe variable empty) raises HCCL error 103900 during connection setup.

### Transfer Channel Configuration

Use the `hixl` transfer channel in both `configs/lmcache-prefiller-config.yaml` and `configs/lmcache-decoder-config.yaml`:

```yaml
transfer_channel: "hixl"
```

### Buffer Size

The `pd_buffer_size` field must be:

1. **Identical on the sender and the decoder** (the decoder reads into a buffer of this size that the sender also allocates).
2. A **multiple of the KV page size** — the paged allocator requires `buffer_size % align_bytes == 0`.

The provided configs use `pd_buffer_size: 2415919104` (= 64 pages x 36 MB), aligned with the KV page shape `[2, 36, 256, 1024]` in fp16 for a Qwen3-8B class model with `--block-size 128`. Adjust it proportionally for other model/token budgets.

> **Note:** HCCS transport also requires the KV transfer buffer VA to be 2 MB aligned. This is guaranteed by the PD backend's aligned NPU buffer allocation (see `lmcache_ascend/v1/storage_backend/pd/backend.py`), which is included in the same change set as this example. Without it, HCCL fails to register the buffer (errors 503900 / 507899 / error 15).

### Usage

All values are configurable via environment variables. Equivalent inline commands are shown below.

Launch prefill (sender):

```bash
export MODEL=/path/to/model
export LMCACHE_CONFIG_FILE=/workspace/LMCache-Ascend/examples/disagg_prefill/hccs/configs/lmcache-prefiller-config.yaml
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_VISIBLE_DEVICES=2
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export PYTHONHASHSEED=0
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=0
python \
    -m vllm.entrypoints.openai.api_server \
    --port 8001 \
    --model $MODEL \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --block-size 128 \
    --max-model-len 32768 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_producer", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}' > prefill.txt 2>&1
```

Launch decode (receiver):

```bash
export MODEL=/path/to/model
export LMCACHE_CONFIG_FILE=/workspace/LMCache-Ascend/examples/disagg_prefill/hccs/configs/lmcache-decoder-config.yaml
export ASCEND_RT_VISIBLE_DEVICES=3
export ASCEND_VISIBLE_DEVICES=3
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export PYTHONHASHSEED=0
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=0
python \
    -m vllm.entrypoints.openai.api_server \
    --port 8002 \
    --model $MODEL \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --block-size 128 \
    --max-model-len 32768 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_consumer", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}' > decode.txt 2>&1
```

Launch the proxy server to coordinate prefill and decode:

```bash
python3 /workspace/LMCache/examples/disagg_prefill/disagg_proxy_server.py \
  --host localhost \
  --port 9100 \
  --prefiller-host localhost \
  --prefiller-port 8001 \
  --num-prefillers 1 \
  --decoder-host localhost \
  --decoder-port 8002 \
  --decoder-init-port "7300" \
  --decoder-alloc-port "7400" \
  --proxy-host localhost \
  --proxy-port 7500 \
  --num-decoders 1 \
  --model $MODEL
```

Send a request through the proxy:

```bash
curl -X POST http://localhost:9100/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/path/to/model\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models in English.%.0s' {1..100})\",
    \"max_tokens\": 100
  }"
```

### Notes

1. **Single peer ports for TP=1.** The proxy derives the expected number of TP ranks from the port list length (`num_tp_rank = len(init_port)`). With TP=1 the ports must be single-element lists (`"7300"` / `"7400"`); using two ports makes the proxy wait for responses that never arrive.
2. **Restart all three processes after changing any port.** The HIXL peer ids / handshake state are derived from the ports; a partial restart leaves stale peer registrations. The same applies after a crashed/restarted peer: the surviving side caches the old buffer UUIDs from the first handshake, and sends fail with `Buffer UUID ... not found in remote peer buffers` until all three are restarted.
3. **Use `LMCacheAscendConnectorV1Dynamic`, not `LMCacheAscendConnector`.** vLLM resolves the connector class by exact `getattr()` on `kv_connector_module_path`; the module only defines the `V1Dynamic` class, so the bare name raises `AttributeError`.
4. Official prerequisites (CANN 8.5+, HDK 25.5.0+, docker patches) still apply.

### Validation (real NPU, 2026-10-10)

Verified on a single 8x Ascend 910B3 node (no RoCE), with prefill on NPU 2 and
decode on NPU 3 (TP=1 each), Qwen3-8B, vLLM 0.18.0 + vLLM-Ascend 0.18.0 +
LMCache v0.4.4 (the CI-pinned upstream tag). Software under test:
LMCache-Ascend `feat/hccs-pd-example` commit `b952f54` (pre-rebase; the
aligned-allocation logic in this change set is unchanged by the later
rebase), with the temporary local shim documented below.

- **2 MB aligned buffer**: each process logs its own allocation from
  `backend.py`:
  - prefill (EngineCore pid 84432):
    `Initialized NPU allocator: 2304.00 MB (aligned base: 0x12e0b1600000)`
  - decode (EngineCore pid 85040):
    `Initialized NPU allocator: 2304.00 MB (aligned base: 0x12e0b1600000)`
  Both bases are 2 MB multiples; the two processes happen to report the same
  VA (equal-size allocations in their address spaces).
- **End-to-end transfer**: prefill logs `Stored 2048/2048` (0.2812 GB) and
  `Stored 753/753` (0.1055 GB); the decoder logs
  `Retrieved 2560 out of 2560 required tokens`, first-request throughput
  55.5 GB/s, second-request 119.5 GB/s. The retrieve throughput is the
  software-reported LMCache metric, not an independently measured HCCS
  link-bandwidth benchmark.
- **Prefix-cache hit**: two identical long prompts (~2800 tokens) through the
  proxy; the vLLM engine log reports `External prefix cache hit rate: 91.4%`
  after the first request and `91.0%` after the second, with prefill TTFT
  dropping from 1579 ms to 893 ms (two-request observation, not a stable
  statistic). All requests returned HTTP 200 with streamed completions via
  the proxy `:9100`.
- **Required env vars** confirmed: `HCCL_INTRA_ROCE_ENABLE=0` +
  `HCCL_INTRA_PCIE_ENABLE=0` (missing PCIe var -> HCCL error 103900).

> Environment note: with LMCache v0.4.4, the Ascend cache-engine adapter
> (`lmcache_ascend/v1/cache_engine.py`, receiver `retrieve()` cleanup path)
> calls `self._is_sync_pd_backend()`, but the v0.4.4 base implementation
> lacks this method and raises `AttributeError` on the first request. This
> validation run therefore used a temporary local compatibility shim
> (returning `False`, since v0.4.4 has no sync PD mode):
>
> ```python
> @torch.inference_mode()
> def _is_sync_pd_backend(self) -> bool:
>     """Check if the PD backend is the sync variant.
>
>     Added for compatibility with LMCache >= 0.5.5, which calls this
>     method from ``retrieve()`` cleanup logic.  LMCache 0.4.4 (the pinned
>     upstream tag) has no ``pd_backend_mode`` config; treat it as async
>     (returns ``False``), matching v0.4.4's own ``remove_after_retrieve``
>     behavior which does not ref-count down in that path.
>     """
>     return getattr(self.config, "pd_backend_mode", "async") == "sync"
> ```
>
> The shim is **not** included in this PR. These results therefore validate
> the HCCS example and the aligned allocation path under the stated
> compatibility modification; they do not demonstrate that the example runs
> unchanged with the pinned dependency version. See also the regression test
> in `tests/v1/storage_backend/test_ascend_pd_backend.py`.
