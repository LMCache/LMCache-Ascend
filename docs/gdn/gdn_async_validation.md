# GDN asynchronous caching: implementation and validation

## Local deployment configuration

| Setting | Value |
| --- | --- |
| Python | 3.12 |
| vLLM | 0.23 RC environment |
| LMCache core | 0.4.5 |
| Model | Qwen-family 27B GDN hybrid model |
| Devices / tensor parallelism | 2 Ascend NPUs / TP=2 |
| CPU cache | 32 GiB per rank, 64 GiB across TP=2 |
| LMCache chunk size | 256 tokens |
| Native prefix cache | Enabled |
| Chunked prefill / step token budget | Enabled / 8,192 tokens |
| Maximum model length | 262,144 tokens |
| Maximum sequences | 128 |
| Device memory utilization | 0.9 |
| Hybrid cache | Enabled, `mamba-cache-mode=align` |
| Speculative decoding | MTP, 3 speculative tokens |
| Graph mode | `FULL_DECODE_ONLY` |
| Scheduling | vLLM asynchronous scheduling |

Local deployment used asynchronous save/load, queue size 8, a 256 MiB per-rank
GDN snapshot budget, and both experimental GDN environment opt-ins. The source
supports those environment aliases and equivalent explicit configuration:

```yaml
local_cpu: true
max_local_cpu_size: 32
chunk_size: 256
store_async: true
store_async_max_queue_size: 8
enable_async_loading: true
use_layerwise: false
enable_blending: false
save_decode_cache: false
save_unfull_chunk: false
min_retrieve_tokens: 0
extra_config:
  gdn_async_store: true
  gdn_async_restore: true
  gdn_async_staging_mib: 256
```

The deployment results below describe the measured combined local configuration;
they are not an isolated causal ablation of one source-code change.

## Local deployment validation

The deployed implementation was exercised with NPU prefix reuse and CPU restore
at concurrency 1, 2 and 4. CPU-path verification clears only NPU cache and
retains the CPU tier before the measured request. Each CPU-restore request has:

- Local NPU hit delta 0 and external hit delta 53,760 tokens.
- Response usage reporting 53,760 cached tokens.
- Both TP ranks reporting one completed receive per request.
- Zero measured preemption and checkpoint-skip deltas in the verified waves.

A separate 65,536-token restore smoke reported 61,440 external hit tokens and
zero local NPU hits after clearing NPU cache only.

### Performance methodology

Input: 65,280 tokens. Output: 256 tokens. Actual cache reuse: 53,760 tokens per
request (82.35%), with 11,520 tokens recomputed. Fixed barrier-started waves,
one warmup and three measured waves at concurrency 1/2/4. Cold preparation,
cache reset and drain are outside the timed request window. Prompts are
synthetic token sequences. TPOT is a streamed-text timing approximation because
MTP can emit multiple tokens per event; output throughput covers the measured
wave windows. Reported P95 uses a small sample.

### CPU restore

| Concurrency | Sync TTFT P50/P95 (s) | Async TTFT P50/P95 (s) | TTFT P50 reduction | TTFT P95 reduction | Sync/async output (token/s) | Output increase |
| --- | --- | --- | ---: | ---: | --- | ---: |
| 1 | 3.831 / 3.838 | 3.861 / 3.861 | -0.78% | -0.61% | 25.46 / 30.78 | +20.88% |
| 2 | 7.135 / 7.557 | 6.419 / 7.558 | +10.04% | -0.01% | 35.44 / 42.06 | +18.66% |
| 4 | 12.480 / 15.708 | 11.964 / 14.703 | +4.13% | +6.40% | 42.56 / 52.69 | +23.80% |

| Concurrency | Sync/async TPOT P50 (ms/token) | TPOT reduction | Sync/async E2E P50 (s) | E2E reduction |
| --- | --- | ---: | --- | ---: |
| 1 | 24.13 / 16.11 | +33.22% | 9.984 / 7.970 | +20.18% |
| 2 | 28.54 / 22.28 | +21.92% | 14.179 / 11.736 | +17.23% |
| 4 | 45.86 / 27.65 | +39.70% | 23.335 / 19.018 | +18.50% |

Reduction = (sync - async) / sync; output increase = (async - sync) / sync.
Percentages use unrounded original measurements. Negative reductions are
regressions, including single-request TTFT; they are not hidden as improvements.

### NPU prefix reuse

| Concurrency | Sync TTFT P50/P95 (s) | Async TTFT P50/P95 (s) | Sync/async TPOT P50 (ms/token) | Sync/async E2E P50 (s) | Sync/async output (token/s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.803 / 3.808 | 3.521 / 3.523 | 24.11 / 16.23 | 9.945 / 7.661 | 25.60 / 31.88 |
| 2 | 7.052 / 7.494 | 5.864 / 7.002 | 28.16 / 22.38 | 14.099 / 11.140 | 35.75 / 44.04 |
| 4 | 12.024 / 15.237 | 10.625 / 13.549 | 45.81 / 28.15 | 22.798 / 17.917 | 43.23 / 55.96 |

| Concurrency | TTFT P50 reduction | TTFT P95 reduction | Output increase |
| --- | ---: | ---: | ---: |
| 1 | +7.42% | +7.48% | +24.53% |
| 2 | +16.85% | +6.57% | +23.18% |
| 4 | +11.63% | +11.07% | +29.45% |

## Automated source regression

The initial validation passed 30 source-bound regressions on Python 3.12: 9 native
multimodal lifecycle tests and 21 async tests. They exercise production source using explicit
framework/device doubles. Covered behaviors include configuration precedence,
thread-affine RPC creation/use/close, pending-versus-miss polling, cancellation,
admitted lease ownership, factory/manager delegation, private snapshot capture,
staging budget, shared KV/state store accounting, both-rank receive ACK timing,
fence-before-release and retained ownership on failed transfer fencing.

```bash
python -m unittest discover -s tests/host -v
```

Ruff 0.11.7 lint/format, isort 6.0.1, codespell 2.4.1, SPDX and Python
compilation checks passed locally. A dependency-free host CI workflow covers
Python 3.10 and 3.12.

### Review follow-up validation

The host suite now has 45 passing tests: 9 native multimodal, 28 async protocol
tests, and 8 correctness-client tests. The additions reproduce late-abort
ordering, bounded lookup queue pressure, deferred cancellation, and retryable
shutdown. The client tests use HTTP/device doubles and are not model evaluations.

A concurrent long-context passkey evaluation is provided in
[gdn_async_correctness.md](gdn_async_correctness.md). The real-device sync/async
correctness comparison is pending execution; the earlier latency measurements
do not establish answer correctness under concurrent CPU restore.

## Source integration

The local deployed implementation and the repository code use the same core
lookup/receive and checkpoint data structures, integrated through explicit
engine/adapter helpers and config-scoped factory/manager classes.

| Core implementation | Repository module | Source check |
| --- | --- | --- |
| `AsyncStateClient` | `v1/gdn_async_lookup.py` | Initial class matched the retained local source; queue-pressure handling was added during review and is covered by host regressions |
| `Receive`, `acquire_load`, `make_receive` | `integration/vllm/gdn_async_receive.py` | Identical symbol ASTs to retained local delivery source |
| `FrozenExecution`, `StateJob` | `v1/gdn_async_store.py` | Identical class ASTs to retained local delivery source |

Additional integration changes are covered by the host tests: shared opt-in
configuration with explicit-value precedence, ordinary-path factory/manager
delegation, a transfer lock held through completion fencing, bounded atomic
completion accounting and retryable lookup-owner shutdown after queue pressure.
