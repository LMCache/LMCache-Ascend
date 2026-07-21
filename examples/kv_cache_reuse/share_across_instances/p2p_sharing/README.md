# Ascend P2P KV cache sharing with vLLM v1

This example runs two LMCache-enabled vLLM instances that share KV cache over
RoCE.

## Ascend remote host-to-device transfer

Ascend P2P supports remote host-to-device (rH2D) transfer, in which KV stored
in a peer's host memory is pulled into an NPU transfer buffer instead of first
being received into the local CPU cache.

The supplied configurations combine rH2D with delayed pull and HostStaging:

```yaml
p2p_pull_mode: True
p2p_delay_pull: True
p2p_use_npu: True

extra_config:
  use_host_staging: True
```

Within the HostStaging implementation, delayed pull is the only supported rH2D
path. Non-delayed (eager) HostStaging supports CPU receive buffers only i.e. `p2p_use_npu: False`.

## Prerequisites

- Ascend HDK 25.5.0 or later drivers and firmware. Earlier releases can only
  register approximately 20 GB of host memory with the NPU NIC.
- A RoCE-connected NPU server. HCCS support is planned.
- At least two NPUs.
- Apply `docker/lmcache-controller.diff` to LMCache. This is required for P2P
  sharing with TP greater than 1.

Reinstall LMCache after applying the patch.

Use both `instance1.yaml` and `instance2.yaml`. They use the same transfer
settings so either instance can serve or retrieve KV; only their instance IDs
and ports differ.

## Delayed pull

On a cache hit, delayed pull returns lightweight proxy objects during lookup
instead of immediately allocating and filling buffers for the entire hit. When
vLLM consumes the KV, the NPU connector reads micro-batches into two alternating
NPU buffer pools. This overlaps the next RDMA read with the previous
micro-batch's scatter into paged KV cache and bounds receiver scratch-buffer use
by `p2p_npu_buffer_size`.

The expected performance differences below are hypotheses, not guarantees.
Verify them with the model, prompt lengths, concurrency, topology, and driver
version used in production.

| HostStaging mode | Expected advantage | Expected trade-off |
| --- | --- | --- |
| Delayed pull (`p2p_delay_pull: True`) | Enables rH2D and overlaps RDMA reads with KV scatter. | Transfer starts when KV is consumed and adds proxy, event, and micro-batch overhead. |

Delayed pull requires both `p2p_pull_mode: True` and `p2p_use_npu: True`.

## HostStaging for KV cache hits

With very large models, HCCL registration of the complete CPU KV cache can
exhaust device OS memory. This can appear as error code `19`; consult the Ascend
PLOG files under `$HOME/ascend/log` for the underlying registration error.

`use_host_staging: True` registers a bounded `os_staging_bytes` arena instead of
the complete CPU KV cache:

```yaml
extra_config:
  use_host_staging: True
  os_staging_bytes: 8589934592 # 8 GiB
```

When HostStaging is disabled or omitted, HCCL registers the full local CPU KV
cache and avoids the producer-side staging copy.

HostStaging currently requires `transfer_channel: "hccl"` and
`p2p_pull_mode: True`. HIXL is experimental and does not support HostStaging.

| Channel | CANN requirement | HostStaging support | Status |
| --- | --- | --- | --- |
| `hccl` | CANN 8.5 or later | Yes | Recommended |
| `hixl` | CANN 8.5 or later | No | Experimental |

### How HostStaging works

Each LMCache worker registers a bounded CPU arena instead of its full local CPU
cache. When that worker serves a pull, it copies the cache-hit chunks into the
arena and returns their registered references. The receiver reads those chunks
over RDMA. The slots are released after the receiver sends `Done`, or after the
pull lease expires.

This avoids full-cache registration at the cost of an additional CPU-to-CPU
copy on the producer and possible arena contention. Prefer
`use_host_staging: False` when full-cache registration is reliable and the
additional copy is material to performance.

### Supported HostStaging configurations

Delayed pull is a form of pull mode, not a separate transfer mode. HostStaging
supports two effective configurations:

| Configuration | `p2p_pull_mode` | `p2p_delay_pull` | `p2p_use_npu` | Result |
| --- | --- | --- | --- | --- |
| Delayed rH2D pull | `True` | `True` | `True` | Supported. KV is pulled into bounded NPU ping-pong buffers when consumed. |
| Eager CPU pull | `True` | `False` | `False` | Supported. KV is pulled through the receiver's registered CPU staging arena into final CPU objects. |
| Eager NPU pull | `True` | `False` | `True` | Unsupported with HostStaging; rejected at startup. |
| Push mode | `False` | Any | Any | Unsupported with HostStaging; rejected at startup. |

If `p2p_delay_pull: True` is combined with `p2p_use_npu: False`, the backend
warns and disables delayed pull. Configure eager CPU pull explicitly instead.

### Recommendation for `save_only_first_rank`

When using `save_only_first_rank: True` with MLA/DSA, use eager HostStaging pull
into CPU memory:

```yaml
p2p_pull_mode: True
p2p_delay_pull: False
p2p_use_npu: False

extra_config:
  save_only_first_rank: True
  use_host_staging: True
  os_staging_bytes: 8589934592
```

Only the first rank owns and retrieves from the storage backend in this mode.
Eager CPU pull provides materialized CPU KV objects that the first rank can load
and broadcast to the remaining ranks. 

### Size `os_staging_bytes`

`os_staging_bytes` is the HostStaging arena capacity per LMCache worker (TP
rank), not per vLLM instance or server. The backend default is 10 GiB when the
field is omitted; the supplied configurations use 8 GiB.

The simplest capacity estimate is:

```text
os_staging_bytes ~= concurrent_requests_per_rank
                    x average_hit_tokens
                    x KV_bytes_per_token_per_rank

```

Here, `concurrent_requests_per_rank` means cache-hit requests for which the
producer rank has staged KV but has not yet received `Done`. The value is often
1 as a baseline and grows when several remote cache-hit requests overlap.

Multiplying by the dtype size alone is insufficient because each token contains
many KV elements across layers, KV heads, head dimensions, and K/V tensors. If
avoiding truncation is important, replace `average_hit_tokens` with a p95 or
maximum expected hit length and add headroom for delayed or failed `Done`
signals.

The arena is rounded down to a whole aligned KV chunk and must hold at least one
chunk. If it cannot hold a complete hit, LMCache serves the prefix that fits and
recomputes the remaining tokens.

Budget the aggregate host memory as well. The supplied configurations pair an
8 GiB staging arena with a 16 GiB CPU cache per worker. With storage enabled on
all ranks, two TP=2 instances on one host can allocate four staging arenas
(32 GiB) and four CPU caches (64 GiB), in addition to normal process memory.
With `save_only_first_rank: True`, only the storage-owning first rank needs this
storage allocation.

## Run the example

Launch the controller:

```bash
PYTHONHASHSEED=123 lmcache_controller \
  --host 0.0.0.0 \
  --port 9000 \
  --monitor-ports '{"pull": 9800, "reply": 9900}'
```

Launch instance 1:

```bash
export LMCACHE_CONFIG_FILE=/workspace/LMCache-Ascend/examples/kv_cache_reuse/share_across_instances/p2p_sharing/instance1.yaml
export ASCEND_RT_VISIBLE_DEVICES=2,3
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=123
python \
  -m vllm.entrypoints.openai.api_server \
  --port 8010 \
  --model /data/models/Qwen/Qwen3-8B \
  --enforce-eager \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --disable-log-requests \
  --block-size 128 \
  --rope-scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
  --max-model-len 32768 \
  --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}' \
  > instance1.txt 2>&1
```

Launch instance 2 in another shell:

```bash
export LMCACHE_CONFIG_FILE=/workspace/LMCache-Ascend/examples/kv_cache_reuse/share_across_instances/p2p_sharing/instance2.yaml
export ASCEND_RT_VISIBLE_DEVICES=6,7
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=123
python \
  -m vllm.entrypoints.openai.api_server \
  --port 8011 \
  --model /data/models/Qwen/Qwen3-8B \
  --enforce-eager \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --disable-log-requests \
  --block-size 128 \
  --rope-scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
  --max-model-len 32768 \
  --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}' \
  > instance2.txt 2>&1
```

Populate the cache on instance 1:

```bash
time curl -X POST http://localhost:8010/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/data/models/Qwen/Qwen3-8B\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models in English.%.0s' {1..1000})\",
    \"max_tokens\": 10,
    \"temperature\": 0
  }"
```

Send the same prompt to instance 2:

```bash
time curl -X POST http://localhost:8011/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/data/models/Qwen/Qwen3-8B\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models in English.%.0s' {1..1000})\",
    \"max_tokens\": 10,
    \"temperature\": 0
  }"
```

Instance 2 should report that it retrieved the matching prompt tokens from
instance 1. A typical log line is:

```text
LMCache INFO: Retrieved 1002 out of total 1002 tokens. size: 0.1223 gb, cost 60.3595 ms, throughput: 2.0264 GB/s
```
