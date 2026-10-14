# DeepSeek V4 (Flash) Cross-Node P2P Deployment Guide

Deploy two independent DeepSeek-V4-Flash instances, each using TP8 / DP1 /
expert parallelism, and share KV cache through LMCache **inprocess** P2P
over HCCL. Each TP rank saves and retrieves its own cache. Both instances
can serve as a cache source or read from their peer.

This guide uses synchronous LMCache loading and storage, NPU delayed pull,
and HCCL Host staging. An HTTP router distributes requests between the two
complete inference instances.

The deployment baseline was exercised on two Ascend 910B (A2) servers,
with eight NPUs per server. The examples below consolidate that configuration;
other hardware and software combinations require separate validation.

## 1. Environment and Topology

| Component | Configuration |
|---|---|
| Image | `quay.io/ascend/vllm-ascend:v0.23.0` |
| Model | `DeepSeek-V4-Flash-w8a8-mtp` (non-DSpark) |
| LMCache | Official `v0.4.5` |
| LMCache-Ascend | `dsv4_support_045` |
| Router | `vllm-router==0.1.15` |
| Per-node parallelism | TP8 / DP1 / EP, local NPUs 0–7 |
| MTP | One draft token, draft eager execution |
| vLLM prefix caching | Disabled to validate P2P reuse without local NPU prefix-cache hits masking its effect |

Use the same image, source commits, submodules, and dependencies on both
nodes. A branch name alone does not pin a version: record `git rev-parse HEAD`
in both repositories when collecting results.

### 1.1 Addresses and Paths

Replace these placeholders throughout the examples:

| Placeholder | Meaning |
|---|---|
| `<NODE_A_IP>` / `<NODE_B_IP>` | Host IPv4 addresses reachable from the peer |
| `<NIC_NAME>` | Communication NIC on this node; names may differ between nodes |
| `<USER_ID>` | User-specific workspace name |
| `<HOST_MODEL_DIR>` | Host directory containing `DeepSeek-V4-Flash-w8a8-mtp/` |

The workspace is `/mnt/sdb/<USER_ID>`, and scripts/configs live in its
`p2p/` subdirectory. Model files are mounted under `/models` in the container.

Identify the NIC whose address is reachable from the other node; it need
not carry the default route. Run these commands on the host if the container
does not include `ip`:

```bash
ip -br -4 addr
ip route get "<PEER_IP>"
ping -c 3 "<PEER_IP>"
```

Use an address actually bound to that NIC for `HCCL_IF_IP` and `p2p_host`.
The host IP connectivity checks do not replace verification of the NPU
communication network required by HCCL.

### 1.2 Services and Ports

| Service | Node A | Node B |
|---|---|---|
| vLLM HTTP API | 8010 | 8011 |
| P2P init | 8200–8207 | 8300–8307 |
| P2P lookup | 8210–8217 | 8310–8317 |
| LMCache worker | 8500–8507 | 8600–8607 |
| LMCache controller | HTTP 9000, pull 9800, reply 9900 | Connects to Node A |
| Router | HTTP 8888, Prometheus 8081 | Not required |

Allow these service ports between the relevant hosts and clients, and
configure the HCCL communication network for the platform. The table lists
the configured application endpoints, not every runtime HCCL connection.
Different physical hosts may reuse port numbers; separate ranges here make
logs and node configurations easier to distinguish.

## 2. Start the Containers

Run on each host after replacing the paths:

```bash
BASE_DIR="/mnt/sdb/<USER_ID>"
HOST_MODEL_DIR="<HOST_MODEL_DIR>"
IMAGE="quay.io/ascend/vllm-ascend:v0.23.0"
mkdir -p "$BASE_DIR/p2p/logs"

docker run -itd \
    --name vllm-ascend-p2p \
    --shm-size=512g \
    --net=host \
    --privileged \
    --device /dev/davinci0 --device /dev/davinci1 \
    --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 \
    --device /dev/davinci6 --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v "$BASE_DIR:$BASE_DIR" \
    -v "$HOST_MODEL_DIR:/models:ro" \
    -w "$BASE_DIR" \
    "$IMAGE" /bin/bash

docker exec -it vllm-ascend-p2p /bin/bash
```

Adapt driver/tool mounts to the host installation. `--net=host` exposes the
host's network interfaces to the container. Shared-memory capacity is not a
CPU cache capacity limit; the host still needs enough available memory for
caches, staging arenas, model loading, and other processes.

## 3. Install from Source

Run inside both containers:

```bash
cd "/mnt/sdb/<USER_ID>"
git clone -b v0.4.5 https://github.com/LMCache/LMCache.git
cd LMCache
NO_CUDA_EXT=1 python3 -m pip install -v --no-build-isolation --no-deps -e .
cd ..

git clone --recurse-submodules -b dsv4_support_045 \
    https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
git submodule update --init --recursive
```

Before building LMCache-Ascend, confirm that the `third_party/hcomm`
revision is compatible with the container's CANN version. If a different
revision is needed, select a matching revision from the
[hcomm repository](https://gitcode.com/cann/hcomm) on both nodes and record it.
Do not infer the active CANN version solely from the presence of an old
directory under `/usr/local/Ascend`. Then install:

```bash
python3 -m pip install -v --no-build-isolation --no-deps -e .
```

`NO_CUDA_EXT=1` above applies to this LMCache v0.4.5 inprocess installation.
`--no-deps` preserves the image's scientific stack; install any missing
dependencies in versions compatible with the image.

Verify package locations and record source revisions:

```bash
python3 -c 'import lmcache, lmcache_ascend; print(lmcache.__file__); print(lmcache_ascend.__file__)'
git -C "/mnt/sdb/<USER_ID>/LMCache" rev-parse HEAD
git -C "/mnt/sdb/<USER_ID>/LMCache-Ascend" rev-parse HEAD
git -C "/mnt/sdb/<USER_ID>/LMCache-Ascend" submodule status --recursive
```

On **Node A only**, install the router directly into the container:

```bash
python3 -m pip install --only-binary=vllm-router 'vllm-router==0.1.15'
vllm-router --help
```

This requests a prebuilt router wheel. If no wheel matches the container,
resolve that platform mismatch before proceeding. The router does not load
the model or need its own NPU allocation.

## 4. Configure LMCache on Both Nodes

Create `/mnt/sdb/<USER_ID>/p2p/` on both nodes and save the following files.

### 4.1 Node A: `lmcache-p2p-a.yaml`

```yaml
chunk_size: 1024
local_cpu: true
max_local_cpu_size: 50
enable_async_loading: false
store_async: false
use_layerwise: false
numa_mode: "auto"
save_unfull_chunk: false

enable_p2p: true
p2p_host: "<NODE_A_IP>"
p2p_init_ports: [8200, 8201, 8202, 8203, 8204, 8205, 8206, 8207]
p2p_lookup_ports: [8210, 8211, 8212, 8213, 8214, 8215, 8216, 8217]
transfer_channel: "hccl"
p2p_use_npu: true
p2p_pull_mode: true
p2p_delay_pull: true
p2p_npu_buffer_size: 134217728

enable_controller: true
lmcache_worker_ids: [0, 1, 2, 3, 4, 5, 6, 7]
lmcache_instance_id: "lmcache_colocated_a"
controller_pull_url: "<NODE_A_IP>:9800"
controller_reply_url: "<NODE_A_IP>:9900"
lmcache_worker_ports: [8500, 8501, 8502, 8503, 8504, 8505, 8506, 8507]

extra_config:
  save_only_first_rank: false
  use_host_staging: true
  os_staging_bytes: 8589934592
  lookup_backoff_time: 0.001
```

### 4.2 Node B: `lmcache-p2p-b.yaml`

```yaml
chunk_size: 1024
local_cpu: true
max_local_cpu_size: 50
enable_async_loading: false
store_async: false
use_layerwise: false
numa_mode: "auto"
save_unfull_chunk: false

enable_p2p: true
p2p_host: "<NODE_B_IP>"
p2p_init_ports: [8300, 8301, 8302, 8303, 8304, 8305, 8306, 8307]
p2p_lookup_ports: [8310, 8311, 8312, 8313, 8314, 8315, 8316, 8317]
transfer_channel: "hccl"
p2p_use_npu: true
p2p_pull_mode: true
p2p_delay_pull: true
p2p_npu_buffer_size: 134217728

enable_controller: true
lmcache_worker_ids: [0, 1, 2, 3, 4, 5, 6, 7]
lmcache_instance_id: "lmcache_colocated_b"
controller_pull_url: "<NODE_A_IP>:9800"
controller_reply_url: "<NODE_A_IP>:9900"
lmcache_worker_ports: [8600, 8601, 8602, 8603, 8604, 8605, 8606, 8607]

extra_config:
  save_only_first_rank: false
  use_host_staging: true
  os_staging_bytes: 8589934592
  lookup_backoff_time: 0.001
```

### 4.3 Memory and Transfer Semantics

- `max_local_cpu_size: 50` is per Worker: TP8 gives a configured CPU cache
  budget of 400 GiB per node. It is not a single shared 400 GiB allocator,
  nor necessarily 400 GiB of unique, non-replicated KV data.
- Host staging is additional memory. `os_staging_bytes` requests an 8 GiB
  arena per channel, rounded down to whole chunks. Account for all active
  ranks, the NPU buffers, and other allocations when sizing host memory.
- `save_only_first_rank: false` and explicit `lmcache_worker_ids` enable
  the all-rank setup. Do not retain `first_rank_max_local_cpu_size` from a
  first-rank-only configuration.
- NPU delayed pull requires both `p2p_use_npu: true` and
  `p2p_pull_mode: true`. In the current implementation, Host staging with
  NPU buffers also requires delayed pull. Keep this combination together.
- `enable_async_loading: false` controls LMCache loading. It does not mean
  disabling vLLM `--async-scheduling`, and delayed pull is a separate
  transfer behavior. This guide enables the latter two features.

## 5. Start the Services

Save the following scripts in `/mnt/sdb/<USER_ID>/p2p/`. Replace placeholders
before starting them. Use a fresh container shell with the image's
CANN/torch environment, without overrides from other LMCache experiments.

### 5.1 Shared Environment: `env.sh`

Use the same addresses and model path on both nodes. Set `NIC_NAME` to the
local communication NIC on each host.

```bash
#!/usr/bin/env bash
export WORK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NODE_A_IP="<NODE_A_IP>"
export NODE_B_IP="<NODE_B_IP>"
export NIC_NAME="<NIC_NAME>"
export MODEL_PATH="/models/DeepSeek-V4-Flash-w8a8-mtp"

case "${1:-}" in
  a)
    export NODE_IP="$NODE_A_IP" API_PORT=8010 NODE_NAME=a
    export LMCACHE_CONFIG_FILE="$WORK_DIR/lmcache-p2p-a.yaml"
    ;;
  b)
    export NODE_IP="$NODE_B_IP" API_PORT=8011 NODE_NAME=b
    export LMCACHE_CONFIG_FILE="$WORK_DIR/lmcache-p2p-b.yaml"
    ;;
  *) echo 'Usage: source env.sh a|b' >&2; return 2 ;;
esac

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HOST_IP="$NODE_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_IF_IP="$NODE_IP"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_TRACK_USAGE=false
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$NODE_A_IP,$NODE_B_IP"
export no_proxy="$NO_PROXY"
mkdir -p "$WORK_DIR/logs"
```

Preserve the image's library paths and `LD_PRELOAD`.
No hard-coded jemalloc preload is required
by this guide; if one is needed for a separate experiment, verify its path
and use `${LD_PRELOAD:+:$LD_PRELOAD}` when appending an existing value.

### 5.2 Controller on Node A: `start-controller.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh" a
lmcache_controller --host 0.0.0.0 --port 9000 \
    --monitor-ports '{"pull":9800,"reply":9900}' \
    2>&1 | tee "$WORK_DIR/logs/controller.log"
```

### 5.3 vLLM on Both Nodes: `start-vllm.sh`

Use this same script on both nodes: pass `a` on Node A and `b` on Node B.

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh" "${1:?Usage: bash start-vllm.sh a|b}"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --served-model-name dsv4 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 60 \
    --seed 1024 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' \
    --gpu-memory-utilization 0.85 \
    --block-size 128 \
    --no-disable-hybrid-kv-cache-manager \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":false}' \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector","kv_connector_extra_config":{"discard_partial_chunks":true}}' \
    2>&1 | tee "$WORK_DIR/logs/instance_${NODE_NAME}.log"
```

`--no-enable-prefix-caching` disables vLLM's local NPU prefix cache for this
test so that local prefix-cache hits do not mask the effect of P2P KV reuse.
Disabling it is a test choice, not a requirement for enabling P2P; LMCache's
local CPU cache remains enabled.

The scheduling and capacity values reproduce the test baseline; they are
not universal requirements for enabling P2P. Keep them identical across
comparison runs. vLLM `--block-size` and LMCache `chunk_size` are different
settings. Record the resolved cache configuration from startup/metrics as
well as the launch arguments.

### 5.4 Router on Node A: `start-router.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/env.sh" a
vllm-router \
    --host 0.0.0.0 --port 8888 \
    --policy cache_aware \
    --intra-node-data-parallel-size 1 \
    --prometheus-host 0.0.0.0 --prometheus-port 8081 \
    --worker-urls "http://$NODE_A_IP:8010" "http://$NODE_B_IP:8011" \
    2>&1 | tee "$WORK_DIR/logs/router.log"
```

Use `cache_aware` for cache-affinity routing. Use the same routing policy in
all comparison groups.

### 5.5 Startup Order

Run each long-running command in its own container terminal. On Node A:

```bash
cd "/mnt/sdb/<USER_ID>/p2p"
bash start-controller.sh
```

Confirm controller startup in its log and check its listeners. A listening
socket alone does not prove it belongs to the newly started process:

```bash
ss -lntp | grep -E ':(9000|9800|9900)\b'
```

After the controller is ready, the two vLLM services may load in parallel:

```bash
# Node A
cd "/mnt/sdb/<USER_ID>/p2p"
bash start-vllm.sh a
```

```bash
# Node B
cd "/mnt/sdb/<USER_ID>/p2p"
bash start-vllm.sh b
```

After both report application startup completion, check the APIs and
controller registration logs on Node A:

```bash
cd "/mnt/sdb/<USER_ID>/p2p"
source ./env.sh a
curl --noproxy '*' -fsS "http://$NODE_A_IP:8010/v1/models"
curl --noproxy '*' -fsS "http://$NODE_B_IP:8011/v1/models"
grep -Ei 'register|lmcache_colocated' "$WORK_DIR/logs/controller.log"
```

Confirm both instance IDs and workers 0–7 for each instance, not just two
instance names. Check each vLLM log for the all-rank settings, Host staging,
and delayed-pull initialization. Finally, on Node A:

```bash
cd "/mnt/sdb/<USER_ID>/p2p"
bash start-router.sh
```

Stop foreground services with Ctrl-C in their own terminals. For background
services, identify and terminate their specific PIDs. Do not use a blanket
`pkill -9 -f vllm` as a routine startup step. Preserve logs before restarting,
since these examples reuse the same log filenames.

## 6. Verify Cross-Node KV Reuse

Send one identical request directly to Node A and then Node B, bypassing
the router for deterministic placement. Use a fresh random prefix and make
sure neither node has already processed this prompt.

On Node A, create a request and use the running server's `/tokenize` endpoint
to verify its length with the same tokenizer used for inference:

```bash
cd "/mnt/sdb/<USER_ID>/p2p"
source ./env.sh a
python3 - <<'PY'
import json
import os
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

prompt = "P2P test " + uuid.uuid4().hex + "\n"
prompt += "Explain how a cache reuses previously computed information. " * 512
request = Request(
    f'http://{os.environ["NODE_A_IP"]}:8010/tokenize',
    data=json.dumps({"model": "dsv4", "prompt": prompt}).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=60) as response:
    token_count = json.load(response)["count"]
assert token_count > 2048, f"Prompt too short: {token_count} tokens"
body = {"model": "dsv4", "prompt": prompt,
        "max_tokens": 32, "temperature": 0}
path = Path(os.environ["WORK_DIR"]) / "logs" / "p2p-request.json"
path.write_text(json.dumps(body))
print("Prompt tokens:", token_count, "Request:", path)
PY

curl --noproxy '*' -fsS "http://$NODE_A_IP:8010/v1/completions" \
    -H 'Content-Type: application/json' \
    -d @"$WORK_DIR/logs/p2p-request.json" \
    -o "$WORK_DIR/logs/p2p-response-a.json"
```

Confirm Node A's successful storage for this request before sending to B.
This configuration has `store_async: false`; an arbitrary sleep is not a
substitute for checking storage and controller visibility.

```bash
grep -E 'Stored|P2P|ERROR|WARNING' "$WORK_DIR/logs/instance_a.log" | tail -40

curl --noproxy '*' -fsS "http://$NODE_B_IP:8011/v1/completions" \
    -H 'Content-Type: application/json' \
    -d @"$WORK_DIR/logs/p2p-request.json" \
    -o "$WORK_DIR/logs/p2p-response-b.json"
```

On Node B, inspect the corresponding time window:

```bash
cd "/mnt/sdb/<USER_ID>/p2p"
grep -E 'P2P|Retrieved|Total tokens|ERROR|WARNING' logs/instance_b.log | tail -80
```

Validate the complete sequence: A stores KV, B finds A's prefix, B completes
the peer transfer and KV load, and B finishes the request. Check the TP worker
records. With delayed pull, lookup/retrieve can produce a proxy before the
actual transfer finishes, so a lookup hit alone does not prove a successful
transfer. If INFO logs do not show enough detail, temporarily enable DEBUG
for a smoke run, then return to INFO for performance measurements.

`LMCache hit tokens`, a faster response, and identical output text are not
individually sufficient proof of P2P reuse. vLLM's external-cache counters
also include local CPU reuse. Use fresh-prefix placement, source/receiver
logs, completed loading, and metrics together.

## 7. Multi-Round Benchmark

Run the following command on Node A after both inference services and the
router are ready. Requests go through the router to both instances.

```bash
python3 -u "/mnt/sdb/<USER_ID>/LMCache/benchmarks/multi_round_qa/multi-round-qa.py" \
    --num-users 20 \
    --num-rounds 15 \
    --qps 0.8 \
    --shared-system-prompt 2000 \
    --user-history-prompt 25000 \
    --answer-len 512 \
    --model dsv4 \
    --base-url "http://<NODE_A_IP>:8888/v1" \
    --time 1200 \
    --enforce-strict-concurrent-users \
    --disable-ramp-up \
    --output "/mnt/sdb/<USER_ID>/p2p/logs/multiround.csv"
```

This configuration uses up to 20 active users, up to 15 rounds per session,
a target QPS of 0.8, and a 1200-second main test phase. The shared prompt and
user-history length parameters are 2000 and 25000; each response is limited
to 512 tokens. Strict user limiting is enabled and ramp-up is disabled.

The workload is synthetic multi-round conversation. Completed sessions can
be replaced during the test, so the total request count is not fixed at
20 × 15. MTP remains enabled in the inference services.

> **Note:** With vLLM's local NPU prefix caching disabled, the external-cache
> hit rate can be observed without local NPU prefix-cache hits masking reuse.
> A higher external-cache hit rate with P2P enabled than with local CPU caching
> alone, under the same workload, routing policy, cache capacity, and initial
> cache state, is evidence that P2P improves KV reuse. This rate includes both
> local CPU and peer-cache hits; confirm completed P2P transfers as described
> in Section 6.
