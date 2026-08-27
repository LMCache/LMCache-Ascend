# DeepSeek V4 (Flash) Cross-Node P2P Deployment Guide

Deploy DeepSeek-V4-Flash as two colocated vLLM instances on two Ascend
servers, with KV cache shared across nodes through LMCache P2P (over HCCL).
Node A hosts both the LMCache controller and a vLLM instance; Node B hosts a
second vLLM instance that registers with the controller and pulls KV cache
from Node A.

> **Hardware validation scope**: This guide has been verified only on
> Ascend 910B (A2 / A3), with one node providing 8 NPUs per instance.

## 1. Environment Preparation

### 1.1 Topology

| Node | Role | vLLM port | Notes |
|---|---|---|---|
| Node A | LMCache controller + vLLM instance | 8010 | Starts the controller first |
| Node B | vLLM instance only | 8011 | Registers with the controller on Node A |

Both nodes must be on the same internal subnet. On each node, identify the
NIC and internal IP used for cross-node traffic:

```bash
ip addr | grep -E "inet |^[0-9]+: "
ip route | grep default
```

Use the internal IP bound to the NIC that carries the default route. Do not
use public/NAT addresses: LMCache resolves `HCCL_IF_IP` from the NIC name,
and addresses not bound to the NIC will fail.

Verify connectivity from Node A to Node B (and vice versa):

```bash
ping -c 3 <NODE_B_IP>
```

Record `NODE_A_IP`, `NODE_B_IP`, and `NIC_NAME` (both nodes usually share
the same NIC name); they are referenced throughout this guide.

### 1.2 Port Requirements

Open the following TCP ports on both nodes (security groups / firewall):

| Port | Purpose |
|---|---|
| 8010, 8011 | vLLM API servers |
| 9000, 9800, 9900 | LMCache controller (Node A listens, Node B connects) |
| 8200-8217 | P2P init / lookup |
| 8500-8517 | LMCache workers |

```bash
firewall-cmd --add-port=8010-8011/tcp --permanent
firewall-cmd --add-port=9000/tcp --permanent
firewall-cmd --add-port=8200-8517/tcp --permanent
firewall-cmd --add-port=9800-9900/tcp --permanent
firewall-cmd --reload
```

If the ports toward the controller are blocked, Node B cannot register and
the cross-node hit rate stays at zero.

---

## 2. Docker Container Startup

Run the same container image on both nodes:

```bash
#!/bin/bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-a3
docker run \
    --name vllm-ascend-p2p \
    --shm-size=512g \
    --net=host \
    --privileged=true \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /mnt/sdb/<USER_ID>:/mnt/sdb/<USER_ID> \
    -it $IMAGE bash
```

`--net=host` is required: cross-node HCCL and LMCache P2P traffic rely on
the host network stack.

---

## 3. Install from Source (Inside the Container)

Perform this section on **both** nodes. Replace `<USER_ID>` with your
identifier.

### 3.1 Install LMCache

```bash
cd /mnt/sdb/<USER_ID>
git clone -b v0.4.5 https://github.com/LMCache/LMCache.git
cd LMCache
export NO_CUDA_EXT=1
python3 -m pip install -v --no-build-isolation --no-deps -e .
python3 -m pip install sortedcontainers
cd ..
```

> `--no-deps` prevents pip from upgrading the scientific stack shipped in
> the container (torch, numpy, etc.). Install missing pure-Python
> dependencies explicitly instead of removing it.

### 3.2 Install LMCache-Ascend

```bash
cd /mnt/sdb/<USER_ID>
git clone --recurse-submodules -b dsv4_support_045 https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
pip install -v --no-build-isolation -e .
```

> **Important — `third_party/hcomm` must match the container's CANN version.**
> Before running `pip install`, check the CANN version
> (`ls /usr/local/Ascend`, e.g. `8.5.0` means CANN 8.5) and, if needed,
> switch the submodule to the matching tag from <https://gitcode.com/cann/hcomm>:
> ```bash
> cd third_party/hcomm
> git fetch --tags
> git checkout v8.5.0   # use the tag matching your CANN version
> cd ../..
> pip install -v --no-build-isolation -e .
> ```

Verify that both nodes resolve the packages from the same source trees:

```bash
python3 -c "import lmcache, lmcache_ascend; print(lmcache.__file__); print(lmcache_ascend.__file__)"
```

---

## 4. Service Startup Configuration

All scripts and configs live under `/mnt/sdb/<USER_ID>/p2p` on both nodes.

### 4.1 LMCache Config Files

Create `/mnt/sdb/<USER_ID>/p2p/lmcache-p2p-a.yaml` on Node A:

```yaml
chunk_size: 1024
local_cpu: true
max_local_cpu_size: 16
enable_async_loading: true
use_layerwise: false
numa_mode: "auto"
save_unfull_chunk: false

# Cross-node P2P KV sharing
enable_p2p: true
p2p_host: "<NODE_A_IP>"
p2p_init_ports: [8200, 8201, 8202, 8203, 8204, 8205, 8206, 8207]
p2p_lookup_ports: [8210, 8211, 8212, 8213, 8214, 8215, 8216, 8217]
transfer_channel: "hccl"
p2p_use_npu: true
p2p_pull_mode: true
p2p_delay_pull: false
p2p_npu_buffer_size: 134217728

enable_controller: true
lmcache_instance_id: "lmcache_colocated_a"
controller_pull_url: "<NODE_A_IP>:9800"
controller_reply_url: "<NODE_A_IP>:9900"
lmcache_worker_ports: [8500, 8501, 8502, 8503, 8504, 8505, 8506, 8507]

extra_config:
  save_only_first_rank: true
  lookup_backoff_time: 0.001
```

Create `/mnt/sdb/<USER_ID>/p2p/lmcache-p2p-b.yaml` on Node B. It is
identical except for two fields:

```yaml
p2p_host: "<NODE_B_IP>"                # this node's own IP
lmcache_instance_id: "lmcache_colocated_b"
```

`controller_pull_url` / `controller_reply_url` still point to Node A on both
nodes. Because the nodes have separate network namespaces, the P2P and
worker port ranges may be identical on both sides.

### 4.2 Node A Startup Script

Save as `/mnt/sdb/<USER_ID>/p2p/start_node_a.sh` on Node A:

```bash
#!/bin/bash
set -euo pipefail

NODE_IP="<NODE_A_IP>"          # this node's internal IP
NIC_NAME="<NIC_NAME>"
MODEL_PATH="/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp"
WORK_DIR="/mnt/sdb/<USER_ID>/p2p"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=123
export LMCACHE_TRACK_USAGE=false
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME
export HCCL_IF_IP=$NODE_IP
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LMCACHE_CONFIG_FILE="${WORK_DIR}/lmcache-p2p-a.yaml"

mkdir -p "${WORK_DIR}/logs"

# LMCache controller (Node A only)
nohup lmcache_controller \
    --host 0.0.0.0 \
    --port 9000 \
    --monitor-ports '{"pull": 9800, "reply": 9900}' \
    > "${WORK_DIR}/logs/controller.log" 2>&1 &
sleep 2

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8010 \
    --served-model-name dsv4 \
    --no-enable-prefix-caching \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --api-server-count 1 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}' \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
    --gpu-memory-utilization 0.8 \
    --block-size 128 \
    --no-disable-hybrid-kv-cache-manager \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector","kv_connector_extra_config":{"discard_partial_chunks":true}}' \
    2>&1 | tee "${WORK_DIR}/logs/instance_a.log"
```

### 4.3 Node B Startup Script

Save as `/mnt/sdb/<USER_ID>/p2p/start_node_b.sh` on Node B. It is identical
to the Node A script except for the node-specific variables and the port; it
does **not** start the controller:

```bash
NODE_IP="<NODE_B_IP>"          # this node's internal IP
...
export LMCACHE_CONFIG_FILE="${WORK_DIR}/lmcache-p2p-b.yaml"
...
vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8011 \
    ...
    2>&1 | tee "${WORK_DIR}/logs/instance_b.log"
```

### 4.4 Startup Procedure

1. Clean up residuals on both nodes (a leftover controller holding port
   9800 prevents the new controller from starting, which silently disables
   cross-node sharing):

   ```bash
   pkill -9 -f vllm
   pkill -9 -f lmcache_controller
   sleep 3
   npu-smi info   # all NPUs should be idle
   ```

2. Start Node A and wait for `Application startup complete` (model loading
   and NPU graph compilation take 10-20 minutes on first start):

   ```bash
   bash /mnt/sdb/<USER_ID>/p2p/start_node_a.sh
   ```

   Verify the controller and the API server:

   ```bash
   ss -lntp | grep -E "9000|9800|9900"        # all three ports in LISTEN
   curl http://<NODE_A_IP>:8010/v1/models
   ```

3. Start Node B only after Node A is ready:

   ```bash
   bash /mnt/sdb/<USER_ID>/p2p/start_node_b.sh
   curl http://<NODE_B_IP>:8011/v1/models
   ```

4. Confirm that both instances registered with the controller (on Node A):

   ```bash
   grep -i "registered" /mnt/sdb/<USER_ID>/p2p/logs/controller.log
   # expect both lmcache_colocated_a and lmcache_colocated_b
   ```

---

## 5. Verification and Benchmark

### 5.1 Smoke Test (Cross-Node KV Sharing)

Send the **same** request body to Node A first, then to Node B. The prompt
must exceed `chunk_size` (1024 tokens): with `save_unfull_chunk: false`,
a shorter prompt is computed but never stored, so nothing can be shared.

```bash
python3 - <<'PY'
import json
prompt = 'Explain KV cache P2P sharing across Ascend instances. ' * 128  # > 1024 tokens
body = {"model":"dsv4","messages":[{"role":"user","content":prompt}],"max_tokens":16,"temperature":0}
json.dump(body, open('/tmp/req.json','w'))
PY

curl -s http://<NODE_A_IP>:8010/v1/chat/completions \
  -H "Content-Type: application/json" -d @/tmp/req.json > /dev/null

sleep 10   # allow the async store on Node A to finish

curl -s http://<NODE_B_IP>:8011/v1/chat/completions \
  -H "Content-Type: application/json" -d @/tmp/req.json > /dev/null
```

Check the logs:

```bash
# Node A: KV stored
grep -E "Stored" /mnt/sdb/<USER_ID>/p2p/logs/instance_a.log | tail -3
# e.g. "Stored 1024 out of total 1024 tokens. size: 0.0133 GB"

# Node B: cross-node hit
grep -E "Total tokens" /mnt/sdb/<USER_ID>/p2p/logs/instance_b.log | tail -3
# e.g. "Total tokens 1541, ... LMCache hit tokens: 1024, need to load: 1024"
```

`LMCache hit tokens: 1024` on Node B (with no prior local compute) confirms
that the KV chunk computed by Node A was retrieved over P2P; only the
remaining suffix tokens were prefilled by Node B.

### 5.2 Multi-Round Conversation Bench

Run against either endpoint (Node A shown):

```bash
python3 /mnt/sdb/<USER_ID>/LMCache/benchmarks/multi_round_qa/multi-round-qa.py \
    --num-users 10 \
    --num-rounds 8 \
    --qps 0.3 \
    --shared-system-prompt 1000 \
    --user-history-prompt 8000 \
    --answer-len 150 \
    --model dsv4 \
    --base-url http://<NODE_A_IP>:8010/v1 \
    --time 1200
```

Per-request hit statistics can be observed live on either node:

```bash
grep -E "Total tokens" /mnt/sdb/<USER_ID>/p2p/logs/instance_b.log | tail -f
```

---

## 6. Tips

> **Operational notes from validation:**
> 1. **Always kill the old controller before restarting** —
>    `pkill -9 -f lmcache_controller`. A stale controller on port 9800 is
>    the most common cause of a zero cross-node hit rate.
> 3. **Smoke-test prompts must exceed `chunk_size`** — otherwise nothing is
>    stored (`save_unfull_chunk: false`) and the second request cannot hit.
> 4. **Transient HCCL init failures** — if a worker aborts during warmup
>    with `aclnnMoeDistributeDispatchV4 ... error 507018` /
>    `HcclAllocComResourceByTiling ret = 15`, kill all processes and restart
>    the container; this clears leaked AICPU communication resources.
> 5. **Keep both nodes on identical code and image** — verify with
>    `git rev-parse HEAD` in both source trees and
>    `docker images --digests` on both hosts.
