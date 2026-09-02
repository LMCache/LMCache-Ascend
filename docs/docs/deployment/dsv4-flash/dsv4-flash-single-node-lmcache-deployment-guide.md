# DeepSeek V4 (Flash) Single-Node Deployment Guide

Deploy DeepSeek-V4-Flash as a single vLLM instance on one Ascend server,
with LMCache-Ascend providing a CPU-memory (DDR) KV cache backend, and
compare it against the native HBM prefix-caching baseline.

> **Hardware validation scope**: verified on a single Ascend server; the
> serving scripts use 8 NPUs (`--data-parallel-size 2 --tensor-parallel-size 4`).

## 1. Environment Preparation

Create a workspace on the host. Replace `<USER_ID>` with your identifier:

```bash
mkdir -p /mnt/sdb/<USER_ID>
```

The directory is mounted into the container at the same path and holds
the source trees, configuration files, and benchmarks. Confirm the model
paths mounted from the host (e.g. `/mnt/sdc/models`) exist before
starting the container.

---

## 2. Docker Container Startup

```bash
#!/bin/bash
# NPU devices exposed to the container (all 16)
export DEVICE_LIST="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

docker run -itd \
  --name lmcache-ds-<USER_ID> \
  --privileged \
  --net=host \
  --shm-size 128g \
  --restart unless-stopped \
  --cap-add=SYS_RESOURCE \
  --cap-add=IPC_LOCK \
  -w /mnt/sdb/<USER_ID>/ds-workspace \
  -e ASCEND_VISIBLE_DEVICES="${DEVICE_LIST}" \
  -e ASCEND_RT_VISIBLE_DEVICES="${DEVICE_LIST}" \
  -e ASCEND_TOTAL_MEMORY_GB=64 \
  -e VLLM_TARGET_DEVICE=npu \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /var/log/npu:/var/log/npu \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/localtime:/etc/localtime \
  -v /root/.cache:/root/.cache \
  -v /mnt/sdc/models:/mnt/sdc/models \
  -v /mnt/sdb/models:/mnt/sdb/models \
  -v /mnt/sdb/<USER_ID>:/mnt/sdb/<USER_ID> \
  quay.io/ascend/vllm-ascend:v0.18.0-a3 \
  bash
```

---

## 3. Install from Source (Inside the Container)

Perform this section inside the container.

### 3.1 Install LMCache

```bash
cd /mnt/sdb/<USER_ID>
git clone -b v0.4.4 https://github.com/LMCache/LMCache.git
cd LMCache
export NO_CUDA_EXT=1
# add -i https://pypi.tuna.tsinghua.edu.cn/simple if PyPI is slow
python3 -m pip install -v --no-build-isolation -e .
cd ..
```

### 3.2 Install LMCache-Ascend

```bash
cd /mnt/sdb/<USER_ID>
git clone --recurse-submodules -b dsv4_support https://github.com/larksudo/LMCache-Ascend.git
cd LMCache-Ascend
python3 -m pip install -v --no-build-isolation -e .
cd ..
```

---

## 4. Service Startup Configuration

### 4.1 LMCache Config File

Create `/mnt/sdb/<USER_ID>/ds-workspace/lmcache-config-ddr.yaml` (used by
the DDR variant in Section 4.3):

```yaml
chunk_size: 1024
local_cpu: true
max_local_cpu_size: 1            # max CPU (DDR) cache size in GB

extra_config:
    save_only_first_rank: false
    first_rank_max_local_cpu_size: 500
    broadcast_shard_size: 16
```

### 4.2 Baseline Script (HBM, native prefix caching)

```bash
#!/bin/sh
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

# Host tuning: performance governor, swap off, NUMA balancing off
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve /workspace/models/DeepSeek-V4-Flash-w8a8-mtp \
    --max_model_len 65536 \
    --max-num-batched-tokens 10240 \
    --api-server-count 1 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32 \
    --data-parallel-size 2 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --no-disable-hybrid-kv-cache-manager \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy 'prefetch' \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --quantization ascend \
    --port 8900 \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
    --async-scheduling \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_flashcomm1": false,
    "enable_dsa_cp": false,
    "enable_shared_expert_dp": false,
    "multistream_overlap_shared_expert":true}' > ds_base.log 2>&1
```

### 4.3 DDR Variant (LMCache CPU backend)

Identical to the baseline script except for:

```bash
export LMCACHE_CONFIG_FILE="/mnt/sdb/<USER_ID>/ds-workspace/lmcache-config-ddr.yaml"
```

```bash
vllm serve /workspace/models/DeepSeek-V4-Flash-w8a8-mtp \
    ...                                        # identical to the baseline script
    --kv-transfer-config '{
        "kv_connector": "LMCacheAscendConnectorV1Dynamic",
        "kv_role": "kv_both",
        "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    }' > ds_lmcache_ddr.log 2>&1              # distinct log name
```

---

## 5. Verification and Benchmark

### 5.1 Smoke Test

```bash
curl http://127.0.0.1:8900/v1/models
```

### 5.2 Multi-Round Conversation Bench

Evaluates incremental retrieval and hit behavior of the external cache in
long, multi-round conversations:

```bash
python3 /mnt/sdb/<USER_ID>/LMCache/benchmarks/multi_round_qa/multi-round-qa.py \
    --num-users 64 \
    --num-rounds 10 \
    --qps 0.8 \
    --shared-system-prompt 10000 \
    --user-history-prompt 30000 \
    --answer-len 300 \
    --model dsv4 \
    --base-url http://localhost:8900/v1 \
    --enforce-strict-concurrent-users \
    --time 1200
```

### 5.3 Prefix-Repetition Bench

Evaluates throughput and latency under high concurrency with many shared
prefixes:

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8900 \
  --served-model-name dsv4 \
  --num-prompts 1000 \
  --max-concurrency 64 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 16000 \
  --prefix-repetition-suffix-len 4000 \
  --prefix-repetition-output-len 500 \
  --prefix-repetition-num-prefixes 50
```

---

## 6. Tips

> **Operational notes from validation:**
> 1. **`PYTHONHASHSEED=0` is mandatory** — without it, per-process hash
>    randomization makes the same tokens hash differently across workers
>    and the LMCache hit rate stays at zero.
> 2. **Push concurrency until HBM evicts** — raise `--num-users` (e.g.
>    128/256) or `--max-concurrency` until the native HBM KV cache fills
>    and cold eviction starts; the LMCache memory/disk swap-in advantage
>    only becomes visible then.
> 3. **Isolate LMCache I/O** — add `--no-enable-prefix-caching` to
>    disable vLLM's internal HBM prefix matching and route all prefix
>    storage/retrieval through the LMCache connector.
