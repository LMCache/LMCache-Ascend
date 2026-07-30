## 1. Environment Preparation

Create the working directory on the host. Replace `<USER_ID>` with your identifier:

```bash
mkdir -p /mnt/sdb/<USER_ID>
```

> This directory is mounted into the container for storing source code, configs, and benchmarks.

---

## 2. Docker Container Startup

Launch the Ascend NPU vLLM container:

```bash
#!/bin/bash
# NPU card IDs (all 16 cards)
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
  quay.nju.edu.cn/ascend/vllm-ascend:v0.20.2rc1-openeuler \
  /bin/bash -c "echo \"export PS1='\[\e[1;32m\]\u@\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\\$ '\" >> ~/.bashrc && exec /bin/bash"
```

---

## 3. Install from Source (Inside the Container)

### 3.1 Install LMCache
```bash
git clone -b v0.4.5 https://github.com/LMCache/LMCache.git
cd LMCache
export NO_CUDA_EXT=1
python3 -m pip install -v --no-build-isolation -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..
```

### 3.2 Install LMCache-Ascend
```bash
git clone --recurse-submodules -b dsv4_support_045 https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
pip install -v --no-build-isolation -e .
```

---

## 4. Service Startup Configuration

### 4.1 LMCache Config File
Create `lmcache-config-ddr.yaml`:

```yaml
chunk_size: 1024
local_cpu: true
max_local_cpu_size: 1

extra_config:
    save_only_first_rank: true
    first_rank_max_local_cpu_size: 150
    broadcast_shard_size: 16
```

### 4.2 Startup Scripts

The Base (HBM) and DDR scripts are identical except the DDR version adds `LMCACHE_CONFIG_FILE` and `--kv-transfer-config`.

#### Base: HBM (Native HBM Prefix Cache)
```bash
#!/bin/sh
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

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

#### DDR: System Memory Cache Only
> Requires `/mnt/sdb/<USER_ID>/ds-workspace/lmcache-config-ddr.yaml`.
```bash
#!/bin/sh
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# DDR config path
export LMCACHE_CONFIG_FILE="/mnt/sdb/<USER_ID>/lmcache-config-ddr.yaml"

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
    "multistream_overlap_shared_expert":true}' \
    --kv-transfer-config '{
        "kv_connector": "LMCacheAscendConnectorV1Dynamic",
        "kv_role": "kv_both",
        "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    }' > ds_lmcache_ddr.log 2>&1
```

---

## 5. Benchmark Testing

### 5.1 Multi-Round Conversation Bench
Evaluates external cache retrieval and hit rate in multi-round long conversations.

```bash
python3 /LMCache/benchmarks/multi_round_qa/multi-round-qa.py \
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

### 5.2 Prefix Repetition Bench
Evaluates throughput and latency under high concurrency with repeated prefixes.

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

> **To reveal LMCache advantages**:
> 1. **Increase concurrency**: Raise `--num-users` (e.g., 128/256) or `--max-concurrency` until HBM fills up and eviction occurs.
> 2. **Disable native prefix cache**: Add `--no-enable-prefix-caching` to test LMCache IO efficiency in isolation.
