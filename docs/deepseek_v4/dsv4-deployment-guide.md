# DeepSeek V4 (Flash) Deployment Guide

## 1. Environment Preparation

> **Hardware validation scope**: This guide has been verified only on
> Ascend 910B (A2 / A3).
> **Reference**: [Ascend 910B (A2 / A3) DeepSeek V4 Deployment Guide](https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html)

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
export IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-a3
docker run \
    --name vllm-ascend \
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
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
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
    -v /home:/home \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

---

## 3. Install from Source (Inside the Container)

### 3.1 Install LMCache

Clone under `/mnt/sdb/<USER_ID>` so the benchmark scripts (Section 5) are reachable from the host mount:

```bash
cd /mnt/sdb/<USER_ID>
git clone -b v0.4.5 https://github.com/LMCache/LMCache.git
cd LMCache
export NO_CUDA_EXT=1
python3 -m pip install -v --no-build-isolation -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..
```

### 3.2 Install LMCache-Ascend
```bash
cd /mnt/sdb/<USER_ID>
git clone --recurse-submodules -b dsv4_support_045 https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
pip install -v --no-build-isolation -e .
```

> **Important — `third_party/hcomm` must match the container's CANN version.**

> 1. Check the container's CANN version:
>    ```bash
>    ls /usr/local/Ascend
>    ```
>    e.g. `8.5.0` (or `0.8.5`) means CANN 8.5; `9.0.0` (or `0.9.0`) means
>    CANN 9.0.
> 2. List available hcomm versions/tags from the upstream repository:
>    <https://gitcode.com/cann/hcomm>
>    Then switch the submodule to the tag that matches your CANN version
>    **before** running `pip install`:
>    ```bash
>    cd third_party/hcomm
>    git fetch --tags
>    git checkout v8.5.0   # CANN 8.5.x; use the matching tag for other versions
>    cd ../..
>    ```
> 3. If you switch (or roll back) the submodule, always rebuild:
>    ```bash
>    pip install -v --no-build-isolation -e .
>    ```

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
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
export PYTHONHASHSEED=0

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
    --max-model-len 1048576 \
    --max-num-batched-tokens 10240 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --port 8900 \
    --block-size 32 \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex": true,
        "enable_static_kernel": false
        },
    "enable_cpu_binding": true,
    "enable_flashcomm1": true,
    "multistream_overlap_shared_expert": true}' > ds_base.log 2>&1
```

#### DDR: System Memory Cache Only
> Requires `/mnt/sdb/<USER_ID>/lmcache-config-ddr.yaml` (Section 4.1).
```bash
#!/bin/sh
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
export PYTHONHASHSEED=0

# DDR config path
export LMCACHE_CONFIG_FILE="/mnt/sdb/<USER_ID>/lmcache-config-ddr.yaml"

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
    --max-model-len 1048576 \
    --max-num-batched-tokens 10240 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --port 8900 \
    --block-size 32 \
    --no-disable-hybrid-kv-cache-manager \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex": true,
        "enable_static_kernel": false
        },
    "enable_cpu_binding": true,
    "enable_flashcomm1": true,
    "multistream_overlap_shared_expert": true}' \
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

