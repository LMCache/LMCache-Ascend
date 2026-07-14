## 1. 环境准备与工作目录创建

首先在宿主机上创建工作目录。请将 `xx` 替换为您自己的特定标识

```bash
# 创建基础工作工作空间
mkdir -p /mnt/sdb/xx

```

> **注意**：该目录将被挂载到容器内的 `/mnt/sdb/xx`，用于存放 LMCache-Ascend 源码、配置文件、benchmark 仓库。

---

## 2. Docker 容器启动脚本

使用下述命令启动昇腾 NPU 适配版的 vLLM 容器。请根据实际情况确认宿主机的模型路径（如 `/mnt/sdc/models`）是否存在。

```bash
#!/bin/bash
# 定义使用的 NPU 卡号（当前配置为 16 卡全开）
export DEVICE_LIST="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

docker run -itd \
  --name lmcache-ds-xx \
  --privileged \
  --net=host \
  --shm-size 128g \
  --restart unless-stopped \
  --cap-add=SYS_RESOURCE \
  --cap-add=IPC_LOCK \
  -w /mnt/sdb/xx/ds-workspace \
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
  -v /mnt/sdb/xx:/mnt/sdb/xx \
  quay.nju.edu.cn/ascend/vllm-ascend:v0.20.2rc1-openeuler \
  /bin/bash -c "echo \"export PS1='\[\e[1;32m\]\u@\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\\$ '\" >> ~/.bashrc && exec /bin/bash"
```

---

## 3. 源码安装与编译（容器内执行）

进入容器后，依次源码安装 `LMCache` 与 `LMCache-Ascend` 插件。

### 3.1 安装 LMCache
```bash
git clone -b v0.4.5 https://github.com/LMCache/LMCache.git
cd LMCache

export NO_CUDA_EXT=1

python3 -m pip install -v --no-build-isolation -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..
```

### 3.2 安装 LMCache-Ascend
```bash
# 1.clone 指定仓库
git clone --recurse-submodules -b dsv4_support https://github.com/larksudo/LMCache-Ascend.git

# 2. 拷贝算子
#将 LMCache-Ascend/new_kernels/multi_layer_mem_kernels_v2_multi_plane.cpp 复制到 LMCache-Ascend/third_party/kvcache-ops/kernels/multi_layer/ 目录下

# 3. 进入 LMCache-Ascend 根目录，执行安装：
pip install -v --no-build-isolation -e .
```

---


## 4. 服务启动配置与脚本

### 5.1 创建 LMCache 配置文件
在工作目录下创建 `lmcache-config-ddr.yaml`：

```yaml
chunk_size: 1024
local_cpu: true
max_local_cpu_size: 1

extra_config:
    save_only_first_rank: true
    first_rank_max_local_cpu_size: 500
    broadcast_shard_size: 16

```

### 5.2 启动脚本对比

为了对比基线与集成 LMCache（不同后端）后的性能差异，提供以下三份启动脚本。在测试前，请确保在对应路径下创建了各自的配置文件。

#### Base方案：HBM（纯原生显存前缀缓存）
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

#### 对比方案 DDR：（仅开启系统内存缓存）
> **注意**：需提前创建 `/mnt/sdb/xx/ds-workspace/lmcache-config-ddr.yaml`
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

# 1. 注入专属的 DDR 配置文件路径，防物理文件冲突
export LMCACHE_CONFIG_FILE="/mnt/sdb/xx/lmcache-config-ddr.yaml"
# export LMCACHE_LOG_LEVEL=DEBUG

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
    }' > ds_lmcache_ddr.log 2>&1  # 2. 修改日志名称，防止被覆盖
```

---

## 6. Benchmark 基准测试

服务拉起后，可以使用以下两种压测手段对 KV 缓存的效果进行量化评估。

### 测试 1：LMCache 多轮对话原生 Bench
主要评估多轮长对话下外置缓存系统在内存/磁盘层面的增量提取和命中情况。

```bash
python3 /LMCache/benchmarks/multi_round_qa/benmulti-round-qa.py \
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

### 测试 2：vLLM Prefix 前缀重复命中 Bench
主要评估大并发、多重复前缀场景下的吞吐与延迟表现。

```bash
vllm bench serve \
  --backend vllm \
  --base-url [http://127.0.0.1:8900](http://127.0.0.1:8900) \
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
## (Tips)

> **如何显现 LMCache 外置缓存优势**：
> 1. **极限并发压测**：大幅提高测试脚本中的 `--num-users` (如增至 128/256) 或 `--max-concurrency`，直到自带的 HBM 显存写满发生冷换出 (Eviction)，此时 LMCache 磁盘/内存换入的优势便会显现。
> 2. **关闭原生前缀缓存**：在启动脚本中加入 `--no-enable-prefix-caching`，强制关闭 vLLM 内部的显存前缀匹配，使所有前缀缓存的存储与检索压力全部下沉到 LMCache 外置连接器，从而精准测试 LMCache 的 IO 效率。