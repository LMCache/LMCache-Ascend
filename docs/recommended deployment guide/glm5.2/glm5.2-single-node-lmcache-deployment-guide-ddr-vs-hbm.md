# GLM-5.2 (w8a8) Single-Node Deployment Guide — DDR vs HBM

Deploy GLM-5.2-w8a8 as a single vLLM service on one Ascend server with
LMCache-Ascend providing a CPU-memory (DDR) KV cache tier, and benchmark
it against the native HBM prefix-caching baseline.

> **Hardware validation scope**: verified on a single Ubuntu server with
> 8 × Ascend 910 NPUs (2 chips per card, 16 chips); the service maps
> `--data-parallel-size 2 --tensor-parallel-size 8` onto the 16 chips,
> each DP replica spanning 8 chips.

## 1. Environment Preparation

Create a workspace on the host. Replace `<USER_ID>` with your identifier:

```bash
mkdir -p /mnt/sdb/<USER_ID>
```

The directory is mounted into the container at the same path and holds
the source trees, configuration files, and benchmarks. Confirm the
GLM-5.2-w8a8 weights exist under `/mnt/sdb/models/GLM-5.2-w8a8`.

---

## 2. Docker Container Startup

The image ships CANN, torch_npu, vllm-ascend, and the matching vLLM
pre-installed; the `glm5.2-a3` tag carries the GLM-5.2 adaptation:

```bash
docker run -itd \
  --shm-size=200g --privileged --net=host \
  --device=/dev/davinci0  --device=/dev/davinci1  --device=/dev/davinci2  --device=/dev/davinci3 \
  --device=/dev/davinci4  --device=/dev/davinci5  --device=/dev/davinci6  --device=/dev/davinci7 \
  --device=/dev/davinci8  --device=/dev/davinci9  --device=/dev/davinci10 --device=/dev/davinci11 \
  --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /var/log/npu:/var/log/npu \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v /usr/src/kernels:/usr/src/kernels:ro \
  -v /lib/modules:/lib/modules:ro \
  -v /mnt/shared/models:/mnt/shared/models \
  -v /mnt/sdb/models:/mnt/sdb/models \
  -v /mnt/sdb/<USER_ID>:/mnt/sdb/<USER_ID> \
  --name vllm-ascend-<USER_ID> \
  --entrypoint /bin/bash \
  quay.io/ascend/vllm-ascend:glm5.2-a3
```

---

## 3. Install from Source (Inside the Container)

```bash
docker exec -it -u root vllm-ascend-<USER_ID> bash
npu-smi info   # expect 8 cards / 16 chips
```

### 3.1 Install LMCache

```bash
# mirror: add -i https://mirrors.aliyun.com/pypi/simple if PyPI is slow
NO_CUDA_EXT=1 pip install lmcache==0.4.3
```

`NO_CUDA_EXT=1` skips the CUDA extension (Ascend hosts have no CUDA
toolchain).

### 3.2 Install LMCache-Ascend

```bash
git clone --recurse-submodules -b v0.4.3 https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
pip install -v --no-build-isolation -e .
```

> **Important — relax the CANN version checks** in
> `third_party/kvcache-ops/ascendc_with_def.cmake` before building:
> change `VERSION_EQUAL "8.3"` to `VERSION_GREATER_EQUAL "8.3"` and
> `VERSION_EQUAL "8.5"` to `VERSION_GREATER_EQUAL "8.5"`, otherwise the
> build fails against the current CANN version.

### 3.3 Clone the LMCache Benchmark Suite

```bash
cd /mnt/sdb/<USER_ID>
git clone -b v0.4.3 https://github.com/LMCache/LMCache.git
```

> Benchmarks run from `LMCache/` (upstream); deployment uses
> `LMCache-Ascend/` (the plugin). Do not mix up the two directories.

---

## 4. Service Startup Configuration

### 4.1 LMCache Config File

`/mnt/sdb/<USER_ID>/LMCache-Ascend/lmcache_config_file.yaml`:

```yaml
chunk_size: 512          # tokens per KV chunk
local_cpu: True          # KV in CPU memory, saves NPU HBM
max_local_cpu_size: 50   # CPU cache cap (GB), LRU eviction beyond
use_layerwise: False
enable_async_loading: False   # covered by store_async
store_async: True        # background writes, never blocks the engine
extra_config:
  save_only_first_rank: true
  lookup_backoff_time: 0.001
  first_rank_max_local_cpu_size: 150
```

### 4.2 Startup Script (with LMCache)

Run from the `LMCache-Ascend` directory:

```bash
export LMCACHE_CONFIG_FILE=/mnt/sdb/<USER_ID>/LMCache-Ascend/lmcache_config_file.yaml
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_VERSION=0.21.0
export TORCH_COMPILE_DISABLE=1
export PYTHONHASHSEED=0

vllm serve /mnt/sdb/models/GLM-5.2-w8a8 \
  --host 0.0.0.0 \
  --port 8077 \
  --data-parallel-size 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm-52 \
  --max-num-seqs 48 \
  --max-model-len 20480 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --quantization ascend \
  --async-scheduling \
  --additional-config '{"enable_npugraph_ex": true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
  --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}'
```

### 4.3 Baseline Variant (no LMCache)

Identical, but drop the `LMCACHE_CONFIG_FILE` export and the
`--kv-transfer-config` flag — this is the HBM/APC-only baseline used for
comparison.

---

## 5. Verification and Benchmark

### 5.1 Smoke Test

```bash
curl http://localhost:8077/health
curl http://localhost:8077/v1/models
curl http://localhost:8077/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-52","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

### 5.2 Multi-Round Conversation Bench

Run the same load against both variants (Section 4.2 vs 4.3). The
workload — a shared system prompt plus round-over-round accumulated
history — maximizes prefix reuse:

```bash
cd /mnt/sdb/<USER_ID>/LMCache/benchmarks/multi_round_qa

python multi-round-qa.py \
    --num-users 10 \
    --num-rounds 8 \
    --qps 0.3 \
    --shared-system-prompt 1000 \
    --user-history-prompt 8000 \
    --answer-len 150 \
    --model glm-52 \
    --time 1200 \
    --base-url http://localhost:8077/v1
```

---

## 6. Tips

> **Operational notes from validation:**
> 1. **`PYTHONHASHSEED=0` is mandatory** — without it, per-process hash
>    randomization makes the same tokens hash differently across the 16
>    worker processes and the LMCache hit rate stays at zero.
> 3. **Measure the baseline without LMCache** (Section 4.3); everything
>    else must stay identical between the two runs.
