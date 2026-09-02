# GLM-5.2 (w8a8) Deployment, Performance, and Accuracy Handbook

> **About this document**: an end-to-end working handbook covering
> deployment, service startup, performance benchmarking, and accuracy
> evaluation. All commands are copy-paste ready; all performance and
> accuracy figures come from measured runs.
>
> **Hardware**: Ubuntu server / 8 × Ascend 910 NPUs (2 chips per card,
> 16 chips)
> **Model**: GLM-5.2-w8a8 (MoE `GlmMoeDsaForCausalLM`, MLA attention,
> INT8 weights / INT8 activations)
> **Inference stack**: vLLM-Ascend image `glm5.2-a3` + LMCache-Ascend
> v0.4.3
>
> `<USER_ID>` is the placeholder for your working directory and container
> name (e.g. `zj`).

## 1. Environment Preparation

Single Ubuntu server: NPUs 0-7, 2 chips per card = 16 davinci devices
(`/dev/davinci0` - `davinci15`). The service maps `DP=2 x TP=8` onto the
16 chips, each replica spanning 8 chips with TP and serving half the
traffic.

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

### 4.4 Verify the Service

```bash
curl http://localhost:8077/health
curl http://localhost:8077/v1/models
curl http://localhost:8077/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-52","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

---

## 5. Performance Benchmark

### 5.1 Experiment Design

Two service configurations are compared under an identical workload; the
only variable is whether the LMCache connector is attached:

| Run | Service configuration | LMCache YAML |
|---|---|---|
| **test1** | APC on, **no** `--kv-transfer-config` | n/a (Section 4.3) |
| **test2** | APC on, **with** `--kv-transfer-config` | Section 4.1 config |

### 5.2 Benchmark Command

The workload — 10 users sharing a 1000-token system prompt, 8 rounds
each with ~8000 tokens of accumulated history — maximizes prefix reuse:

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

### 5.3 Summary Results

| Metric | test1 (no connector) | test2 (with connector) | Verdict |
|---|:---:|:---:|:---:|
| Completed requests | 411 | 411 | On par |
| Completion rate during injection | 99% | 99% | On par |
| Wall time | 1204 s | 1206 s | On par |
| **Throughput (output tok/s)** | 51.22 | 51.11 | On par |
| **Mean TTFT** | 3.54 s | **2.44 s** | **test2 -31%** |
| Median TTFT | 3.94 s | **1.52 s** | **test2 -61%** |
| TTFT P90 | 6.30 s | 4.94 s | test2 -22% |
| TTFT P99 | 7.61 s | 9.68 s | test2 slightly higher (first-round cold start) |
| **Mean generation time** | 13.52 s | **7.47 s** | **test2 -45%** |
| **Decode rate (wall clock)** | 11.10 tok/s | **20.08 tok/s** | **test2 +81%** |
| First-round TTFT | 5.34 s | 5.60 s | On par (no history to reuse) |
| Later-round TTFT | 3.22 s | **1.89 s** | test2 -41% |
| Output completeness (=150) | 100% | 100% | On par |

Completed requests per round (identical in both runs):
61, 58, 56, 53, 50, 47, 44, 42.

**Mean TTFT per round** (the multi-round reuse effect):

| Round | test1 | test2 | Improvement |
|---|:---:|:---:|:---:|
| 1 | 5.34 | 5.60 | ~On par (no history) |
| 2 | 3.10 | 2.94 | -5% |
| 3 | 3.00 | 2.05 | -32% |
| 4 | 3.46 | 1.40 | -60% |
| 5 | 2.80 | 1.36 | -51% |
| 6 | 3.67 | 2.45 | -33% |
| 7 | 2.96 | 1.50 | -49% |
| 8 | 3.64 | 1.25 | -66% |

The later the round, the larger the test2 advantage — reuse compounds as
history accumulates, reaching -66% in round 8.

### 5.4 Key Metrics

> Metrics were collected with
> `curl http://127.0.0.1:8077/metrics > metrics.log`.

**Prompt token source breakdown** (the most reliable reuse statistic;
APC hit rate was ~48% vs ~45%, and LMCache served 1,026,048 tokens in
test2):

| Source | test1 engine0 | test2 engine0 | test2 engine1 | Meaning |
|---|:---:|:---:|:---:|---|
| local_compute (actually computed) | 50.5% | **31.5%** | **26.2%** | Tokens actually prefilled |
| local_cache_hit (APC reuse) | 49.5% | 46.9% | 43.5% | Native APC hits |
| external_kv_transfer (LMCache reuse) | 0% | 21.7% | 30.2% | LMCache hits |
| **Total reuse** | **49.5%** | **68.5%** | **73.8%** | — |

In test2 only 26-31% of tokens still need computing (versus ~50% in
test1) — more than half of the prefill work is eliminated, while APC
holds its own 44-47% share.

**Latency breakdown** (averaged over both engines):

| Stage | test1 | test2 |
|---|:---:|:---:|
| **TTFT** | 3.47 s | **2.64 s** |
| — queue | 0.03 s (1%) | 0.04 s (1%) |
| — prefill | 2.99 s | **2.29 s** (0.7 s saved) |
| **Decode** | 13.26 s | **7.45 s** |
| **End-to-end** | 16.74 s | **10.09 s** |
| **Inter-token latency** | 0.2436 s/tok (4.11 tok/s) | **0.1346 s/tok (7.43 tok/s)** |

**Speculative decoding** behaved identically (acceptance 57.74% vs
56.29%, ~1.7 tokens accepted per step) and is not the source of the
difference. Neither run had a single preemption; KV cache usage returned
to zero at the end of both runs.

**LMCache store/retrieve timings** (the decisive difference versus the
old chunk-256 / synchronous-store configuration):

| Metric | test2 (tuned config) | Old config |
|---|:---:|:---:|
| `time_to_lookup` | 0.0002 s/call | 0.0002 s/call |
| **`time_to_retrieve`** | **0.0185 s/call** | **3.01 s/call (162x slower)** |
| `retrieve_to_gpu_time` | 0.0180 s/call | 0.0555 s/call |
| `time_to_store` | 0.0259 s/call | 0.0086 s/call |
| Slow-retrieve count | **0** | — |
| CPU evictions | 172 (asynchronous, no impact) | 0 |

### 5.5 Conclusions

LMCache with the tuned configuration is a net win over the baseline:
TTFT -31%, generation time -45%, decode rate +81%, throughput on par,
total prompt reuse 71% (baseline 49%), zero preemptions and a 99%
completion rate in both runs. The advantage grows with conversation
length (-66% TTFT by round 8).

---

## 6. Accuracy Evaluation

### 6.1 Tooling

AISBench (OpenCompass family) in service (API) mode, on MMLU 5-shot chat
prompts. Two STEM subsets for quick validation —
`college_computer_science` (100 questions) and `high_school_physics`
(151 questions) — in `gen` mode with `first_option_postprocess`
(first valid A/B/C/D) and `temperature=0.01`.

### 6.2 Service Configuration

Same as Section 4.2 plus `--no-enable-prefix-caching`, so every
inference is independent and reproducible. This disables only vLLM's
built-in APC — the LMCache connector is unaffected:

```bash
vllm serve /mnt/sdb/models/GLM-5.2-w8a8 \
  ... (as in Section 4.2) \
  --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}' \
  --no-enable-prefix-caching      # accuracy runs only
```

### 6.3 Install AISBench and Download MMLU

```bash
cd /mnt/sdb/<USER_ID>
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517          # core package
pip3 install -r requirements/api.txt      # service-mode model support
pip3 install -r requirements/extra.txt    # extra dependencies
ais_bench -h                              # verify the installation

cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip
unzip mmlu.zip && rm mmlu.zip
cd ../../
```

### 6.4 Model Configuration

File: `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",                       # evaluate a served model
        type=VLLMCustomAPIChat,               # vLLM/OpenAI-compatible API client
        abbr="vllm-api-glm52",                # result table column name
        path="",                              # empty in service mode
        model="glm-52",                       # must match served-model-name
        stream=False,                         # evaluation needs full outputs
        request_rate=0,                       # 0 = no throttling
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",                  # change for remote hosts
        host_port=8077,                       # must match the service port
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

### 6.5 Dataset Configuration

File: `benchmark/ais_bench/benchmark/configs/datasets/mmlu/mmlu_gen_5_shot_chat_prompt.py`

Only `college_computer_science` and `high_school_physics` are enabled
(the other 55 subsets are commented out). Each question is 5-shot, with
`first_option_postprocess` extracting the first valid A/B/C/D.

### 6.6 Run

```bash
ais_bench \
  --models vllm_api_general_chat \
  --datasets mmlu_gen_5_shot_chat_prompt \
  --mode all \
  --dump-eval-details \
  --merge-ds
```

### 6.7 Results

The evaluation was repeated twice under the same service configuration:

| Subset | Run 1 | Run 2 | Mean |
|---|:---:|:---:|:---:|
| `college_computer_science` (100 q) | **80.00** | **80.00** | 80.00 |
| `high_school_physics` (151 q) | **82.12** | **81.46** | 81.79 |
| **Mean over both subsets** | **81.06** | **80.73** | **80.90** |

Correct-answer counts: CS 80/100 in both runs (zero fluctuation);
Physics 124 vs 123 of 151 (one question).

### 6.8 Analysis

CS scored 80.00% and Physics 81.79% (two-run means) — solid STEM
performance. CS was identical across runs and Physics moved by a single
question, well within binomial noise (~4% standard deviation for
100-question subsets); the reproducibility comes from
`PYTHONHASHSEED=0` + `temperature=0.01`. The w8a8 quantization typically
costs 1-2 points on MMLU versus FP16 (no FP16 baseline was run here).
Only 2 of 57 subsets were evaluated — extend to the full set, use
`temperature=0`, and add an FP16 baseline for a complete picture.

---

## 7. Tips

> **Operational notes from validation:**
> 1. **`PYTHONHASHSEED=0` is mandatory** — LMCache indexes KV by the
>    hash of the token sequence, and Python's hash randomization makes
>    the same tokens hash differently in each of the 16 worker
>    processes. Measured: without it, 2.43 M lookups and 0 hits; with
>    it, 69% token-level and 98% request-level hit rates.
> 3. **`store_async: true` + `chunk_size: 512`** complete the tuned
>    configuration — background writes plus larger chunks keep both
>    store and retrieve off the critical path.
> 4. **Accuracy runs** — add `--no-enable-prefix-caching` and keep
>    `temperature` at or near 0 for reproducibility; native APC and the
>    LMCache connector are independent mechanisms.

---

## Appendix

### A. Version Matrix

| Component | Version |
|---|---|
| Image | `quay.io/ascend/vllm-ascend:glm5.2-a3` |
| LMCache (upstream) | 0.4.3 |
| LMCache-Ascend | v0.4.3 |
| Declared vLLM version | 0.21.0 (via `VLLM_VERSION`) |

### B. File Index

| File | Contents |
|---|---|
| `01-deployment-guide.md` | Deployment reproduction guide (source of Sections 1-4) |
| `inputs/performance benchmark/new/` | Full test1 artifact set (no connector) |
| `inputs/performance benchmark/new/new yaml/` | Full test2 artifact set (tuned LMCache) |
| `inputs/AISBench Accuracy/` | Accuracy test results and configuration |
| `10-lmcache-optimized-final-conclusion.md` | Detailed performance conclusions |
| `04-accuracy-report.md` | Detailed accuracy analysis |

> All performance and accuracy data come from measured runs; the
> performance figures can be reproduced with
> `new/new yaml/compare.py` in one step.
