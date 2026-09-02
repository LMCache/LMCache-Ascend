# Qwen3.6-35B-A3B Single-Node Deployment Guide (LMCache, DDR + SSD)

Deploy Qwen3.6-35B-A3B as a single vLLM instance on one Ascend server
with the HMA-adapted LMCache builds, comparing the native HBM
prefix-caching baseline against LMCache with a DDR backend and a two-tier
DDR + SSD backend.

> **Hardware validation scope**: verified on a single Ascend server; the
> serving scripts use 8 NPUs (`--tensor-parallel-size 8`).

## 1. Environment Preparation

```bash
mkdir -p /mnt/sdb/<USER_ID>                    # workspace
mkdir -p /mnt/sdb/<USER_ID>/lmcache_ssd_dir    # disk KV cache (SSD backend)
```

Replace `<USER_ID>` with your identifier. Both directories are mounted
into the container and hold the source trees, configuration files,
benchmarks, and disk cache files. Confirm the model paths mounted from
the host (e.g. `/mnt/sdb/models`) exist before starting the container.

---

## 2. Docker Container Startup

```bash
#!/bin/bash
# NPU devices exposed to the container (all 16)
export DEVICE_LIST="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

docker run -itd \
  --name lmcache-qwen36-<USER_ID> \
  --privileged \
  --net=host \
  --shm-size 128g \
  --restart unless-stopped \
  --cap-add=SYS_RESOURCE \
  --cap-add=IPC_LOCK \
  -w /mnt/sdb/<USER_ID>/qwen36-workspace \
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
cd /mnt/sdb/<USER_ID>/qwen36-workspace
git clone -b adapt_hma_4.3 https://github.com/Syan0o0/LMCache-HMA.git
cd LMCache-HMA
export NO_CUDA_EXT=1
# add -i https://pypi.tuna.tsinghua.edu.cn/simple if PyPI is slow
python3 -m pip install -v --no-build-isolation -e .
cd ..
```

### 3.2 Install LMCache-Ascend

```bash
cd /mnt/sdb/<USER_ID>/qwen36-workspace
git clone --recurse-submodules -b adapt_hma_on_main_for_lmcache_4.3 https://github.com/Syan0o0/LMCache-Ascend-HMA.git
cd LMCache-Ascend-HMA
python3 -m pip install -v --no-build-isolation -e .
cd ..
```

### 3.3 Patch vLLM (scheduler assertion)

Comment out the assertion in `vllm/vllm/v1/core/sched/scheduler.py`:

```python
def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        # assert num_external_computed_tokens == 0, (
        #     "External KV connector is not verified yet"
        # )
```

### 3.4 Patch vLLM-Ascend (Mamba preprocessing order)

Move Mamba pre-processing after external KV loading. Apply under
`/vllm-workspace/vllm-ascend`:

```diff
diff --git a/vllm_ascend/worker/model_runner_v1.py b/vllm_ascend/worker/model_runner_v1.py
index 5df036d..73a8990 100644
--- a/vllm_ascend/worker/model_runner_v1.py
+++ b/vllm_ascend/worker/model_runner_v1.py
@@ -1231,23 +1231,6 @@ class NPUModelRunner(GPUModelRunner):
                 pad_attn = cudagraph_mode == CUDAGraphMode.FULL
-                # NOTE(Angazenn): According to https://github.com/vllm-project/vllm/pull/30877,
-                # there should be a corresponding 'postprocess_mamba'. However, it is called inside
-                # '_update_states_after_model_execute', which is not overridden in vLLM-Ascend.
-                # We simply utilize the implementation in vLLM.
-                if self.cache_config.mamba_cache_mode == "align":
-                    mamba_utils.preprocess_mamba(
-                        scheduler_output,
-                        self.kv_cache_config,
-                        self.cache_config,
-                        self.mamba_state_idx,
-                        self.input_batch,
-                        self.requests,
-                        self.compilation_config.static_forward_context,
-                        self.model.get_mamba_state_copy_func(),
-                        self._get_mamba_copy_bufs(),
-                    )
-
                 use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                 ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices
@@ -1356,6 +1339,54 @@ class NPUModelRunner(GPUModelRunner):
                 ),
             ) as kv_connector_output,
         ):
+            # Run preprocess_mamba after external KV load so LMCache-restored
+            # Mamba/GDN states can be copied into the active running-state
+            # window before model forward.
+            if self.cache_config.mamba_cache_mode == "align":
+                mamba_utils.preprocess_mamba(
+                    scheduler_output,
+                    self.kv_cache_config,
+                    self.cache_config,
+                    self.mamba_state_idx,
+                    self.input_batch,
+                    self.requests,
+                    self.compilation_config.static_forward_context,
+                    self.model.get_mamba_state_copy_func(),
+                    self._get_mamba_copy_bufs(),
+                )
+            if (
+                self.cache_config.mamba_cache_mode == "align"
+                and use_spec_decode
+                and self._has_gdn
+            ):
+                rebuilt_attn_metadata, rebuilt_spec_decode_common_attn_metadata = (
+                    self._build_attention_metadata(
+                        num_tokens=num_tokens_unpadded
+                        if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
+                        else total_num_scheduled_tokens,
+                        num_tokens_padded=num_tokens_padded,
+                        num_reqs=num_reqs,
+                        num_reqs_padded=num_reqs_padded,
+                        max_query_len=max_num_scheduled_tokens,
+                        ubatch_slices=ubatch_slices_attn,
+                        logits_indices=logits_indices,
+                        use_spec_decode=use_spec_decode,
+                        num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
+                        num_scheduled_tokens_np=num_scheduled_tokens_np,
+                        cascade_attn_prefix_lens=cascade_attn_prefix_lens,
+                    )
+                )
+                if (
+                    isinstance(attn_metadata, dict)
+                    and isinstance(rebuilt_attn_metadata, dict)
+                ):
+                    attn_metadata.clear()
+                    attn_metadata.update(rebuilt_attn_metadata)
+                else:
+                    attn_metadata = rebuilt_attn_metadata
+                spec_decode_common_attn_metadata = (
+                    rebuilt_spec_decode_common_attn_metadata
+                )
             hidden_states = self._model_forward(
                 num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
             )
```

---

## 4. Service Startup Configuration

### 4.1 LMCache Config Files

`lmcache-config-ssd.yaml` (DDR + SSD, both tiers):

```yaml
chunk_size: 1024
local_cpu: true
max_local_cpu_size: 120           # max CPU (DDR) cache size in GB
use_gpu_connector_v3: true

internal_api_server_enabled: true
internal_api_server_host: "0.0.0.0"
internal_api_server_port_start: 6999

local_disk: "file:///mnt/sdb/<USER_ID>/lmcache_ssd_dir/qwen36-test"
max_local_disk_size: 200          # max disk cache size in GB
```

`lmcache-config-ddr.yaml`: identical with the last two `local_disk` lines
removed.

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

vllm serve /mnt/sdb/models/Qwen3.6-35B-A3B \
  --served-model-name qwen3.6 \
  --host 0.0.0.0 \
  --port 8055 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-num-seqs 128 \
  --max-model-len 262144 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}' \
  --async-scheduling > qwen36_base.log 2>&1
```

### 4.3 DDR Variant

Identical to the baseline script except for:

```bash
export LMCACHE_CONFIG_FILE="/mnt/sdb/<USER_ID>/qwen36-workspace/lmcache-config-ddr.yaml"
```

```bash
vllm serve /mnt/sdb/models/Qwen3.6-35B-A3B \
  ...                                        # identical to the baseline script
  --async-scheduling \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{
      "kv_connector": "LMCacheAscendConnectorV1Dynamic",
      "kv_role": "kv_both",
      "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
  }' > qwen36_lmcache_ddr.log 2>&1          # distinct log name
```

### 4.4 DDR + SSD Variant

Identical to the DDR variant except for:

```bash
export LMCACHE_CONFIG_FILE="/mnt/sdb/<USER_ID>/qwen36-workspace/lmcache-config-ssd.yaml"
...
  }' > qwen36_lmcache_ssd.log 2>&1          # distinct log name
```

---

## 5. Verification and Benchmark

### 5.1 Smoke Test

```bash
curl http://127.0.0.1:8055/v1/models
```

### 5.2 Multi-Round Conversation Bench

Evaluates incremental retrieval and hit behavior of the external cache in
long, multi-round conversations:

```bash
python3 /mnt/sdb/<USER_ID>/qwen36-workspace/LMCache-HMA/benchmarks/multi_round_qa/multi-round-qa.py \
    --num-users 64 \
    --num-rounds 10 \
    --qps 0.8 \
    --shared-system-prompt 10000 \
    --user-history-prompt 30000 \
    --answer-len 300 \
    --model qwen3.6 \
    --base-url http://localhost:8055/v1 \
    --enforce-strict-concurrent-users \
    --time 1200
```

### 5.3 Prefix-Repetition Bench

Evaluates throughput and latency under high concurrency with many shared
prefixes:

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8055 \
  --model /mnt/sdb/models/Qwen3.6-35B-A3B \
  --served-model-name qwen3.6 \
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
> 2. **HBM headroom is large for this model** — under TP8 the per-card
>    weight footprint is only ~8.56 GB HBM, so the native block manager
>    already holds a very large KV cache on its own; use the two notes
>    below to expose LMCache's value.
> 3. **Push concurrency until HBM evicts** — raise `--num-users` (e.g.
>    128/256) or `--max-concurrency` until the native HBM KV cache fills
>    and cold eviction starts; the LMCache memory/disk swap-in advantage
>    only becomes visible then.
> 4. **Isolate LMCache I/O** — add `--no-enable-prefix-caching` to
>    disable vLLM's internal HBM prefix matching and route all prefix
>    storage/retrieval through the LMCache connector.
