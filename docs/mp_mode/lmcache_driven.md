# LMCache-Driven Mode Deployment Guide

In `lmcache_driven` mode the LMCache server pulls and pushes KV cache through device IPC handles: the engine submits a lookup/retrieve request, and the server performs the data movement on the engine's behalf. This mode supports standard GQA/MHA layouts as well as vLLM **MLA/DSA multi-plane** packed KV (`(latent, scale)` layouts) and is the required mode for DeepSeek-V4.

> **Not supported — Mamba / GDN hybrids (for example Qwen3.5-27B).** MP mode does not support models with vLLM `MambaSpec` state that must be snapshotted for reuse. Those models fail LMCache KV group validation at startup. Use the in-process Ascend connector for single-process serving, or see [README.md](README.md#supported-models-and-limitations).

> **Important — `--chunk-size 4096` is a DeepSeek-V4 requirement, not a general one.** DeepSeek-V4's packed MLA/DSA KV layout is designed for 4096-token chunks, and the connector expects this size for that model family. Other models with a single KV-cache format (GQA/MHA, e.g. Qwen) do not need 4096 and can use a different chunk size (for example 128, which matches the vLLM block size). Do not apply the 4096 value to other models, and do not run DeepSeek-V4 with any other value.

## 1. Environment Preparation

> **Hardware validation scope**: This guide has been verified on Ascend 910B (A2 / A3).

For base environment preparation (NPU driver, CANN toolkit, container images), follow the official [vLLM-Ascend tutorials](https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/models/index.html).

Create the working directory on the host:

```bash
mkdir -p /workspace
```

> This directory is mounted into the container for storing source code, configs, and logs.

---

## 2. Docker Container Startup

Launch the Ascend NPU vLLM container:

```bash
#!/bin/bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0-a3
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
    -v /workspace:/workspace \
    -v /home:/home \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

---

## 3. Install from Source (Inside the Container)

### 3.1 Install LMCache

The LMCache main branch (`dev`) provides the MP transfer architecture (the `lmcache server` CLI and the `LMCacheMPConnector`):

```bash
cd /workspace
git clone https://github.com/LMCache/LMCache.git
cd LMCache
python3 -m pip install -v --no-build-isolation -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..
```

> Do not mix this checkout with the LMCache release used by the in-process mode (e.g. `v0.4.x` installed for `LMCacheAscendConnectorV1Dynamic`): both are installed editable and only one can own the `lmcache` package in a given environment.

### 3.2 Install LMCache-Ascend

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
```

> **Important — `third_party/hcomm` must match the container's CANN version.**

> 1. Check the container's CANN version:
>    ```bash
>    ls /usr/local/Ascend
>    ```
>    e.g. `8.5.0` (or `0.8.5`) means CANN 8.5; `9.0.0` (or `0.9.0`) means CANN 9.0.
> 2. List available hcomm versions/tags from the upstream repository:
>    <https://gitcode.com/cann/hcomm>
>    Then switch the submodule to the tag that matches your CANN version **before** running `pip install`:
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

Install the package:

```bash
pip install -v --no-build-isolation -e .
```

### 3.3 Verify the Installation

```bash
pip show lmcache lmcache-ascend                       # both installed editable
python3 -c "import lmcache_ascend"                    # imports cleanly on the NPU host
lmcache server --help | grep supported-transfer-mode  # shows lmcache_driven / engine_driven / auto
```

---

## 4. Service Startup

### 4.1 Start the LMCache Server

```bash
lmcache server \
    --host 127.0.0.1 \
    --port 5555 \
    --chunk-size 4096 \
    --supported-transfer-mode lmcache_driven \
    --l1-size-gb 50 \
    --eviction-policy LRU \
    --disable-metrics &
```

Wait for readiness before starting vLLM:

```
LMCache INFO: LMCache zmq cache server is running on 127.0.0.1:5555
INFO:     Uvicorn running on http://127.0.0.1:8080
```

Notes:

- The server binds the ZMQ endpoint on `--port` and an HTTP health/ops endpoint on port `8080` on the same host; make sure both are free.
- Size `--l1-size-gb` so that the whole working set (number of distinct prefixes × prefix tokens × KV bytes per token) fits comfortably; if the working set exceeds L1, LRU eviction turns hits back into misses and requests fall back to full prefill.
- The server and the engines share failure semantics: restart the server and the engines together, and never restart the server while requests are in flight.

### 4.2 Start vLLM

Example: DeepSeek-V4-Flash, TP8/EP on 8 NPUs.

```bash
#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

vllm serve /workspace/models/DeepSeek-V4-Flash-0731-w8a8 \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --no-enable-prefix-caching \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --port 8900 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_cpu_binding": false,
    "enable_dsa_cp": false,
    "multistream_overlap_shared_expert": true}' \
      --kv-transfer-config '{
      "kv_connector": "LMCacheMPConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "lmcache.mp.host": "127.0.0.1",
        "lmcache.mp.port": 5555,
        "lmcache.mp.mp_transfer_mode": "lmcache_driven"
      }
    }'
```

A successful connector start shows this line for every worker:

```
LMCache INFO: lmcache.mp.mp_transfer_mode = lmcache_driven (overridden, default: auto)
```

On NPU the `auto` value resolves to `engine_driven`, so DeepSeek-V4 deployments must set `lmcache_driven` explicitly as above.

---

## 5. Verification

Send the same long prompt twice with `temperature: 0`. The first request prefills the prompt and stores the KV cache; the second request hits the external cache:

```bash
curl -s http://localhost:8900/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"dsv4","messages":[{"role":"user","content":"<long prompt>"}],"max_tokens":100,"temperature":0}'
```

Expected evidence:

- The server log shows `Stored ... tokens` for the first request and `Retrieved ... tokens` for the second request.
- The vLLM request log shows `External prefix cache hit rate` growing above zero.
- The two responses are coherent completions; with the quantized KV round-trip the warm continuation may drift by a few tokens from the cold one, which is expected.

---

## 6. Tips

- `--no-enable-prefix-caching` is used in the reference command so that external-cache hits are attributable to LMCache; internal prefix caching can be enabled independently if desired.
- The connector reports lookup/load progress asynchronously: requests waiting for an external load appear as `Deferred` in the vLLM engine log line instead of blocking the engine.
- Do not resize or delete the shared-memory pool while any engine is attached.
- **To reveal LMCache advantages** under load: increase request concurrency, or keep `--no-enable-prefix-caching` to test LMCache IO efficiency in isolation.
