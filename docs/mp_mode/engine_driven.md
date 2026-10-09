# Engine-Driven Mode Deployment Guide

In `engine_driven` mode the vLLM workers gather and scatter the KV copies themselves: the server hands over the requested chunks through a named shared-memory pool, and the workers copy them into the paged KV on the device. On NPU, `lmcache.mp.mp_transfer_mode: auto` resolves to this mode. It supports models with a single KV-cache format only (standard GQA/MHA layouts, e.g. Qwen3); multi-plane MLA/DSA models such as DeepSeek-V4 must use [lmcache_driven](lmcache_driven.md). **Mamba / GDN hybrids (e.g. Qwen3.5) are not supported in MP mode** — see [README.md](README.md#supported-models-and-limitations).

> **Important — `--shm-name` and `--no-l1-use-lazy` are mandatory for this mode.** Without a named shared-memory pool the server hands back ordinary (non-pinned) host memory, and every retrieve pays a dynamic pinning and extra copy on the worker, which degrades warm-hit performance severely. Always pair `--shm-name` with `--no-l1-use-lazy` so the pool is allocated eagerly at startup. Note that a shared-memory pool of `--l1-size-gb` size is created under `/dev/shm`, so the container must be started with a matching `--shm-size`.

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

> The shared-memory pool created by `--shm-name` lives under `/dev/shm`; the `--shm-size` above must be at least as large as the server's `--l1-size-gb`.

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
    --chunk-size 128 \
    --supported-transfer-mode engine_driven \
    --l1-size-gb 20 \
    --shm-name lmcache-mp-e2e --no-l1-use-lazy \
    --eviction-policy LRU \
    --disable-metrics &
```

Wait for readiness before starting vLLM:

```
LMCache INFO: Checking if shm capacity is larger than L1 request
LMCache INFO: LMCache zmq cache server is running on 127.0.0.1:5555
INFO:     Uvicorn running on http://127.0.0.1:8080
```

Notes:

- `--chunk-size 128` matches the vLLM block size on Ascend and is a good default for single-KV-format models; unlike DeepSeek-V4 (see [lmcache_driven.md](lmcache_driven.md)), there is no fixed chunk-size requirement here.
- `--l1-size-gb` sizes the shared-memory pool; size it for the expected working set (number of distinct prefixes × prefix tokens × KV bytes per token) so that it fits comfortably.

### 4.2 Start vLLM

Example: Qwen3-32B, TP8 on 8 NPUs.

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

vllm serve /workspace/models/Qwen3-32B \
    --served-model-name qwen3 \
    --max-model-len 40960 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 32 \
    --tensor-parallel-size 8 \
    --no-enable-prefix-caching \
    --port 8900 \
    --kv-transfer-config '{
      "kv_connector": "LMCacheMPConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "lmcache.mp.host": "127.0.0.1",
        "lmcache.mp.port": 5555,
        "lmcache.mp.mp_transfer_mode": "engine_driven"
      }
    }'
```

A successful start shows, for every worker:

```
LMCache INFO: lmcache.mp.mp_transfer_mode = engine_driven (overridden, default: auto)
LMCache INFO: Engine KV Format: EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS
LMCache INFO: Registered non-GPU context for instance ... (world_size=8)
LMCache INFO: Using shm non-GPU transfer strategy
LMCache INFO: Creating EngineDrivenContextShm (shm_name=...)
```

`Creating EngineDrivenContextShm` confirms the workers attached the shared-memory pool; if this line is absent, re-check the server's `--shm-name` / `--no-l1-use-lazy` flags.

> Leave NPU memory headroom for transient transfer buffers inside the worker: keep `--gpu-memory-utilization` at or below `0.8` instead of the usual `0.9`, because the retrieve path allocates temporary device buffers on top of vLLM's profiled budget.

---

## 5. Verification

Send the same long prompt twice with `temperature: 0`:

```bash
curl -s http://localhost:8900/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3","messages":[{"role":"user","content":"<long prompt>"}],"max_tokens":100,"temperature":0}'
```

- The server log shows `Stored ... tokens` for the first request and `Retrieved ... tokens` for the second.
- The vLLM request log shows `External prefix cache hit rate` growing above zero.
- The two responses are identical, which confirms the KV round-trip is correct for this single-format model.
