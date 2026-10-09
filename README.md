<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache-Ascend/main/docs/logos/lmcache-ascend-logo.png" width="720" alt="lmcache-ascend logo">
  </p>
  <h3 align="center">
  LMCache-Ascend Plugin
  </h3>

  [![Code Quality](https://github.com/LMCache/LMCache-Ascend/actions/workflows/code-quality.yml/badge.svg?branch=main)](https://github.com/LMCache/LMCache-Ascend/actions/workflows/code-quality.yml)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LMCache/LMCache-Ascend)
  
  <br />

  <p align="center">
  | <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a>
  | <a href="https://blog.lmcache.ai/"><b> LMCache Blog</b></a> 
  | <a href="https://docs.lmcache.ai/"><b>Documentation</b></a> 
  | <a href="https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q"><b> Slack</b></a>
  | <a href="https://deepwiki.com/LMCache/LMCache-Ascend"><b>LMCache-Ascend Wiki</b></a>
  </p>
</div>

--------------------------------------------------------------------------------

## Overview

LMCache-Ascend is a community maintained plugin for running LMCache on the Ascend NPU.

## Prerequisites

To use LMCache-Ascend on the NPU hardware, please make sure the following prerequisites are satisfied.

- **Hardware**: Atlas 800I A2 Inference series. (A3 Inference/Training and 300I Duo are experimental).
- **OS**: Linux-based.
- **Software**:
  - **Python**: >= 3.10
  - **CANN Toolkit**: >= 8.5.0
  - **Ascend Driver**: >= 25.5
  - **PyTorch**: >= 2.8.0
  - **vLLM**: >=v0.18.0 & **vLLM-Ascend**: >=v0.18.0
- **Container preparation**: see the official [vLLM-Ascend tutorials](https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/models/index.html) for preparing the base environment (NPU driver, CANN toolkit, and container images).

### Compatibility Matrix

Please ensure your environment matches the versions below.

| LMCache-Ascend | LMCache | vLLM Version |
| :--- | :--- | :--- |
| **main** | **dev** | **>=v0.18.0** |

## Getting Started

The minimal deployment is three steps: install the two packages, start the `lmcache server`, then start vLLM with the connector.

### 1. Install

```bash
git clone https://github.com/LMCache/LMCache.git
cd LMCache
python3 -m pip install -v --no-build-isolation -e .
cd ..

git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
pip install -v --no-build-isolation -e .
```

> `third_party/hcomm` must match the container's CANN version — see [docs/mp_mode/lmcache_driven.md](docs/mp_mode/lmcache_driven.md) for the full installation guide (environment preparation, container startup, source installation).

### 2. Start the LMCache server

```bash
lmcache server \
    --host 127.0.0.1 \
    --port 5555 \
    --chunk-size 128 \
    --supported-transfer-mode lmcache_driven \
    --l1-size-gb 50 \
    --eviction-policy LRU \
    --disable-metrics
```

Wait for `LMCache INFO: LMCache zmq cache server is running on 127.0.0.1:5555`.

`--chunk-size 128` matches the vLLM block size and suits single-KV-format models (e.g. Qwen3); DeepSeek-V4 requires `4096` — see [docs/mp_mode/lmcache_driven.md](docs/mp_mode/lmcache_driven.md) for the full deployment guide.

### 3. Start vLLM with the connector

```bash
vllm serve /path/to/Qwen3-32B \
    --served-model-name qwen3 \
    --max-model-len 40960 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 \
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
        "lmcache.mp.mp_transfer_mode": "lmcache_driven"
      }
    }'
```

A successful start shows `LMCache INFO: lmcache.mp.mp_transfer_mode = lmcache_driven (overridden, default: auto)` for every worker.

For the `engine_driven` transfer mode (workers copy KV from a named SHM pool), see [docs/mp_mode/engine_driven.md](docs/mp_mode/engine_driven.md).

> **Mamba / GDN models (e.g. Qwen3.5-27B) are not supported with `LMCacheMPConnector`.** Use in-process `LMCacheAscendConnectorV1Dynamic` instead. Details: [docs/mp_mode/README.md#supported-models-and-limitations](docs/mp_mode/README.md#supported-models-and-limitations).

## Documentation

- [docs/mp_mode/README.md](docs/mp_mode/README.md) — MP mode overview.
- [docs/mp_mode/lmcache_driven.md](docs/mp_mode/lmcache_driven.md) — `lmcache_driven` full deployment guide (environment preparation, container, installation, server, vLLM, verification).
- [docs/mp_mode/engine_driven.md](docs/mp_mode/engine_driven.md) — `engine_driven` deployment guide (Qwen3 example).
