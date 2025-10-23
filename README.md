<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="lmcache logo">
  </p>
  <h3 align="center">
  LMCache-Ascend Plugin
  </h3>

  <p align="center">
  | <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://blog.lmcache.ai/"><b> LMCache Blog</b></a> 
| <a href="https://docs.lmcache.ai/"><b>Documentation</b></a> | <a href="https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q"><b> Slack</b></a>
  </p>
</div>

--------------------------------------------------------------------------------

## Overview

LMCache-Ascend is a community maintained plugin for running LMCache on the Ascend NPU.


## Prerequisites

To use LMCache-Ascend on the NPU hardware, please make sure the following prerequisites are satisfied.

- Hardware: Atlas 800I A2 Inference series. The rest of the series like A3 Inference/Training and 300I Duo are experimental.
- OS: Linux-based.
- Software:
  - **Python**: >= 3.10, <= 3.11
  - **CANN Toolkit**: >= 8.2rc1
  - **Ascend Driver**: >= 24.1
  - **PyTorch**: == 2.5.1, **Torch-npu**: == 2.5.1.post1.dev20250619
  - **vLLM**: v0.9.2 & **vLLM-Ascend**: v0.9.2rc1

## Getting Started

### Clone LMCache-Ascend Repo

Our repo contains a kvcache ops submodule for ease of maintainence, therefore we recommend cloning the repo with submodules.

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
```

### Docker

```bash
cd /workspace/LMCache-Ascend
docker build -f docker/Dockerfile.a2.openEuler -t lmcache-ascend:v0.3.3-vllm-ascend-v0.9.2rc1-910b-cann-8.2rc1-py3.11-openeuler-22.03 .
```

Once that is built, run it with the following cmd
```bash
DEVICE_LIST="0,1,2,3,4,5,6,7"
docker run -it \
    --privileged \
    --cap-add=SYS_RESOURCE \
    --cap-add=IPC_LOCK \
    -p 8000:8000 \
    -p 8001:8001 \
    --name lmcache-ascend-dev \
    -e ASCEND_VISIBLE_DEVICES=${DEVICE_LIST} \
    -e ASCEND_RT_VISIBLE_DEVICES=${DEVICE_LIST} \
    -e ASCEND_TOTAL_MEMORY_GB=32 \
    -e VLLM_TARGET_DEVICE=npu \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/localtime:/etc/localtime \
    -v /var/log/npu:/var/log/npu \
    -v /dev/davinci_manager:/dev/davinci_manager \
    -v /dev/devmm_svm:/dev/devmm_svm \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    lmcache-ascend:v0.3.3-vllm-ascend-v0.9.2rc1-910b-cann-8.2rc1-py3.11-openeuler-22.03 \
    /bin/bash
```

For further info about deployment notes, please refer to the [guide about deployment](docs/deployment.md)

### Manual Installation

Assuming your working directory is ```/workspace```.

1. Clone and Install vLLM Repo
```bash
VLLM_REPO=https://github.com/vllm-project/vllm.git
VLLM_TAG=v0.9.2
git clone --depth 1 $VLLM_REPO --branch $VLLM_TAG /workspace/vllm
# NOTE: There is an Ascend Triton but we don't currently support it properly.
VLLM_TARGET_DEVICE="empty" python3 -m pip install -e /workspace/vllm/ --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip uninstall -y triton
```

2. Clone and Install vLLM Ascend Repo
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

VLLM_ASCEND_REPO=https://github.com/vllm-project/vllm-ascend.git
VLLM_ASCEND_TAG=v0.9.2rc1
git clone --depth 1 $VLLM_ASCEND_REPO --branch $VLLM_ASCEND_TAG /workspace/vllm-ascend
# apply patch to v0.9.2rc1
cd /workspace/vllm-ascend && \
    git apply -p1 /workspace/LMCache-Ascend/docker/kv-connector-v1.diff

export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib && \
python3 -m pip install -v -e /workspace/vllm-ascend/ --extra-index https://download.pytorch.org/whl/cpu/
```

3. Clone and Install LMCache Repo

```bash
LMCACHE_REPO=https://github.com/LMCache/LMCache.git
LMCACHE_TAG=v0.3.3
git clone --depth 1 $LMCACHE_REPO --branch $LMCACHE_TAG /workspace/LMCache
# our build is based on arm64
sed -i "s/^infinistore$/infinistore; platform_machine == 'x86_64'/" /workspace/LMCache/requirements/common.txt
export NO_CUDA_EXT=1 && python3 -m pip install -v -e /workspace/LMCache
```

4. Install LMCache-Ascend Repo

```bash
cd /workspace/LMCache-Ascend
python3 -m pip install -v --no-build-isolation -e .
```

### Usage

We introduce a dynamic KVConnector via LMCacheAscendConnectorV1Dynamic, therefore LMCache-Ascend Connector can be used via the kv transfer config in the two following setting.

#### Online serving
```bash
python \
    -m vllm.entrypoints.openai.api_server \
    --port 8100 \
    --model /data/models/Qwen/Qwen3-32B \
    --trust-remote-code \
    --disable-log-requests \
    --block-size 128 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}'
```

#### Offline
```python
ktc = KVTransferConfig(
        kv_connector="LMCacheAscendConnectorV1Dynamic",
        kv_role="kv_both",
        kv_connector_module_path="lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    )
```

## FAQ

1. Why do I have HostRegisterError ? 
  - If you encounter the Host Register Error within a container environment, please make sure you add the IPC_LOCK capabilities.
  - Otherwise, please check your driver version is >= 24.0
