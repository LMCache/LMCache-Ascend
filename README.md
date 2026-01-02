<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache-Ascend/main/docs/logos/lmcache-ascend-logo.png" width="720" alt="lmcache-ascend logo">
  </p>
  <h3 align="center">
  LMCache-Ascend Plugin
  </h3>

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
  - **Python**: >= 3.10, <= 3.11
  - **CANN Toolkit**: >= 8.2rc1
  - **Ascend Driver**: >= 24.1
  - **PyTorch**: == 2.7.1 (For vLLM 0.10.2+)
  - **vLLM**: v0.10.2 & **vLLM-Ascend**: v0.10.2rc1

### Compatibility Matrix

Please ensure your environment matches the versions below.

#### for PyTorch
| LMCache-Ascend | vLLM Version | PyTorch / Torch-NPU | Status |
| :--- | :--- | :--- | :--- |
| **v0.3.7** | **v0.10.2** | **2.7.1** | ✅ **Verified (Recommended)** |
| **v0.3.7** | **v0.11.x** | **2.7.1** | 🚧 **Experimental** |
| v0.3.3 | v0.9.2 | 2.5.1 | ⚠️ Legacy Support |

#### for MindSpore
| LMCache-Ascend | vLLM Version | MindSpore | Status |
| :--- | :--- | :--- | :--- |
| **v0.3.7** | **v0.9.1** | **2.7.1** | ✅ **Verified (Recommended)** |

> **Note**: If you require legacy support for vLLM 0.9.2, you must use PyTorch 2.5.1. See the [Compatibility Matrix](#compatibility-matrix) above.


## Getting Started

### Clone LMCache-Ascend Repo

Our repo contains a kvcache ops submodule for ease of maintenance, therefore we recommend cloning the repo with submodules.

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
```

### Docker

```bash
cd /workspace/LMCache-Ascend
docker build -f docker/Dockerfile.a2.openEuler -t lmcache-ascend:v0.3.7-vllm-ascend-v0.10.2rc1-openeuler .
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
    lmcache-ascend:v0.3.7-vllm-ascend-v0.10.2rc1-openeuler \
    /bin/bash
```

For further info about deployment notes, please refer to the [guide about deployment](docs/deployment.md)

### Manual Installation

Assuming your working directory is ```/workspace``` and vllm/vllm-ascend have already been installed.

3. Clone and Install LMCache Repo

- from pip
```bash
NO_CUDA_EXT=1 pip install lmcache==0.3.7
```

- from source
```bash
LMCACHE_REPO=https://github.com/LMCache/LMCache.git
LMCACHE_TAG=v0.3.7
git clone --depth 1 $LMCACHE_REPO --branch $LMCACHE_TAG /workspace/LMCache
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

## Getting Started With MindSpore

### Clone LMCache-Ascend Repo

Our repo contains a kvcache ops submodule for ease of maintenance, therefore we recommend cloning the repo with submodules.

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
```

### Manual Installation

1. Start the base container
```bash
docker run -itd \
--shm-size 200g --privileged \
--net=host \
--device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
--device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
--device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /var/log/npu/:/var/log/npu \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /lib/modules:/lib/modules:ro \
-v /usr/src/kernels:/usr/src/kernels:ro \
-v /mnt/storage1/data:/data \
-v /home/:/home \
--name lmcache-ascend-ms \
--entrypoint /bin/bash \
hub.oepkgs.net/oedeploy/openeuler/aarch64/intelligence_boom:0.2.0-aarch64-800I-A2-mindspore2.7-openeuler24.03-lts-sp2

docker exec -it -u root lmcache-ascend-ms bash
```

2. Update CANN version to CANN 8.3.RC1

```bash
bash Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --upgrade
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash Ascend-cann-kernels-910b_8.3.RC1_linux-aarch64.run --install
bash Ascend-cann-nnrt_8.3.RC1_linux-aarch64.run --upgrade
source /usr/local/Ascend/nnrt/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

3. Install LMCache

```bash
NO_CUDA_EXT=1 pip install lmcache==0.3.7 --no-deps
```

4. Install LMCache-Ascend

```bash
cd /workspace/LMCache-Ascend
USE_MINDSPORE=True pip install --no-build-isolation -v -e .
pip install -r requirement_ms.txt
```

### Usage

We introduce a dynamic KVConnector via LMCacheAscendConnectorV1Dynamic, therefore LMCache-Ascend Connector can be used via the kv transfer config in the two following setting.

#### Online serving
```bash
python \
    -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server \
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
2. Why do I have build error related to `cstdint` during manual installation using openEuler 24.03 ?
  - The `CPLUS_INCLUDE_PATH` requires user manual setup, please see the [dockerfile](./docker/Dockerfile.a2.openEuler)
3. Why do I have error for the `example/offload.py` in the main LMCache repo ?
  - The import order can affect the LMCacheAscend connector, therefore please see our example [here](./examples/offload.py).
4. Raise a missing header file error while `#include <numaif.h>`.
  - Execute `yum install numactl-devel`.
