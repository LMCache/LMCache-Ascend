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
  - **Python**: >= 3.10, <= 3.11
  - **CANN Toolkit**: >= 8.2.RC1
  - **Ascend Driver**: >= 24.1.0
  - **PyTorch**: >= 2.7.1 (For vLLM 0.10.2+)
  - **vLLM**: >=v0.11.0 & **vLLM-Ascend**: >=v0.11.0

### Compatibility Matrix

Please ensure your environment matches the versions below.

#### For PyTorch / vLLM
| LMCache-Ascend | LMCache | vLLM Version | PyTorch / Torch-NPU | Status |
| :--- | :--- | :--- | :--- | :--- |
| **main** | **v0.4.2** | **>=v0.11.0** | **>=2.7.1** | 🚧 **Experimental** |
| **v0.3.12** | **v0.3.12** | **>=v0.11.0** | **>=2.7.1** | ✅ **Verified (Recommended)** |

#### For PyTorch / SGLang
| LMCache-Ascend | LMCache | SGLang Version | PyTorch / Torch-NPU | Status |
| :--- | :--- | :--- | :--- | :--- |
| **main** | **v0.4.2** | **0.5.8** | **2.8.0.post2.dev20251113** | 🚧 **Experimental** |
| **v0.3.12** | **v0.3.12** | **0.5.8** | **2.8.0.post2.dev20251113** | ✅ **Verified (Recommended)** |

#### for MindSpore
| LMCache-Ascend | LMCache | vLLM Version | MindSpore | Status |
| :--- | :--- | :--- | :--- | :--- |
| **main** | **v0.4.2** | **v0.11.0** | **2.7.1.post1** | 🚧 **Experimental** |
| **v0.3.12** | **v0.3.12** | **v0.11.0** | **2.7.1.post1** | ✅ **Verified (Recommended)** |

> **Note**: If you require legacy support for vLLM 0.9.2, you must use PyTorch 2.5.1. See the [Compatibility Matrix](#compatibility-matrix) above.


## Getting Started

### Clone LMCache-Ascend Repo

Our repo contains a kvcache ops submodule for ease of maintenance, therefore we recommend cloning the repo with submodules.

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
```

### Docker(for vLLM)

Build the image using the provided Dockerfile:
```bash
cd /workspace/LMCache-Ascend
docker build -f docker/Dockerfile.a2.openEuler -t lmcache-ascend:v0.4.2-vllm-ascend-v0.18.0rc1-openeuler .
```

Once that is built, run it with the following cmd
```bash
docker run -itd \
    --shm-size=200g --privileged --net=host \
    --cap-add=SYS_RESOURCE \
    --cap-add=IPC_LOCK \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/localtime:/etc/localtime \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /var/log/npu:/var/log/npu \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /usr/src/kernels:/usr/src/kernels:ro \
    --name lmcache-ascend-dev \
    lmcache-ascend:v0.4.2-vllm-ascend-v0.18.0rc1-openeuler \
    /bin/bash
```

For further info about deployment notes, please refer to the [guide about deployment](docs/deployment.md)

### Manual Installation
Assuming your working directory is ```/workspace``` and base environment(vllm/vllm-ascend or sglang) have already been installed.

1. Prepare Base Environment

** For vLLM-Ascend **
If you are using vllm-ascend, it is recommended to use the official [vLLM-Ascend image](https://quay.io/repository/ascend/vllm-ascend?tab=tags) as a base:

```bash
# Pull and run the official vLLM-Ascend image
docker pull quay.io/ascend/vllm-ascend:v0.18.0rc1
docker run -it --privileged --net=host --name lmcache-vllm-dev quay.io/ascend/vllm-ascend:v0.18.0rc1 /bin/bash
```

** For SGLang **
If you are using SGLang, it is recommended to use the official [Ascend SGLang image](https://quay.io/repository/ascend/sglang?tab=tags) as a base:

```bash
# Pull and run the official SGLang image
docker pull quay.io/ascend/sglang:v0.5.8-cann8.3.rc2-910b
docker run -it --privileged --net=host --name lmcache-sglang-dev quay.io/ascend/sglang:v0.5.8-cann8.3.rc2-910b /bin/bash
```

2. Install LMCache Repo

- from pip
```bash
NO_CUDA_EXT=1 pip install lmcache==0.4.2
```

3. Install LMCache-Ascend Repo

```bash
cd /workspace/LMCache-Ascend
pip install -v --no-build-isolation -e .
```

### Usage

We introduce a dynamic KVConnector via LMCacheAscendConnectorV1Dynamic, therefore LMCache-Ascend Connector can be used via the kv transfer config in the two following setting.

#### Using with vLLM
LMCache-Ascend integrates with vLLM via a dynamic KVConnector.

##### Online serving
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

##### Offline
```python
ktc = KVTransferConfig(
        kv_connector="LMCacheAscendConnectorV1Dynamic",
        kv_role="kv_both",
        kv_connector_module_path="lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    )
```

> **Note**: For vllm-ascend versions >=0.17.0rc1, you can specify `--kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}'`

#### Using with SGLang
For SGLang, integration is simplified. You do not need to specify a kv_connector; simply enable the LMCache flag(`--enable-lmcache`).
```bash
python \
    -m sglang.launch_server \
    --model-path /data/models/Qwen/Qwen3-32B \
    --trust-remote-code \
    --device npu \
    --attention-backend ascend \
    --mem-fraction-static 0.8 \
    --cuda-graph-max-bs 16 \
    --tp-size 4 \
    --host 0.0.0.0 \
    --enable-lmcache \
    --port 8100
```

## Getting Started With MindSpore

### Docker

1. Clone LMCache-Ascend Repo
Our repo contains a kvcache ops submodule for ease of maintenance, therefore we recommend cloning the repo with submodules.

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
```

2. Build Docker Image
```bash
cd /workspace/LMCache-Ascend
docker build -f docker/mindspore/Dockerfile.a2.openEuler -t lmcache-ascend:v0.4.2-mindspore2.7.1.post1-openeuler .
```

3. Start Container
Once that is built, run it with the following cmd
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
    -v /home:/home \
    --name lmcache-ascend-ms \
    --entrypoint /bin/bash \
    lmcache-ascend:v0.4.2-mindspore2.7.1.post1-openeuler

docker exec -it -u root lmcache-ascend-ms bash
```

For further info about deployment notes, please refer to the [guide about deployment](docs/deployment.md)

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
hub.oepkgs.net/oedeploy/openeuler/aarch64/intelligence_boom:0.2.0-aarch64-800I-A2-mindspore2.7.1.post1-openeuler24.03-lts-sp2-20260116

docker exec -it -u root lmcache-ascend-ms bash
```

2. Install LMCache

```bash
NO_CUDA_EXT=1 pip install lmcache==0.4.2 --no-deps
```

3. Install LMCache-Ascend

```bash
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
USE_MINDSPORE=1 pip install -r requirement_ms.txt --no-build-isolation -v -e .
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
  - Otherwise, please check your driver version is >= 24.1.0
2. Why do I have build error related to `cstdint` during manual installation using openEuler 24.03 ?
  - The `CPLUS_INCLUDE_PATH` requires user manual setup, please see the [dockerfile](./docker/Dockerfile.a2.openEuler)
3. Why do I have error for the `example/offload.py` in the main LMCache repo ?
  - The import order can affect the LMCacheAscend connector, therefore please see our example [here](./examples/offload.py).
4. Raise a missing header file error while `#include <numaif.h>`.
  - Execute `yum install numactl-devel`.
