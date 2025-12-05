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
# LMCache-Ascend合并版本部署指南

LMCache-Ascend 对 torch 版本（原 LMCache-Ascend） 和 mindspore 版本（原 LMCache-Mindspore）进行了合并，通过一系列环境变量进行 build 时的分支控制

**若需要在两个版本之间切换，请手动删除 /build 目录**

## Torch

参照 [LMCache-Ascend 部署指南](https://codehub-g.huawei.com/j00803670/LMCache-Ascend/files?ref=main&filePath=README.md&isFile=true)，除了修改 clone 的分支以外没有改动需求，不应该设置多余的环境变量等信息。

在运行时，`kv_connector_module_path` 需要将 `lmcache_ascend.integration` 修改成 `lmcache_ascend.integration`

个人遇到的一个问题：首次 build 需要 `python setup.py build_py` 获取 `_build_info.py`，不知道有没有普适性，这部分逻辑应该和 LMCache-Ascend 保持了一致。

## Mindspore

先参考 [Deepseek-R1&LMCache部署指南](https://gitee.com/src-openeuler/LMCache/blob/vllm-ms-dev/docs/tutorials/Deepseek-R1%26LMCache%E9%83%A8%E7%BD%B2%E6%8C%87%E5%8D%97.md#)，完成 vllm、vllm-mindspore 等组件的部署

在成功运行后，克隆本仓库和 LMCache-opensource 到 `/workspace` 目录下（可更改，下文同），设置必要的环境变量：

`vim /workspace/lmcache_config.yaml` 创建 config 文件，内容为：
```bash
chunk_size: 256
local_cpu: true
max_local_cpu_size: 60
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
export LMCACHE_TARGET_DEVICE="ASCEND"
export USE_TORCH=0
export LMCACHE_CONFIG_FILE="/workspace/lmcache_config.yaml"
export LD_LIBRARY_PATH=/workspace/python-LMCache-0.3.1.post1/lmcache/:${LD_LIBRARY_PATH}
export LMCACHE_LOG_LEVEL=INFO
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export USE_TORCH=False
export DISABLE_CUSTOM_OPS=0
export VLLM_USE_V1=1
export HCCL_IF_BASE_PORT=50559
export HCCL_SOCKET_IFNAME= #实际网卡名
export GLOO_SOCKET_IFNAME= #实际网卡名
export TP_SOCKET_IFNAME= #实际网卡名
export HCCL_CONNECT_TIMEOUT=7200
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ALLOC_CONF=enable_vmm:True
export ASCEND_RT_VISIBLE_DEVICES=3
export ASCEND_CUSTOM_PATH=$ASCEND_HOME_PATH/../
export ASCEND_TOTAL_MEMORY_GB=64
export VLLM_LOGGING_LEVEL=DEBUG
export CPU_AFFINITY=0
export PYTHONPATH=/workspace/mindformers:$PYTHONPATH
export EXPERIMENTAL_KERNEL_LAUNCH_GROUP="thread_num:4,kernel_group_num:16"
export MS_INTERNAL_ENABLE_NZ_OPS="QuantBatchMatmul,MlaPreprocess,GroupedMatmulV4"
export MS_DISABLE_INTERNAL_KERNELS_LIST="AddRmsNorm,Add,MatMul,Cast"
export USE_MINDSPORE=1 # 非常重要
```

可能需要补充一个依赖包 `pip install nvtx`，版本无要求直接装就行

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
安装
```bash
cd /workspace/LMCache-Ascend && pip install --no-build-isolation -v -e .
```

随后可以拉起服务
```bash
cd /workspace/LMCache-opensource/ && python -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server \
--model /path_to_model/ \
--port 12345 \
--max-model-len 32768 \
--max-num-seqs 2 \
--gpu-memory-utilization 0.84 \
--trust-remote-code \
--tensor-parallel-size 1 \
--disable-log-requests \
--block-size 64 \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```
