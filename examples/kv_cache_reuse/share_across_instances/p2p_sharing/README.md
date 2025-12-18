## Example of P2P KV Cache Sharing in vLLM v1

This example demonstrates how to run LMCache with P2P KV Cache Sharing using HCCL on a single node.

### Prerequisites

- CANN 8.2.RC1+
- RoCE connected NPU server (HCCS will be supported later)
- Install Torch and Torch NPU 2.7.1
    ```bash
    pip3 install 'numpy<2.0' decorator
    pip3 install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    pip3 install 'torch-npu==2.7.1'
    pip3 install wheel pybind11 pyyaml psutil scipy attrs
    ```
- Install VLLM v0.10.2
    ```bash
    VLLM_REPO=https://github.com/vllm-project/vllm.git
    VLLM_TAG=v0.10.2
    git clone --depth 1 $VLLM_REPO --branch $VLLM_TAG /home/amory/vllm
    cd /home/amory/vllm
    # Apply transfer performance fix on ARM in anticipation of https://github.com/vllm-project/vllm/pull/30228
    git apply /home/amory/LMCache-Ascend/docker/vllm-utils.diff
    # NOTE: There is an Ascend Triton but we don't currently support it properly.
    VLLM_TARGET_DEVICE="empty" python3 -m pip install -e /home/amory/vllm/ --extra-index https://download.pytorch.org/whl/cpu/ && \
        python3 -m pip uninstall -y triton
    ```
- Install VLLM-Ascend v0.10.2rc1
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    VLLM_ASCEND_REPO=https://github.com/vllm-project/vllm-ascend.git
    VLLM_ASCEND_TAG=v0.10.2rc1
    git clone --depth 1 $VLLM_ASCEND_REPO --branch $VLLM_ASCEND_TAG /home/amory/lmcache/vllm-ascend
    cd /home/amory/lmcache/vllm-ascend
    # Apply fix for vllm scheduler to postpone scheduling of async fetched sequences
    git apply /home/amory/LMCache-Ascend/docker/vllm-sched.diff

    export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib && \
    python3 -m pip install -v -e /home/amory/lmcache/vllm-ascend/ --extra-index https://download.pytorch.org/whl/cpu/
    ```
- Install LMcache
    ```bash
    LMCACHE_REPO=https://github.com/LMCache/LMCache.git
    LMCACHE_TAG=v0.3.9post2
    git clone --depth 1 $LMCACHE_REPO --branch $LMCACHE_TAG /home/amory/lmcache/LMCache
    # our build is based on arm64
    sed -i "s/^infinistore$/infinistore; platform_machine == 'x86_64'/" /home/amory/lmcache/LMCache/requirements/common.txt
    export NO_CUDA_EXT=1 && python3 -m pip install -v -e /home/amory/lmcache/LMCache
    ```
- Install LMCache-Ascend
    ```bash
    cd /home/amory/LMCache-Ascend
    pip install -e . --no-build-isolation
    ```
- At least 2 GPUs

### Usage

> Note: Currently `LMCACHE_MAX_LOCAL_CPU_SIZE` can not be larger than 21GB due to a device page table limitation. This will be fixed in a later driver update.

Launch controller

```bash
PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-ports '{"pull": 8600, "reply": 8700}'
```

Launch instance 1

```bash
export LMCACHE_CONFIG_FILE=/home/amory/lmcache/LMCache-Ascend/examples/kv_cache_reuse/share_across_instances/p2p_sharing/example1.yaml
export ASCEND_RT_VISIBLE_DEVICES=4,5
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=123
export LMCACHE_MAX_LOCAL_CPU_SIZE=9 # Avoid https://github.com/LMCache/LMCache/pull/1924
python \
    -m vllm.entrypoints.openai.api_server \
    --port 8010 \
    --model /data/models/Qwen/Qwen3-8B \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --disable-log-requests \
    --block-size 128 \
    --rope-scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
    --max-model-len 32768 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}' > instance1.txt 2>&1 

```

Launch instance 2
```bash
export LMCACHE_CONFIG_FILE=/home/amory/lmcache/LMCache-Ascend/examples/kv_cache_reuse/share_across_instances/p2p_sharing/example2.yaml
export ASCEND_RT_VISIBLE_DEVICES=6,7
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=123
export LMCACHE_MAX_LOCAL_CPU_SIZE=9 # Avoid https://github.com/LMCache/LMCache/pull/1924
python \
    -m vllm.entrypoints.openai.api_server \
    --port 8011 \
    --model /data/models/Qwen/Qwen3-8B \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --disable-log-requests \
    --block-size 128 \
    --rope-scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
    --max-model-len 32768 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}' > instance2.txt 2>&1 
```

Send request to engine 1
```bash
curl -X POST http://localhost:8010/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/data/models/Qwen/Qwen3-8B\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models in English.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```

Send request to engine 2
```bash
curl -X POST http://localhost:8011/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/data/models/Qwen/Qwen3-8B\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models in English.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```

The cache will be automatically retrieved from vllm engine 1. You should be able to see logs (from vllm engine 2) like the following:
```
(EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,706] LMCache INFO:�[0m Established connection to peer_init_url localhost:8200. The peer_lookup_url: localhost:8201 (p2p_backend.py:278:lmcache.v1.storage_backend.p2p_backend)
(EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,792] LMCache INFO: Retrieved 1002 out of total 1002 out of total 1002 tokens. size: 0.1223 gb, cost 60.3595 ms, throughput: 2.0264 GB/s; (cache_engine.py:496:lmcache.v1.cache_engine)
```
