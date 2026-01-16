## Example of Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill using HCCL on a single node.

> Note: PD Disaggregation currently does not work for prompts smaller than vLLM block size (256 default on Ascend) as described [here](https://github.com/LMCache/LMCache/issues/2459)

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
- Install VLLM v0.11.0
    ```bash
    VLLM_REPO=https://github.com/vllm-project/vllm.git
    VLLM_TAG=v0.11.0
    git clone --depth 1 $VLLM_REPO --branch $VLLM_TAG /workspace/vllm
    cd /workspace/vllm
    # Apply transfer performance fix on ARM in anticipation of https://github.com/vllm-project/vllm/pull/30228
    git apply /workspace/LMCache-Ascend/docker/vllm-utils.diff
    # NOTE: There is an Ascend Triton but we don't currently support it properly.
    VLLM_TARGET_DEVICE="empty" python3 -m pip install -e /workspace/vllm/ --extra-index https://download.pytorch.org/whl/cpu/ && \
        python3 -m pip uninstall -y triton
    ```
- Install VLLM-Ascend v0.11.0
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    VLLM_ASCEND_REPO=https://github.com/vllm-project/vllm-ascend.git
    VLLM_ASCEND_TAG=v0.11.0
    git clone --depth 1 $VLLM_ASCEND_REPO --branch $VLLM_ASCEND_TAG /workspace/vllm-ascend
    cd /workspace/vllm-ascend
    # Apply fix for vllm scheduler to postpone scheduling of async fetched sequences
    git apply /workspace/LMCache-Ascend/docker/vllm-sched.diff

    export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib && \
    python3 -m pip install -v -e /workspace/vllm-ascend/ --extra-index https://download.pytorch.org/whl/cpu/
    ```
- Install LMcache v0.3.12
    ```bash
    LMCACHE_REPO=https://github.com/LMCache/LMCache.git
    LMCACHE_TAG=v0.3.12
    git clone --depth 1 $LMCACHE_REPO --branch $LMCACHE_TAG /workspace/LMCache
    # our build is based on arm64
    sed -i "s/^infinistore$/infinistore; platform_machine == 'x86_64'/" /workspace/LMCache/requirements/common.txt
    export NO_CUDA_EXT=1 && python3 -m pip install -v -e /workspace/LMCache
    ```
- Install LMCache-Ascend
    ```bash
    cd /workspace/LMCache-Ascend
    git submodule update --init --recursive
    
    pip install -e . --no-build-isolation
    ```
- At least 2 GPUs

### Usage

Launch prefill
```bash
export LMCACHE_CONFIG_FILE=/workspace/LMCache-Ascend/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-config.yaml
export ASCEND_RT_VISIBLE_DEVICES=4,5
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
export VLLM_VERSION=0.11.0
python \
    -m vllm.entrypoints.openai.api_server \
    --port 7100 \
    --model /data/models/Qwen/Qwen3-8B \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --disable-log-requests \
    --block-size 128 \
    --rope-scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
    --max-model-len 32768 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_producer", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}' > prefill.txt 2>&1 
```

Launch decode
```bash
export LMCACHE_CONFIG_FILE=/workspace/LMCache-Ascend/examples/disagg_prefill/1p1d/configs/lmcache-decoder-config.yaml
export ASCEND_RT_VISIBLE_DEVICES=6,7
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
export VLLM_VERSION=0.11.0
python \
    -m vllm.entrypoints.openai.api_server \
    --port 7200 \
    --model /data/models/Qwen/Qwen3-8B \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --disable-log-requests \
    --block-size 128 \
    --rope-scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
    --max-model-len 32768 \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_consumer", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}' > decode.txt 2>&1 
```

Launch proxy server to coordinate prefill and decode

```bash
python3 /workspace/LMCache/examples/disagg_prefill/disagg_proxy_server.py \
  --host localhost \
  --port 9100 \
  --prefiller-host localhost \
  --prefiller-port 7100 \
  --num-prefillers 1 \
  --decoder-host localhost \
  --decoder-port 7200  \
  --decoder-init-port "7300,7301" \
  --decoder-alloc-port "7400,7401" \
  --proxy-host localhost \
  --proxy-port 7500 \
  --num-decoders 1
```

Send request to engine 1
```bash
curl -X POST http://localhost:9100/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/data/models/Qwen/Qwen3-8B\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models in English.%.0s' {1..100})\",
    \"max_tokens\": 100
  }"
```