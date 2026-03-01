## Example of Disaggregated Prefill in vLLM v1

This example demonstrates how to run LMCache with disaggregated prefill on a single node.

> Note: for multi-nodes setting, replace the localhost with ip addresses accordingly.

### Prerequisites

- CANN 8.3+ (for `hccl` channel) or CANN 8.5+ (for `hcomm_onesided` / `hixl` channels)
- Ascend HDK 25.5.0+ drivers and firmware
- RoCE connected NPU server (HCCS will be supported later)
- At least 2 NPUs
- The following patches from `docker/` must be applied before use:
  - `docker/vllm-utils.diff` to vLLM
  - `docker/vllm-sched.diff` to vLLM-Ascend

> After applying patches, reinstall the affected packages (vLLM, vLLM-Ascend, LMCache, LMCache-Ascend) for the changes to take effect.

### Transfer Channel Configuration

The `transfer_channel` field in the LMCache YAML config selects the NPU communication backend used for KV cache transfer. Set this in both `configs/lmcache-prefiller-config.yaml` and `configs/lmcache-decoder-config.yaml`.

| Channel | CANN Requirement | Status |
| :--- | :--- | :--- |
| `hccl` | CANN 8.3+ | Legacy |
| `hcomm_onesided` | CANN 8.5+ | **Recommended** |
| `hixl` | CANN 8.5+ | Experimental |

To switch channels, update the `transfer_channel` field in your YAML configs:

```yaml
# CANN 8.3+ (legacy)
transfer_channel: "hccl"

# CANN 8.5+ (recommended)
transfer_channel: "hcomm_onesided"

# CANN 8.5+ (experimental)
transfer_channel: "hixl"
```

The build system auto-detects the installed CANN version and compiles the correct backend. The provided example configs default to `hcomm_onesided`.

#### Buffer Size

The `pd_buffer_size` field in the YAML configs allocates extra memory on the NPU (or CPU, depending on `pd_buffer_device`) as a transfer buffer. This size should be tuned based on the longest expected sequence length -- larger buffers allow longer sequences but consume more device memory.

### Usage

Launch prefill
```bash
export LMCACHE_CONFIG_FILE=/workspace/LMCache-Ascend/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-config.yaml
export ASCEND_RT_VISIBLE_DEVICES=4,5
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
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
