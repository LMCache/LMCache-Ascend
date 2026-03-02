## Example of P2P KV Cache Sharing in vLLM v1

This example demonstrates how to run LMCache with P2P KV Cache Sharing on a single node.

### Prerequisites

- CANN 8.3+ (for `hccl` channel) or CANN 8.5+ (for `hcomm_onesided` / `hixl` channels)
- Ascend HDK 25.5.0+ drivers and firmware. Previous drivers only support registering up to ~20GB of host memory to the NPU NIC.
- RoCE connected NPU server (HCCS will be supported later)
- At least 2 NPUs
- The following patches from `docker/` must be applied before use:
  - `docker/vllm-utils.diff` to vLLM
  - `docker/vllm-sched.diff` to vLLM-Ascend
  - `docker/lmcache-controller.diff` to LMCache (required for P2P sharing with TP>1)

> After applying patches, reinstall the affected packages (vLLM, vLLM-Ascend, LMCache, LMCache-Ascend) for the changes to take effect.

### Transfer Channel Configuration

The `transfer_channel` field in the LMCache YAML config selects the NPU communication backend used for KV cache transfer. Set this in both `example1.yaml` and `example2.yaml`.

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

### Usage

Ensure the LMCache config files are correctly configured for the desired parallelism level. To use P2P KV Cache Sharing across hosts, substitute localhost with the IP of the corresponding server.

Launch controller

```bash
PYTHONHASHSEED=123 lmcache_controller --host 0.0.0.0 --port 9000 --monitor-ports '{"pull": 8600, "reply": 8700}'
```

Launch instance 1

```bash
export LMCACHE_CONFIG_FILE=/workspace/LMCache-Ascend/examples/kv_cache_reuse/share_across_instances/p2p_sharing/example1.yaml
export ASCEND_RT_VISIBLE_DEVICES=2,3
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=123
export LMCACHE_MAX_LOCAL_CPU_SIZE=16
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
export LMCACHE_CONFIG_FILE=/workspace/LMCache-Ascend/examples/kv_cache_reuse/share_across_instances/p2p_sharing/example2.yaml
export ASCEND_RT_VISIBLE_DEVICES=6,7
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=123
export LMCACHE_MAX_LOCAL_CPU_SIZE=16
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
time curl -X POST http://localhost:8010/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/data/models/Qwen/Qwen3-8B\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models in English.%.0s' {1..1000})\",
    \"max_tokens\": 10,
    \"temperature\": 0
  }"
```

Send request to engine 2
```bash
time curl -X POST http://localhost:8011/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/data/models/Qwen/Qwen3-8B\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models in English.%.0s' {1..1000})\",
    \"max_tokens\": 10,
    \"temperature\": 0
  }"
```

The cache will be automatically retrieved from vllm engine 1. You should be able to see logs (from vllm engine 2) like the following:
```
(EngineCore_DP0 pid=2577584) LMCache INFO: Established connection to peer_init_url localhost:8200. The peer_lookup_url: localhost:8201
(EngineCore_DP0 pid=2577584) LMCache INFO: Retrieved 1002 out of total 1002 tokens. size: 0.1223 gb, cost 60.3595 ms, throughput: 2.0264 GB/s
```
