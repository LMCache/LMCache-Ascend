#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# TODO: Replace /xxx/ with your workspace and set the actual model path below.
# Alternatively, export WORK_DIR and MODEL before running this script.
WORK_DIR="${WORK_DIR:-/xxx/}"
MODEL="${MODEL:-/xxx/models/Qwen3.8-27B}"

test -d "$MODEL" || { echo "Model directory not found: $MODEL" >&2; exit 1; }
cd "$WORK_DIR"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_BUFFSIZE=512
# LMCache prefix hashing.
export PYTHONHASHSEED=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=INFO
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_CONFIG_FILE="$SCRIPT_DIR/lmcache_qwen38.yaml"

mkdir -p "$WORK_DIR/logs/qwen38"
LOG_FILE="$WORK_DIR/logs/qwen38/lmcache-mtp3-eager-$(date +%Y%m%d-%H%M%S).log"
printf '%s\n' "$LOG_FILE" > "$SCRIPT_DIR/latest-qwen38-log.txt"
echo "Model: $MODEL; BF16 TP8; MTP3; target eager; draft eager"
echo "LMCache config: $LMCACHE_CONFIG_FILE"
echo "Server log: $LOG_FILE"

# General serving settings follow the official Qwen3.8 A2/A3 example.
# Local BF16 model, TP8 and port 8056 follow this deployment.
# LMCache hybrid requires align, hybrid cache manager and synchronous scheduling.
# Qwen3.8 keeps target eager due to the observed short-prefill graph failure.
vllm serve "$MODEL" \
    --served-model-name qwen3.8 \
    --host 0.0.0.0 \
    --port 8056 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --enforce-eager \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --trust-remote-code \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --no-disable-hybrid-kv-cache-manager \
    --no-async-scheduling \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 32 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --additional-config '{"enable_cpu_binding":true}' \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector"}' \
    2>&1 | tee "$LOG_FILE"
