#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export HCCL_IF_IP=192.168.0.223
export GLOO_SOCKET_IFNAME=enp23s0f3
export TP_SOCKET_IFNAME=enp23s0f3
export HCCL_SOCKET_IFNAME=enp23s0f3

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export ASCEND_CONNECT_TIMEOUT=120000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_DEBUG_CONFIG="ALG,TASK,RESOURCE,AIV_OPS_EXC"
export HCCL_DFS_CONFIG="connection_fault_detection_time:30,cluster_heartbeat:on,stuck_detection:on,inconsistent_check:off,task_monitor_interval:0"
export ASCEND_RT_VISIBLE_DEVICES="$1"
export PROFILING_MODE=dynamic

export LMCACHE_CONFIG_FILE="${SCRIPT_DIR}/lmcache-prefill-config.yaml"
export PYTHONHASHSEED=0

exec vllm serve /mnt/sdb/models/DeepSeek-V4-Flash-w8a8-mtp \
--host 0.0.0.0 \
--port "$2" \
--api-server-count 1 \
--data-parallel-size "$3" \
--data-parallel-rank "$4" \
--data-parallel-address "$5" \
--data-parallel-rpc-port "$6" \
--tensor-parallel-size "$7" \
--enable-expert-parallel \
--seed 1024 \
--served-model-name dsv4 \
--max-model-len 131072 \
--max-num-batched-tokens 8192 \
--max-num-seqs 16 \
--no-disable-hybrid-kv-cache-manager \
--safetensors-load-strategy 'prefetch' \
--speculative-config '{"num_speculative_tokens":1,"method":"mtp"}' \
--trust-remote-code \
--tokenizer-mode deepseek_v4 \
--tool-call-parser deepseek_v4 \
--enable-auto-tool-choice \
--reasoning-parser deepseek_v4 \
--gpu-memory-utilization 0.92 \
--quantization ascend \
--enforce-eager \
--block-size 128 \
--enable-prefix-caching \
--model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
--profiler-config \
    '{"profiler": "torch",
    "torch_profiler_dir": "./vllm_profile",
    "torch_profiler_with_stack": false}' \
--additional-config '{"enable_cpu_binding":true}' \
--kv-transfer-config '{
    "kv_connector":"MultiConnector",
    "kv_role":"kv_producer",
    "engine_id": "0",
    "kv_connector_extra_config":{
        "connectors":[
            {
                "kv_connector":"MooncakeHybridConnector",
                "kv_role":"kv_producer",
                "kv_port":"36000",
                "kv_connector_extra_config":{
                    "prefill":{"dp_size":8,"tp_size":2},
                    "decode":{"dp_size":16,"tp_size":1}
                }
            },
            {
                "kv_connector": "LMCacheAscendConnectorV1Dynamic",
                "kv_role": "kv_producer",
                "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
            }
        ]
    }
}' \
2>&1 | tee "model_lmcache_noprefix_dp${4}.log"

# --no-enable-prefix-caching \
