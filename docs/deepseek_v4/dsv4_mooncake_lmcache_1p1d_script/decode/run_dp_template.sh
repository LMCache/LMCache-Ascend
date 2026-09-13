#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

nic_name="enp23s0f3"
local_ip=192.168.0.50

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export HCCL_INTRA_ROCE_ENABLE=1
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1500
export TASK_QUEUE_ENABLE=1
export HCCL_DEBUG_CONFIG="ALG,TASK,RESOURCE,AIV_OPS_EXC"
export HCCL_DFS_CONFIG="connection_fault_detection_time:30,cluster_heartbeat:on,stuck_detection:on,inconsistent_check:off,task_monitor_interval:0"
export ASCEND_CONNECT_TIMEOUT=120000
export ASCEND_RT_VISIBLE_DEVICES=$1
export PROFILING_MODE=dynamic

export PYTHONHASHSEED=0

vllm serve /mnt/sdb/models/DeepSeek-V4-Flash-w8a8-mtp \
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
    --max-num-batched-tokens 120 \
    --max-num-seqs 60 \
    --async-scheduling \
    --no-disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --safetensors-load-strategy 'prefetch' \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --block-size 128 \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --profiler-config \
        '{"profiler": "torch",
        "torch_profiler_dir": "./vllm_profile",
        "torch_profiler_with_stack": false}' \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_consumer",
    "kv_port": "36100",
    "engine_id": "1",
    "kv_connector_extra_config": {
                "prefill": {"dp_size": 8, "tp_size": 2},
                "decode": {"dp_size": 16, "tp_size": 1}
        }
    }' \
    --additional-config '{
        "ascend_compilation_config":{
              "enable_npugraph_ex":true,
              "enable_static_kernel":false
        },
       "enable_cpu_binding":true,
       "multistream_overlap_shared_expert":false,
       "multistream_dsa_preprocess":false,
       "recompute_scheduler_enable":true
    }' \
    2>&1 | tee "model_decode_dp${4}.log"
