set -euo pipefail

#!/bin/bash
MODE="both"  # Default mode
if [ $# -gt 0 ]; then
    MODE="$1"
fi

# Validate mode
if [[ "$MODE" != "server" && "$MODE" != "client" && "$MODE" != "both" ]]; then
    echo "Usage: $0 [server|client|both]" >&2
    echo "  server: Start vLLM server" >&2
    echo "  client: Run RAG benchmark client (assuming server is running)" >&2
    echo "  both:   Sequentially run all benchmarks (default)" >&2
    exit 1
fi

# Configuration for online benchmark
MODEL_NAME="LLM-Research/Meta-Llama-3.1-8B-Instruct"
DATASET_PATH="musique_s.json"
END=10 # process 10 examples

LMCache_CONFIG_FILE_PATH="./config/lmcache.yaml"
CacheBlend_CONFIG_FILE_PATH="./config/lmcache_blend.yaml"

PROMPT_BUILD_METHOD=QA
MAX_TOKENS=32
TEMPERATURE=0.0
API_KEY="dummy-key"
BASE_PORT=8500

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS_DIR="$SCRIPT_DIR/online_benchmark_results"
mkdir -p "$RESULTS_DIR"

BENCHMARKS=(
    "lmcache"
    "cacheblend"
)

declare -A SERVER_PIDS
CSV_PATHS=()

# Function to start vLLM server
start_vllm_server() {
    local config_name="$1"
    local port="$2"
    local config_file="$3"
    local kv_args="$4"
    local wait_for_ready=${5:-true}
    local api_base="http://localhost:$port/v1"

    echo "--- Starting $config_name server on port $port ---" >&2

    local cmd="vllm serve \"$MODEL_NAME\" \
        --port $port \
        --host localhost \
        --api-key \"$API_KEY\" \
        --disable-log-requests \
        --max-model-len 32000 \
        --gpu-memory-utilization 0.8 \
        --served-model-name \"$MODEL_NAME\""

    if [[ -n "$config_file" ]]; then
        cmd="LMCACHE_CONFIG_FILE=$config_file $cmd"
    fi

    if [[ -n "$kv_args" ]]; then
        cmd="$cmd --kv-transfer-config '$kv_args'"
    fi

    if [[ "$config_name" == "cacheblend" ]]; then
        cmd="$cmd --no-enable-prefix-caching"
        echo "Note: Disabling prefix caching for $config_name." >&2
    else
        echo "Note: Prefix caching enabled/default for $config_name." >&2
    fi

    eval "$cmd &"
    SERVER_PIDS["$config_name"]=$!
    echo "Server $config_name started with PID: ${SERVER_PIDS[$config_name]}" >&2
    
    if [ "$wait_for_ready" = "true" ]; then
        echo "Waiting for $config_name server to be ready..." >&2
        for i in {1..300}; do
            if curl -s "$api_base/models" > /dev/null 2>&1; then
                echo "$config_name server is ready!" >&2
                return 0
            fi
            if [ $i -eq 300 ]; then
                echo "ERROR: $config_name server failed to start within 300 seconds" >&2
                stop_vllm_server "$config_name"
                return 1
            fi
            sleep 1
            echo -n "." >&2
        done
    fi
    return 0
}

# Function to run RAG benchmark client
run_client() {
    local config_name="$1"
    local port="$2"
    
    DATASET_NAME=$(echo $DATASET_PATH | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
    local output_file="$RESULTS_DIR/${DATASET_NAME}_${config_name}_online.csv"
    
    echo "========================================" >&2
    echo "Running $config_name benchmark on port $port..." >&2
    echo "Output: $output_file" >&2
    echo "========================================" >&2
    
    LMCACHE_CONFIG_FILE="$3" python3 rag.py \
        --online \
        --start-index 0 \
        --end-index $END \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_PATH" \
        --prompt-build-method $PROMPT_BUILD_METHOD \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --openai-api-base "http://localhost:$port/v1" \
        --openai-api-key "$API_KEY" \
        --output "$output_file" \
        --verbose >&2
    
    echo "--- $config_name benchmark completed. Results saved to $output_file ---" >&2

    CSV_PATHS+=("$output_file")
}

# Function to stop specific vLLM server
stop_vllm_server() {
    local config_name="$1"
    local pid="${SERVER_PIDS[$config_name]}"

    if [ ! -z "$pid" ]; then
        echo "Stopping $config_name server (PID: $pid)..." >&2
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        echo "$config_name server stopped." >&2
        unset SERVER_PIDS["$config_name"]
    fi
}

# Function to stop all servers
stop_all_servers() {
    for config_name in "${!SERVER_PIDS[@]}"; do
        stop_vllm_server "$config_name"
    done
}

# Function to run single benchmark
run_single_benchmark() {
    local name="$1"
    local port="$2"
    local config_file="$3"
    local kv_args="$4"

    echo "" >&2
    echo "======================================================" >&2
    echo "         RUNNING BENCHMARK: $name" >&2
    echo "======================================================" >&2
    
    if start_vllm_server "$name" "$port" "$config_file" "$kv_args"; then
        run_client "$name" "$port" "$config_file"
        stop_vllm_server "$name"
        echo "=== $name benchmark completed ===" >&2
    else
        echo "Skipping benchmark $name due to server startup failure." >&2
    fi
    
    echo "" >&2
    sleep 3
}

# Set up trap for clean up
if [[ "$MODE" == "server" || "$MODE" == "both" ]]; then
    trap stop_all_servers EXIT
fi

echo "Starting online RAG benchmark in '$MODE' mode..." >&2
echo "Model: $MODEL_NAME" >&2
echo "Dataset: $DATASET_PATH" >&2
echo "Benchmarks to run: ${BENCHMARKS[@]}" >&2
echo "----------------------------------------" >&2

# Execute based on mode
case "$MODE" in
    "both")
        echo "Sequentially running all benchmarks..." >&2
        
        KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}'
        
        current_port=$BASE_PORT
        
        for benchmark in "${BENCHMARKS[@]}"; do
            case $benchmark in
                "lmcache")
                    run_single_benchmark "lmcache" "$current_port" "$LMCache_CONFIG_FILE_PATH" "$KV_TRANSFER_CONFIG"
                    ;;
                "cacheblend")
                    run_single_benchmark "cacheblend" "$current_port" "$CacheBlend_CONFIG_FILE_PATH" "$KV_TRANSFER_CONFIG"
                    ;;
            esac
            ((current_port++))
        done
        
        echo "======================================================" >&2
        echo "      ALL BENCHMARKS COMPLETED" >&2
        echo "======================================================" >&2
        echo "CSV files saved to $RESULTS_DIR:" >&2

        if [ ${#CSV_PATHS[@]} -gt 0 ]; then
            echo -e "\nRunning analysis script..." >&2

            pip install -q pandas matplotlib seaborn >&2

            if python3 "$SCRIPT_DIR/analysis.py" \
                --results-dir "$RESULTS_DIR" \
                "${CSV_PATHS[@]}" >&2; then
                echo -e "\n=== All done! Plots and results in $RESULTS_DIR ===" >&2
            else
                echo "Error: Analysis script failed." >&2
            fi
        else
            echo "Skipping analysis: No CSV files were successfully generated." >&2
        fi
esac