set -euo pipefail

# Configuration for offline LMCache RAG benchmark
MODEL_NAME="model/Meta-Llama-3.1-8B-Instruct"
# [CacheBlend musique_s.json](https://github.com/YaoJiayi/CacheBlend/blob/main/inputs/musique_s.json)
DATASET_PATH="musique_s.json"
PROMPT_BUILD_METHOD=QA
MAX_TOKENS=32
END=10 # process 10 examples
TEMPERATURE=0.0

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS_DIR="$SCRIPT_DIR/offline_benchmark_results"
mkdir -p "$RESULTS_DIR"

SCENARIOS=(
    "LMCache (No Blend)" "./config/lmcache.yaml" "lmcache_offline"
    "CacheBlend" "./config/lmcache_blend.yaml" "cacheblend_offline"
)

DATASET_NAME=$(echo $DATASET_PATH | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')

declare -a ACTUAL_CSV_PATHS=()

run_benchmark_scenario() {
    local scenario_name="$1"
    local config_file="$2"
    local output_suffix="$3"
    local output_file="$RESULTS_DIR/${DATASET_NAME}_${output_suffix}.csv"

    echo -e "\n========================================" >&2
    echo "Scenario:$scenario_name" >&2
    echo "Config:$config_file" >&2
    echo "Output_file:$output_file" >&2
    echo "========================================" >&2
    
    if [[ ! -f "$config_file" ]]; then
        echo "Config file not found: $config_file" >&2
        return 1
    fi

    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "Dataset file not found: $DATASET_PATH" >&2
        return 1
    fi

    echo "Starting benchmark with config: $config_file" >&2
    # Run the unified benchmark (precompute + RAG in same LLM instance)
    # Note: --online flag is NOT used, so this runs in offline mode
    if LMCACHE_CONFIG_FILE="$config_file" python3 rag.py \
        --start-index 0 \
        --end-index $END \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_PATH" \
        --prompt-build-method $PROMPT_BUILD_METHOD \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --output "$output_file" \
        --verbose >&2; then

        if [[ -f "$output_file" ]]; then
            echo "Benchmark completed successfully" >&2
            echo "\n [$scenario_name] done, Output file: $output_file" >&2
            echo "$output_file"
            return 0
        else
            echo "Output file was not created: $output_file" >&2
            return 1
        fi
    else:
        echo "\n [$scenario_name] failed" >&2
    fi
}

echo "===== Offline LMCache RAG Benchmark Configuration =====" >&2
echo "Model:$MODEL_NAME" >&2
echo "Dataset:$DATASET_PATH" >&2
echo "Sample range:0-$END" >&2
echo "========================================" >&2

CSV_PATHS=()
for ((i=0; i<${#SCENARIOS[@]}; i+=3)); do
    SCENARIO_NAME=${SCENARIOS[$i]}
    CONFIG_FILE=${SCENARIOS[$i+1]}
    OUTPUT_SUFFIX=${SCENARIOS[$i+2]}

    csv_path=$(run_benchmark_scenario "$SCENARIO_NAME" "$CONFIG_FILE" "$OUTPUT_SUFFIX")
    CSV_PATHS+=("$csv_path")
done

echo "======================================================" >&2
echo "      ALL BENCHMARKS COMPLETED" >&2
echo "======================================================" >&2
echo "save file to:" >&2
for ((i=0; i<${#SCENARIOS[@]}; i+=3)); do
    OUTPUT_SUFFIX=${SCENARIOS[$i+2]}
    echo "  - ${DATASET_NAME}_${OUTPUT_SUFFIX}.csv" >&2
done
echo "========================================" >&2

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