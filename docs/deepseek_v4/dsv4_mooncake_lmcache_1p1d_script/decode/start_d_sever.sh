#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

python3 "${SCRIPT_DIR}/launch_online_dp.py" \
    --dp-size 16 \
    --tp-size 1 \
    --dp-size-local 16 \
    --dp-rank-start 0 \
    --device-start 0 \
    --dp-address 7.246.92.165 \
    --dp-rpc-port 12322 \
    --vllm-start-port 7100 \
    "$@"
