#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

python3 "${SCRIPT_DIR}/launch_online_dp.py" \
    --dp-size 8 \
    --tp-size 2 \
    --dp-size-local 8 \
    --dp-rank-start 0 \
    --device-start 0 \
    --dp-address 7.246.92.163 \
    --dp-rpc-port 12321 \
    --vllm-start-port 7100 \
    "$@"
