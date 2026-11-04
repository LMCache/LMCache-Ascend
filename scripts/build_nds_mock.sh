#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
OUT="${1:-$ROOT/build/libdpc_nds_mock.so}"
mkdir -p "$(dirname "$OUT")"
gcc -shared -fPIC -O2 \
  -I"$CANN/include" \
  "$ROOT/csrc/nds_mock/nds_mock.c" \
  -o "$OUT" \
  -L"$CANN/lib64" -lascendcl \
  -Wl,-rpath,"$CANN/lib64"
chmod 444 "$OUT"
echo "Built mock NDS library: $OUT"
