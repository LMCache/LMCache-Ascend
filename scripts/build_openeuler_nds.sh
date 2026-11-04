#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build openEuler libnds.so from the kernel tree (openEuler-26.09 branch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${OPENEULER_NDS_SRC:-${ROOT}/third_party/openeuler-kernel-nds/tools/lib/nds/file_p2p}"
KERNEL_ROOT="$(cd "$(dirname "$SRC")/../../../.." && pwd)"
UAPI="${OPENEULER_NDS_UAPI:-${KERNEL_ROOT}/include/uapi}"
OUT="${NDS_BUILD_DIR:-${ROOT}/build/openeuler-nds}"

if [[ ! -f "$SRC/nds_api.c" ]]; then
  echo "Missing $SRC/nds_api.c" >&2
  echo "Clone: git clone --depth 1 -b openEuler-26.09 --filter=blob:none --sparse \\" >&2
  echo "  https://gitcode.com/openeuler/kernel.git third_party/openeuler-kernel-nds" >&2
  echo "  cd third_party/openeuler-kernel-nds && git sparse-checkout set tools/lib/nds include/uapi/linux" >&2
  exit 1
fi

mkdir -p "$OUT"
make -C "$SRC" UAPI_DIR="$UAPI" clean all
install -m 755 "$SRC/libnds.so" "$OUT/libnds.so"
if [[ -f "$SRC/libnds.so.0.1" ]]; then
  install -m 755 "$SRC/libnds.so.0.1" "$OUT/"
fi
echo "Built $OUT/libnds.so"
