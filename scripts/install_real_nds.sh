#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install vendor libdpc_nds.so to the canonical Ascend path on the HOST.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sudo $0 /path/to/libdpc_nds.so" >&2
  echo "   or: sudo $0 /path/to/OceanStor_DataTurbo_*_Linux.zip" >&2
  exit 1
fi

SRC="$1"
DEST_DIR="/opt/oceanstor/dataturbo/sdk/lib"
DEST="${DEST_DIR}/libdpc_nds.so"

if [[ -f "$SRC" && "$SRC" == *.zip ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  unzip -q "$SRC" -d "$WORK"
  FOUND="$(find "$WORK" -name 'libdpc_nds.so' | head -1)"
  if [[ -z "$FOUND" ]]; then
    echo "No libdpc_nds.so inside zip; run the package install.sh first." >&2
    exit 1
  fi
  SRC="$FOUND"
fi

if [[ ! -f "$SRC" ]]; then
  echo "Not a file: $SRC" >&2
  exit 1
fi

if strings "$SRC" | grep -q 'nds_mock.c'; then
  echo "Refusing to install LMCache mock as vendor NDS." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
install -m 444 "$SRC" "$DEST"
echo "Installed $DEST"
nm -D "$DEST" | grep -E 'NdsFile(Driver|Handle|Read|Write)' || true
echo "Next: mount DPC, bind-mount $DEST into training containers, set NDS_TEST_DIR on DPC."
