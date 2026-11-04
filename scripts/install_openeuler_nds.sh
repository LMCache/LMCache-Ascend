#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
LIB="${1:-build/openeuler-nds/libnds.so}"
DEST="${NDS_INSTALL_LIB:-/usr/local/lib/libnds.so}"
if [[ ! -f "$LIB" ]]; then
  echo "Run scripts/build_openeuler_nds.sh first" >&2
  exit 1
fi
sudo install -d "$(dirname "$DEST")"
sudo install -m 755 "$LIB" "$DEST"
echo "Installed $DEST"
echo "Requires kernel module exposing /dev/p2p_device (openEuler-26.09 drivers/nds)."
