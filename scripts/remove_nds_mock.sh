#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
DEST="/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so"
if [[ ! -f "$DEST" ]]; then
  echo "No $DEST"
  exit 0
fi
if strings "$DEST" 2>/dev/null | grep -q 'nds_mock.c'; then
  rm -f "$DEST"
  echo "Removed mock $DEST"
else
  echo "Leaving vendor library in place (not mock): $DEST" >&2
  exit 1
fi
