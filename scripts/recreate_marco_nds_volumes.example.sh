#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Example volume flags for real NDS — adjust image/name/command to match your site.
# Do NOT run blindly; merge with your existing marco-dsv4-mp-tpot docker run.
cat <<'EOF'
# After host install of DataTurbo + DPC mount at /mnt/dpc:

  -v /opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so:/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so:ro
  -v /mnt/dpc:/mnt/dpc

# Environment inside container:
  export NDS_LIBRARY_PATH=/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so
  export NDS_TEST_DIR=/mnt/dpc/lmcache_nds_scratch
  export NDS_ALLOW_MOCK=0
EOF
