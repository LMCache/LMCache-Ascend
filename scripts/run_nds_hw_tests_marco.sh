#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run NDS hardware/component tests in marco-dsv4-mp-tpot (openEuler or DataTurbo).
set -euo pipefail
CONTAINER="${NDS_MARCO_CONTAINER:-marco-dsv4-mp-tpot}"
NDS_STACK="${NDS_STACK:-openeuler}"
ALLOW_MOCK="${NDS_ALLOW_MOCK:-0}"

if [[ "$NDS_STACK" == "dataturbo" ]]; then
  NDS_LIB="${NDS_LIBRARY_PATH:-/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so}"
  SCRATCH="${NDS_TEST_DIR:-/mnt/dpc/lmcache_nds_scratch}"
else
  NDS_LIB="${NDS_LIBRARY_PATH:-/usr/local/lib/libnds.so}"
  SCRATCH="${NDS_TEST_DIR:-/data1/m00926961/nds_scratch}"
fi

if [[ "$ALLOW_MOCK" != "1" ]]; then
  if [[ ! -f "$NDS_LIB" ]]; then
    echo "Missing NDS library ($NDS_STACK): $NDS_LIB" >&2
    if [[ "$NDS_STACK" == "dataturbo" ]]; then
      echo "Install: sudo scripts/install_real_nds.sh /path/to/libdpc_nds.so" >&2
    else
      echo "Build: bash scripts/build_openeuler_nds.sh && sudo scripts/install_openeuler_nds.sh" >&2
      echo "See ../09-openeuler-nds-local-nvme.md" >&2
    fi
    exit 1
  fi
  if strings "$NDS_LIB" 2>/dev/null | grep -q 'nds_mock.c'; then
    echo "Refusing mock at $NDS_LIB." >&2
    exit 1
  fi
fi

docker exec "$CONTAINER" bash -lc "
set -euo pipefail
if [[ ! -f '$NDS_LIB' ]]; then
  echo 'Inside container: bind-mount $NDS_LIB' >&2
  exit 1
fi
if [[ '$NDS_STACK' == 'openeuler' ]] && [[ ! -c /dev/p2p_device ]]; then
  echo 'Missing /dev/p2p_device — load openEuler drivers/nds kernel module on the host and bind-mount the device node.' >&2
  exit 1
fi
mkdir -p '$SCRATCH'
export NDS_STACK='$NDS_STACK'
export PYTHONPATH=/workspace/LMCache-Ascend:/workspace/LMCache
export NDS_TEST_DIR='$SCRATCH'
export NDS_LIBRARY_PATH='$NDS_LIB'
source /usr/local/Ascend/cann-9.1.0/set_env.sh 2>/dev/null || true
cd /workspace/LMCache-Ascend
pytest tests/v1/gds/test_nds_direct_read.py -q --noconftest
cd /workspace/LMCache
pytest tests/v1/gpu_connector/gds_backends/test_nds.py::test_nds_handle_reads_slab_without_lmcache -q --noconftest
pytest tests/v1/gpu_connector/gds_backends/test_nds.py::test_nds_handle_roundtrip_without_lmcache -q --noconftest
"
