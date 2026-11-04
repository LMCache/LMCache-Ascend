# Install real Huawei NDS (not the dev mock)

Official NDS is **not** a generic “read NVMe from ext4” API. It is part of
**OceanStor DataTurbo** with the **DPC** client filesystem and
`libdpc_nds.so` on the **training node**.

References:

- [MindCluster: DPC + NDS checkpoint read](https://www.hiascend.com/doc_center/source/en/mindcluster/730/clustersched/schedulingug/mindioacp047.html)
- OceanStor DataTurbo 25.x DTFS User Guide (Huawei EDOC1100539415)
- [08-ascend-nds.md](../../08-ascend-nds.md) in this repo

## What you need from storage / platform

1. **OceanStor DataTurbo client** (or AI Storage Kit) for Linux aarch64 on this
   training node — RPM/zip from Huawei support, not public download.
2. **`/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so`** after client install,
   file mode **444** (MindIO requirement).
3. **DPC filesystem mounted** on the node (e.g. `/mnt/dpc`). LMCache slab and
   tests must use paths **on that mount**, not on local ext4 such as
   `/data1` unless that directory is the DPC mount point.
4. **NDS 1.0** enabled on the storage side for that namespace/volume.

This lab’s marco container currently has **`/data1` = ext4 on `/dev/nvme0n1`**.
That is normal block storage. It is **not** a substitute for DPC + NDS until
the DataTurbo client and DPC mount exist.

## Host install (training node)

From the DataTurbo KV/client package (example from Huawei docs):

```bash
unzip OceanStor_DataTurbo_for_kv_*_Linux.zip
cd OceanStor_DataTurbo_for_kv_*_Linux
sudo sh install.sh
# or: sudo CUSTOM_USER=dataturbo sh install.sh
```

Verify:

```bash
test -f /opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so
sudo chmod 444 /opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so
nm -D /opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so | grep NdsFile
```

Mount DPC per the DTFS guide (path is site-specific), then:

```bash
export NDS_TEST_DIR=/mnt/dpc/lmcache_nds_scratch
mkdir -p "$NDS_TEST_DIR"
```

Run PR 0 probe on the host or in container:

```bash
python tools/nds_probe/nds_probe.py \
  --library /opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so \
  --test-dir "$NDS_TEST_DIR"
```

Success log string: `initial and open nds file driver success` (from MindIO/NDS).

## Marco MP container (`marco-dsv4-mp-tpot`)

Bind-mount the **host file** (do not copy the mock into the image path):

```text
/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so -> same path in container (ro)
/mnt/dpc -> /mnt/dpc   # or your DPC mount
```

Huawei MindIO Docker doc: bind the `.so` at the **exact** path
`/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so`.

Recreate or update the container with those volumes, then:

```bash
export NDS_TEST_DIR=/mnt/dpc/lmcache_nds_scratch
export NDS_LIBRARY_PATH=/opt/oceanstor/dataturbo/sdk/lib/libdpc_nds.so
export NDS_ALLOW_MOCK=0
bash scripts/run_nds_hw_tests_marco.sh
```

## Remove the development mock

If a previous run installed `build/libdpc_nds_mock.so` as `libdpc_nds.so`:

```bash
sudo scripts/remove_nds_mock.sh
```

Only install the vendor `.so` with `scripts/install_real_nds.sh`.
