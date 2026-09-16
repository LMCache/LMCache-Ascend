# EIC remote KV backend (`eic://`) on Ascend

LMCache core ships an EIC remote KV connector since [LMCache#1930](https://github.com/LMCache/LMCache/pull/1930): the `EICConnectorAdapter` registers the `eic://` URL scheme and stores/retrieves prefix KV through an EIC cluster. This package makes that connector run on Ascend NPU.

## What this package changes

The upstream `EICConnector.__init__` eagerly calls `ctypes.CDLL("libcudart.so")` to support CUDA GDR. CANN images do not ship libcudart, so importing the connector fails before the adapter can be selected. `lmcache_ascend` patches the connector at import time:

- when libcudart is absent, the CUDA binding becomes a no-op shim and the connector initializes normally;
- the NPU deployment uses RDMA transport (`eic_trans_type: 2`), where the CUDA GDR path is never entered.

The patch is applied automatically when `lmcache_ascend` is imported, and is a no-op when the connector or its vendor `eic` client package is absent.

No new connector, configuration key, or fork of core code is introduced.

## Requirements

- an EIC cluster reachable over RDMA and the vendor `eic` Python client provided by the EIC runtime image (the package is not published on public PyPI, same as in LMCache core);
- EIC client `flag_file` describing cluster UUID, master addresses, and NICs;
- a local CPU tier enabled (`local_cpu: true`), as required by the connector.

## Configuration

Use the standard LMCache config file pointed to by `LMCACHE_CONFIG_FILE`; see [`examples/eic/lmcache-eic-config.yaml`](../../examples/eic/lmcache-eic-config.yaml). Essential keys:

```yaml
remote_url: "eic://MASTER_IP_1:12500;MASTER_IP_2:12500"
eic_instance_id: "YOUR_EIC_CLUSTER_UUID"
eic_trans_type: 2       # RDMA
eic_flag_file: "/path/to/eic_flag_file"
local_cpu: true
```

Then start the engine the same way as any other LMCache-Ascend deployment; the `eic://` backend is selected from `remote_url` by the core adapter discovery.

## Validation

- `tests/v1/storage_backend/test_eic_npu.py` covers import/init compatibility without libcudart, idempotent patching, and graceful skip when the `eic` package is unavailable. It stubs the vendor client and does not need a live cluster.
- End-to-end correctness and performance must be validated on a real Atlas + EIC cluster.
