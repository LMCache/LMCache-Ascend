# EIC remote KV backend (`eic://`) on Ascend

LMCache core ships an EIC remote KV connector since [LMCache#1930](https://github.com/LMCache/LMCache/pull/1930): the `EICConnectorAdapter` registers the `eic://` URL scheme and stores/retrieves prefix KV through an EIC cluster. This package makes that connector run on Ascend NPU.

## What this package changes

The upstream `EICConnector.__init__` loads `libcudart.so` to support CUDA GDR. Older core releases did this eagerly, so the connector could not be imported on CANN images that do not ship libcudart. Two layers handle it:

- **Core LMCache#5141** loads the library defensively: when libcudart is absent `cuda_lib` stays `None`, RDMA still constructs, and only an explicit `TRANSPORT_GDR` configuration is rejected. No platform patch is needed on such core.
- **`lmcache_ascend`** keeps a ctypes shim for older core that still loads eagerly. It detects the core capability marker `_LMCACHE_EIC_CUDART_OPTIONAL` and is a no-op on new core, because a truthy CDLL stand-in would also defeat core's `cuda_lib is None` GDR guard. The NPU deployment uses RDMA transport (`eic_trans_type: 2`), where the CUDA GDR path is never entered.

The patch is applied automatically when `lmcache_ascend` is imported, and is a no-op when the connector, its vendor `eic` client package, or the need for it is absent.

No new connector, configuration key, or fork of core code is introduced.

## Requirements

- an EIC cluster reachable over RDMA and the vendor `eic` Python client provided by the EIC runtime image (the package is not published on public PyPI, same as in LMCache core);
- EIC client `flag_file` describing cluster UUID, master addresses, and NICs;
- a local CPU tier enabled (`local_cpu: true`), as required by the connector.

## Configuration

Use the standard LMCache config file pointed to by `LMCACHE_CONFIG_FILE`; see [`examples/eic/lmcache-eic-config.yaml`](../examples/eic/lmcache-eic-config.yaml). Essential keys:

```yaml
remote_url: "eic://MASTER_IP_1:12500;MASTER_IP_2:12500"
eic_instance_id: "YOUR_EIC_CLUSTER_UUID"
eic_trans_type: 2       # RDMA
eic_flag_file: "/path/to/eic_flag_file"
local_cpu: true
```

Then start the engine the same way as any other LMCache-Ascend deployment; the `eic://` backend is selected from `remote_url` by the core adapter discovery.

## Validation

- `tests/v1/storage_backend/test_eic_npu.py` covers the package boundary without a live cluster or libcudart: the patch is a no-op against core that marks `_LMCACHE_EIC_CUDART_OPTIONAL`, it applies an idempotent ctypes proxy on older core, the shim is confined to the cudart lookup and never mutates process-global `ctypes`, and a missing connector/vendor package is swallowed. All `sys.modules` overrides use function-scoped `monkeypatch` so they do not leak into other tests.
- End-to-end correctness and performance must be validated on a real Atlas + EIC cluster.
