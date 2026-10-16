# Experimental GDN asynchronous lookup and restore

GDN lookup and restore can run with snapshot-backed stores and vLLM asynchronous
scheduling. The default remains synchronous; opt in with:

```yaml
store_async: true
store_async_max_queue_size: 8
enable_async_loading: true
extra_config:
  gdn_async_store: true
  gdn_async_staging_mib: 256
  gdn_async_restore: true
```

Use the existing vLLM asynchronous scheduler for this mode. Ordinary KV clients
continue to use the existing factory behavior when the GDN opt-in is absent.
This path requires the existing supported GDN layout, all TP workers and local
CPU storage; it does not enable layerwise mode, CacheBlend or remote tiers.

For environment-based deployment, the same flags can also be supplied
as `LMCACHE_GDN_ASYNC_EXPERIMENTAL=1` (store),
`LMCACHE_GDN_FULL_ASYNC_EXPERIMENTAL=1` (restore), and
`LMCACHE_GDN_ASYNC_STAGING_MIB=256`. An explicitly supplied `extra_config` value
wins, including `false`. The ordinary `store_async` and `enable_async_loading`
options must still be enabled; the aliases alone do not bypass compatibility
guards. See [local validation](gdn_async_validation.md) for deployment settings,
source integration and device measurements.

## Protocol

1. The scheduler submits and polls the existing all-rank GDN selection operation.
   The actual RPC client, token database and transport are owned by one thread.
   A pending lookup is not a cache miss. Cancellation is serialized on that
   same thread.
2. On external admission, construct receive-only metadata from the real allocated
   block tables. Receiving requests have no forward tokens in that step.
3. A worker-owned load queue obtains the selected Attention objects and recurrent
   checkpoint lease, orders H2D after target-block initialization, and copies all
   required groups.
4. After transfer fencing and source cleanup, the completion ledger makes a
   receive acknowledgement available. The normal vLLM all-rank protocol controls
   admission to the next forward.
5. Send and receive acknowledgements are separate. Native multimodal requests
   remain outside external GDN cache transfers throughout their lifecycle.

A receive retains ownership through cancellation. Preemption cannot recycle its
targets while the ledger owns them. Shutdown drains the worker and refuses to
close storage if an operation remains unsafe. Shared transfer scratch is
serialized and fenced on the background threads: asynchronous operation does
not mean every H2D and D2H can overlap.

## Validation

This implementation has 30 passing source-bound host regressions
(9 multimodal plus 21 async), Python compilation and Ruff checks. Run them with:

```bash
python -m unittest discover -s tests/host -v
```

Host tests execute production host logic with explicit framework/device doubles.
They cover manager/factory delegation, lookup ownership, receive completion,
snapshot/store accounting and failure fencing. Local deployment verification
covered CPU restore and NPU prefix reuse at concurrency 1/2/4. The configuration,
per-rank completion evidence and performance results are documented in
[gdn_async_validation.md](gdn_async_validation.md).
