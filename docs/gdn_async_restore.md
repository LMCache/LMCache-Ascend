# Experimental GDN asynchronous lookup and restore

Builds on the multimodal-bypass and snapshot-store changes. The default remains
synchronous. The receive protocol is selected only with the explicit GDN opt-in:

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
   retain the bypass lifecycle from the preceding change.

A receive retains ownership through cancellation. Preemption cannot recycle its
targets while the ledger owns them. Shutdown drains the worker and refuses to
close storage if an operation remains unsafe. Shared transfer scratch is
serialized and fenced on the background threads: asynchronous operation does
not mean every H2D and D2H can overlap.

## Draft limitations and required validation

This upstream-native extraction has passed Python syntax compilation only.
The pinned local prototype's measurements are not runtime validation of this
diff. Maintainers must confirm the LMCache/vLLM version interfaces, manager and
factory integration, receive-only/zero-forward metadata, and checkpoint lease
lifecycle before merge. Add automated tests before removing Draft status.

Required coverage includes all-rank correctness, matching cold/restored outputs,
MTP and graph combinations, cancellation before/after admission, preemption,
queue pressure, timeout, shutdown, mixed image/text/restore traffic and failures.
No speedup is promised from a particular concurrency.

Failures do not generate successful receive acknowledgements or allow partially
restored recurrent state into model forward. Unknown device errors remain
fail-closed. This proposal does not guarantee transparent recomputation or
survival of the same engine process for every fault, and does not import the
prototype's broader availability/circuit-breaker policies.
