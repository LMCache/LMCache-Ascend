# Experimental GDN asynchronous checkpoint stores

This is an opt-in extension of the existing asynchronous Attention store queue.
It does not enable asynchronous GDN lookup, receive, or vLLM scheduling.

Set `store_async: true`, a bounded `store_async_max_queue_size` (1..32), and:
```yaml
extra_config:
  gdn_async_store: true
  gdn_async_staging_mib: 256
```

These settings apply to all cache engines in the configured instance, so this
opt-in is intended for a dedicated supported GDN deployment. The local CPU tier
is the only asynchronous checkpoint destination; other GDN compatibility guards
remain in force. Synchronous mode remains the default.

A checkpoint capture reserves a bounded per-rank staging budget and clones every
required recurrent-state plane into a private NPU tensor. A producer event orders
the capture after inference; a device-side wait prevents the producer from
overwriting the source before capture finishes. The CPU store runs on the same
queue as Attention stores, and both count against request completion.

Budget or queue exhaustion skips the optional checkpoint. Capture/transfer
failures are not published as successful stores. If submitted device work cannot
be drained, buffers are retained and shutdown refuses to close the allocator.
This does not promise recovery from device faults or process/container OOM.

## Draft validation status

The upstream-native extraction has passed Python syntax compilation only.
NPU runtime, cancellation/preemption, queue-pressure, shutdown and failure-path
tests are still required before this should be considered for merging.
Performance results from the combined local prototype are not an isolated
measurement of this save-side change.
