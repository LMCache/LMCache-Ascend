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

The environment-based `LMCACHE_GDN_ASYNC_EXPERIMENTAL=1` opt-in and
`LMCACHE_GDN_ASYNC_STAGING_MIB` budget are accepted as migration aliases.
Explicit `extra_config` values take precedence, including `false`. The queue
and `store_async` guards still apply. Restore has a separate opt-in described
in [gdn_async_restore.md](gdn_async_restore.md).

A checkpoint capture reserves a bounded per-rank staging budget and clones every
required recurrent-state plane into a private NPU tensor. A producer event orders
the capture after inference; a device-side wait prevents the producer from
overwriting the source before capture finishes. The CPU store runs on the same
queue as Attention stores, and both count against request completion.

Budget or queue exhaustion skips the optional checkpoint. Capture/transfer
failures are not published as successful stores. If submitted device work cannot
be drained, buffers are retained and shutdown refuses to close the allocator.

## Validation

The source-bound host regressions cover
private clone/config ownership, producer/capture ordering, staging pressure and
Attention/state pending-store accounting. Python compilation and Ruff checks
also pass. Run `python -m unittest discover -s tests/host -v` from the repository
root. These execute production host logic with framework/device doubles.
See [gdn_async_validation.md](gdn_async_validation.md) for local deployment
settings, verified CPU/NPU paths and performance comparisons. The performance
results describe the measured deployment configuration.
