# GDN multimodal native-path regression

This fix bypasses external GDN checkpoint caching for the entire lifetime of a
multimodal request. It does not disable the model's image support, or bypass
native vLLM inference. Text requests still use the existing hybrid cache path.

## Host reproduction and regression

```bash
python -m unittest discover -s tests/host -p test_gdn_mm_bypass.py -v
```

Nine source-bound tests exercise the scheduler and worker methods with explicit
framework doubles. In particular, the finish-only frame must carry native image
IDs before scheduler bookkeeping is removed. Neither TP worker may turn those
IDs into an external send/receive acknowledgement, even on repeated polling.
They also check resumed records without `mm_features`, both cached-record
formats, absence of load/save/state metadata, and normal text lookup behavior.
The old integration test expecting a text-only exception is updated to expect
native bypass instead; it needs the ordinary integration-test dependencies.

## Existing device smoke and minimal device follow-up

The reporter's patched deployment has already completed a real image request:
an in-memory PNG with a red left half and blue right half was correctly described,
the response completed with HTTP 200, three subsequent health checks returned
200, and a subsequent text request completed normally. That is useful device
evidence and is not invalidated by adding automated regressions.

It is a smoke result for the local deployment, not a proof of all fault paths or
of this extracted upstream commit. On a dedicated supported GDN instance, a
minimal follow-up for this exact diff is:

1. Record the tested commit and package versions; enable the connector.
2. Send the same image request and retain its request ID. Confirm
   `LMCACHE_MM_BYPASS` for that ID and a correct, complete response.
3. Confirm health and subsequent text generation, and verify the service was not
   silently restarted. Repeat with an image and text request in the same batch.
4. Confirm native image completions never query external pending-store state;
   text cache lookup/store remains active.

Do not run cache-reset tests on a shared production service. A passing smoke
does not guarantee survival of arbitrary driver failures or container OOM.
