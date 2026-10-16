# GDN multimodal native-path regression

## Problem

An image request with GDN hybrid caching can cause an LMCache connector error
to propagate into vLLM EngineCore and terminate the model service. The local
failure included `EngineDeadError` and subsequent HTTP 503 health responses;
the impact was not limited to rejecting one image request.

The GDN lookup path rejects multimodal requests with
`ValueError("GDN checkpoints support only text requests")`. Native image
requests also must not produce external-transfer completion notifications when
they did not offload anything. The fix keeps the entire request on the native
vLLM multimodal path, including chunking, resumed records and completion.
Text requests continue to use the hybrid cache path.

## Local deployment configuration

| Setting | Value |
| --- | --- |
| Python | 3.12 |
| vLLM | 0.23 RC environment |
| LMCache core | 0.4.5 |
| Model | Qwen-family 27B GDN hybrid multimodal model |
| Devices / tensor parallelism | 2 Ascend NPUs / TP=2 |
| CPU cache | 32 GiB per rank |
| LMCache chunk size | 256 tokens |
| Native prefix cache | Enabled |
| Chunked prefill / step token budget | Enabled / 8,192 tokens |
| Hybrid cache | Enabled, `mamba-cache-mode=align` |
| Speculative decoding | MTP, 3 speculative tokens |
| Graph mode | `FULL_DECODE_ONLY` |
| Image-smoke scheduling | vLLM asynchronous scheduling |

## Local deployment validation

The local deployed implementation completed the image smoke successfully:

1. Generate an in-memory 320 x 160 PNG: red left half, blue right half.
2. Send it through the chat-completions API and confirm the response correctly
   identifies both colors and finishes with HTTP 200.
3. Confirm three subsequent health probes return HTTP 200.
4. Confirm a subsequent text-generation request completes normally.
5. In the instrumented run, confirm API/TP process identities remain unchanged.

`LMCACHE_MM_BYPASS` identifies the native image path. No image-specific external
lookup/load/store or synthetic send completion is needed.

## Automated source regression

Nine tests passed on Python 3.12 against production scheduler/worker methods
with framework and device doubles. Coverage includes sticky bypass after
`mm_features` disappears, new/resumed/cached records, finish-only frames,
repeated completion polling on both TP ranks, no external ownership for native
images, and unchanged text/non-GDN behavior.

```bash
python -m unittest discover -s tests/host -p test_gdn_mm_bypass.py -v
```

Python compilation and Ruff lint/format checks passed. The host CI workflow
runs the same dependency-free regressions on Python 3.10 and 3.12. The normal
integration test now expects native bypass instead of multimodal rejection.
