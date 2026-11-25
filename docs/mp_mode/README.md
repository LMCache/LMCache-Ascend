# MP Mode (Multiprocess LMCache) on Ascend NPU

MP mode is the multiprocess deployment architecture of LMCache: the KV cache is owned by a standalone `lmcache server` process, and vLLM attaches to it through the `LMCacheMPConnector`. The server owns L1/L2 storage, eviction, and lookup; the engine talks to it over the MP protocol (ZMQ control plane + shared-memory / device-handle data plane). Compared with running LMCache inside the vLLM process, this keeps cache work (store, retrieve, chunking, eviction) out of the engine loop and allows one cache to be shared by several engines.

The data plane has two transfer modes, selected with the `lmcache.mp.mp_transfer_mode` key in `kv_connector_extra_config`:

| | `lmcache_driven` | `engine_driven` |
| :--- | :--- | :--- |
| Data path | The LMCache server pulls/pushes KV through device IPC handles | The vLLM workers gather/scatter the KV copies themselves, zero-copy from a named shared-memory pool |
| KV cache formats | Standard GQA/MHA, MLA/DSA multi-plane layouts (incl. DeepSeek-V4 `(latent, scale)`), and hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8; [qwen3_5.md](qwen3_5.md)) | Single KV-cache format models only (standard GQA/MHA, e.g. Qwen3); not Mamba/GDN or MLA/DSA |
| Server requirements | `--chunk-size 4096` for DeepSeek-V4 (see [lmcache_driven.md](lmcache_driven.md)) | `--shm-name <name> --no-l1-use-lazy` are mandatory (see [engine_driven.md](engine_driven.md)) |
| Deployment guide | [lmcache_driven.md](lmcache_driven.md) | [engine_driven.md](engine_driven.md) |

On NPU, leaving `lmcache.mp.mp_transfer_mode` unset (or `auto`) resolves to `lmcache_driven`, so DeepSeek-V4 needs no extra setting; to use `engine_driven` you must set it explicitly.

## Supported models and limitations

**Supported in MP mode (both transfer modes, where the format applies):**

- Standard dense GQA/MHA models (for example Qwen3).
- Multi-plane MLA/DSA packed KV layouts (for example DeepSeek-V4) when using **`lmcache_driven`**.
- Hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8, architecture `Qwen3_5ForConditionalGeneration`) when using **`lmcache_driven`**. The flags that match the LMCache recipe, and the Ascend differences, are in [qwen3_5.md](qwen3_5.md).

In the table above, “hybrid KV cache management formats” means **MLA/DSA multi-plane attention KV**, not SSM/Mamba hybrids.

**Not supported in MP mode (current release):**

- Hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8) in **`engine_driven`**. Use `lmcache_driven`.
- `--mamba-cache-mode all`. These models require `align`.

## Documents

- [lmcache_driven.md](lmcache_driven.md) — the `lmcache_driven` transfer mode full deployment guide: environment preparation, container startup, source installation, server/vLLM startup, and verification (DeepSeek-V4 example).
- [engine_driven.md](engine_driven.md) — deploy and run the `engine_driven` transfer mode (Qwen3 example).
- [qwen3_5.md](qwen3_5.md) — hybrid Mamba models (for example Qwen3.5, Qwen3.6, and Qwen3.8) on Ascend: same flags as the [LMCache recipe](https://docs.lmcache.ai/recipes/qwen3_5.html), plus the Ascend differences.

## Relationship to the LMCache dev branch

The MP architecture (the `lmcache server` CLI, the `LMCacheMPConnector`, and the transfer-mode layer) is developed on the LMCache `dev` branch — the main line — and the matching Ascend backends (NPU platform routing, block-transfer ops, hccl/hixl bindings) live in the LMCache-Ascend repository. Install both from the official repositories only — see the installation section of [lmcache_driven.md](lmcache_driven.md) for the exact procedure. The classic in-process connector (`LMCacheAscendConnectorV1Dynamic`) is a separate integration that uses its own LMCache release line; its installation must not be mixed with the MP-mode checkout in one Python environment.

## Experimental Qwen3 selective blend controls

`MPSelectiveBlendConnector` supports dense, unquantized Qwen3 with NHD KV,
eager execution, synchronous scheduling, one sequence, and APC disabled.
Its `kv_connector_extra_config` accepts:

| Key | Default | Effect |
| --- | --- | --- |
| `lmcache.mp.blend_selection_metric` | `"k"` | Rank cached tokens by squared FP32 K deviation; `"v"` ranks by V deviation, matching the author's public selection metric. |
| `lmcache.mp.blend_refresh_check_layer` | `false` | If true, keep all freshly projected KV at the check layer; later layers still update only selected positions. |

These controls allow separate ablations of token selection and check-layer
updates. They do not implement the paper's layerwise I/O pipeline or establish
quality guarantees for Qwen3. Both preserve mandatory uncached positions and
the scheduler-confirmed prefix.

Document precomputation must avoid recursively blending earlier documents.
With the accompanying LMCache connector support, pass request-level
`kv_transfer_params={"lmcache.skip_load": true}` during precomputation to
bypass cache lookup and compute fresh KV while retaining normal cache stores.
