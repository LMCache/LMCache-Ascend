# MP Mode (Multiprocess LMCache) on Ascend NPU

MP mode is the multiprocess deployment architecture of LMCache: the KV cache is owned by a standalone `lmcache server` process, and vLLM attaches to it through the `LMCacheMPConnector`. The server owns L1/L2 storage, eviction, and lookup; the engine talks to it over the MP protocol (ZMQ control plane + shared-memory / device-handle data plane). Compared with running LMCache inside the vLLM process, this keeps cache work (store, retrieve, chunking, eviction) out of the engine loop and allows one cache to be shared by several engines.

The data plane has two transfer modes, selected with the `lmcache.mp.mp_transfer_mode` key in `kv_connector_extra_config`:

| | `lmcache_driven` | `engine_driven` |
| :--- | :--- | :--- |
| Data path | The LMCache server pulls/pushes KV through device IPC handles | The vLLM workers gather/scatter the KV copies themselves, zero-copy from a named shared-memory pool |
| KV cache formats | Standard GQA/MHA plus MLA/DSA multi-plane layouts (incl. DeepSeek-V4 `(latent, scale)`); not Mamba/GDN hybrids — see [limitations](#supported-models-and-limitations) | Single KV-cache format models only (standard GQA/MHA, e.g. Qwen3); not Mamba/GDN |
| Server requirements | `--chunk-size 4096` for DeepSeek-V4 (see [lmcache_driven.md](lmcache_driven.md)) | `--shm-name <name> --no-l1-use-lazy` are mandatory (see [engine_driven.md](engine_driven.md)) |
| Deployment guide | [lmcache_driven.md](lmcache_driven.md) | [engine_driven.md](engine_driven.md) |

On NPU, leaving `lmcache.mp.mp_transfer_mode` unset (or `auto`) resolves to `engine_driven`; for DeepSeek-V4 you must set `lmcache_driven` explicitly.

## Supported models and limitations

**Supported in MP mode (both transfer modes, where the format applies):**

- Standard dense GQA/MHA models (for example Qwen3).
- Multi-plane MLA/DSA packed KV layouts (for example DeepSeek-V4) when using **`lmcache_driven`**.

In the table above, “hybrid KV cache management formats” means **MLA/DSA multi-plane attention KV**, not SSM/Mamba hybrids.

**Not supported in MP mode (current release):**

- **Mamba / GDN / Qwen3.5-style hybrids** — models whose vLLM KV cache includes `MambaSpec` groups with reusable state snapshots (`mamba_cache_mode` `align` or `all`, for example Qwen3.5-27B). LMCache MP rejects these KV cache groups at startup. Use the in-process connector (`LMCacheAscendConnectorV1Dynamic`) for SP deployments, or wait for a future MP release that adds Mamba/GDN support.

## Documents

- [lmcache_driven.md](lmcache_driven.md) — the `lmcache_driven` transfer mode full deployment guide: environment preparation, container startup, source installation, server/vLLM startup, and verification (DeepSeek-V4 example).
- [engine_driven.md](engine_driven.md) — deploy and run the `engine_driven` transfer mode (Qwen3 example).

## Relationship to the LMCache dev branch

The MP architecture (the `lmcache server` CLI, the `LMCacheMPConnector`, and the transfer-mode layer) is developed on the LMCache `dev` branch — the main line — and the matching Ascend backends (NPU platform routing, block-transfer ops, hccl/hixl bindings) live in the LMCache-Ascend repository. Install both from the official repositories only — see the installation section of [lmcache_driven.md](lmcache_driven.md) for the exact procedure. The classic in-process connector (`LMCacheAscendConnectorV1Dynamic`) is a separate integration that uses its own LMCache release line; its installation must not be mixed with the MP-mode checkout in one Python environment.
