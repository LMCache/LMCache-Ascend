# Hole Feature: Maintainer Overview

## 1. What this feature does

Standard CacheBlend assumes the reusable portion of a request is a contiguous cached prefix. Once the first cache miss is encountered, the remainder of the prompt is treated as uncached and recomputed. Hole-mode extends that model to handle non-contiguous reuse: prompts where the cached prefix has one or more gaps ("holes") but later segments are still reusable.

The motivating case is a prompt whose prefix is mostly cached but has a missing middle segment. In a multi-document RAG prompt, for example, one retrieved document may have been evicted or never cached while surrounding documents are still present. Standard CacheBlend stops at the first miss and re-prefills everything from that point onward. Hole-mode computes fresh K/V for the missing segment and still reuses cached segments after it.

Operationally, the feature preserves the same high-level CacheBlend goal: reuse cached K/V, recompute only what must be refreshed, and resume normal vLLM execution after the covered prefix. The difference is that the reusable prefix is no longer modeled as a single contiguous span. Hole-mode carries a segment-aware lookup result through scheduler, worker, blender, and save paths so that reuse can continue across gaps.

## 2. Integration approach

The feature follows upstream LMCache extension points where upstream already provides them.

- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole` inherits directly from upstream `LMCacheConnectorV1Impl`.
- `lmcache_ascend/integration/vllm/lmcache_ascend_hole_connector_v1.py:LMCacheAscendHoleConnectorV1Dynamic` provides the dynamic connector entry point in the same pattern as upstream connector modules.
- `lmcache_ascend/__init__.py` patches `VllmServiceFactory.get_or_create_lmcache_engine`, `VllmServiceFactory.maybe_create_lookup_client`, and `VllmServiceFactory.maybe_create_lookup_server` so that the existing manager/factory flow can construct hole-aware services when the hole connector module is selected.
- Connector entry points remain interface-compatible with upstream: `LMCacheConnectorV1ImplHole.get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`, `start_load_kv`, `save_kv_layer`, `wait_for_save`, and `request_finished` use the same roles and call sites as the standard connector.

Where upstream does not have a hook for non-contiguous reuse semantics, the feature introduces parallel structure rather than trying to force the behavior into a contiguous-prefix abstraction. That choice is deliberate. The feature needs to represent covered ranges, hole positions, prefix-miss regions, and hole-aware save behavior in a way that upstream contiguous-prefix CacheBlend does not model.

## 3. New types

The hole-specific type surface is small but fundamental.

- `lmcache_ascend/v1/hole_types.py:HoleLookupResult`
  Carries the result of a hole-aware lookup. Instead of a single "matched prefix length", it describes the reusable covered prefix, hit ranges, prefix-miss ranges, and the mode used to handle the request.

- `lmcache_ascend/v1/hole_types.py:HoleLoadSpec`
  Scheduler-to-worker load specification for a hole request. It carries the hole-aware state needed by the worker-side connector and blender to materialize the covered region correctly.

- `lmcache_ascend/v1/hole_types.py:HoleSaveSpec`
  Hole-aware save semantics for the prefix-miss region. This lets the worker save exactly the recomputed portions that should become reusable on later requests.

- `lmcache_ascend/v1/hole_segment_utils.py:HoleSegmentHelper`
  Segment-aware helper for separator-delimited prompts. It splits ranges, maps lookup results onto semantic segments, and derives the hole-aware lookup result from segment hit data.

Together, these types replace the standard contiguous-prefix mental model with a segment-aware one while keeping the rest of the connector lifecycle recognizable.

## 4. Request flow compared to standard CacheBlend

### Connector construction

Construction still starts in the standard upstream places: the dynamic connector module instantiates `LMCacheConnectorV1ImplHole`, which participates in the usual connector -> service factory -> manager setup. The hole-specific divergence is that the factory patch in `lmcache_ascend/__init__.py` returns hole-aware engine and lookup services when the hole connector is selected.

Relevant code:

- `lmcache_ascend/integration/vllm/lmcache_ascend_hole_connector_v1.py:LMCacheAscendHoleConnectorV1Dynamic`
- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole.__init__`
- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole._init_connector_state`
- `lmcache_ascend/__init__.py:get_or_create_lmcache_engine`
- `lmcache_ascend/__init__.py:maybe_create_lookup_client`
- `lmcache_ascend/__init__.py:maybe_create_lookup_server`

### Lookup phase

Standard CacheBlend asks a simple question: how long is the reusable contiguous prefix? Hole-mode asks a richer one: which semantic segments in the prefix are hits, which are misses, and how much of the overall prefix can be reused if the missing segments are computed fresh? The hole lookup server performs segment-aware lookup and returns a `HoleLookupResult`; the client passes that result into the connector.

Relevant code:

- `lmcache_ascend/v1/lookup_client/lmcache_hole_lookup_server.py:LMCacheHoleLookupServer._lookup_tokens`
- `lmcache_ascend/v1/lookup_client/lmcache_hole_lookup_client.py:LMCacheHoleLookupClient.lookup`
- `lmcache_ascend/v1/hole_segment_utils.py:derive_lookup_result`
- `lmcache_ascend/v1/hole_types.py:HoleLookupResult`

### Scheduler metadata

Once the lookup result exists, the scheduler side needs more than a scalar matched-prefix length. `HoleRequestTracker` and `HoleReqMeta` carry the hole-aware load specification through `get_num_new_matched_tokens`, `update_state_after_alloc`, and `build_connector_meta`. This is where the connector decides how many tokens are externally covered, which prefix region is a miss, and what the worker should load versus recompute.

Relevant code:

- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:HoleRequestTracker`
- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:HoleReqMeta`
- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole.get_num_new_matched_tokens`
- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole.update_state_after_alloc`
- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole.build_connector_meta`

### Worker load path

On the worker, `start_load_kv` diverges from the standard path most clearly. Standard CacheBlend loads a contiguous reusable prefix and calls the standard blender on that prefix. Hole-mode loads the covered region described by the `HoleLoadSpec`, including hit segments before and after the hole, and hands that hole-aware layout to the hole-specific NPU connector and blender.

Concretely, the hole connector retrieves cached K/V for each hit range and assembles them into a dense buffer covering the entire `covered_tokens` span, leaving hole positions as gaps to be filled by fresh computation during the blender's layer-0 forward.

Relevant code:

- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole.start_load_kv`
- `lmcache_ascend/v1/npu_hole_connector.py:VLLMBufferLayerwiseNPUHoleConnector`
- `lmcache_ascend/v1/blend/hole_blender.py:LMCBlenderHole`

### Blender behavior

The algorithmic divergence lives in the blender pair: `blender.py` for the standard contiguous-prefix path and `hole_blender.py` for segment-aware hole reuse. At layer 0, the hole blender processes QKV over the entire covered region (`R1 + H + R2`), materializing fresh K/V for hole positions into the same dense buffer as cached K/V from hit positions. At the check layer, it computes refreshed `diff_k` only on hit positions (the hole positions have no cached `old_k` to compare against), selects a topK set for refresh from those candidates, and forms the final recompute set as topK ∪ hole positions. For later layers, the recompute set propagates: hit positions outside topK continue to reuse cached K/V, while topK-selected positions and all hole positions are computed fresh.

Relevant code:

- `lmcache_ascend/v1/blend/hole_blender.py:LMCBlenderHole.process_qkv`
- `lmcache_ascend/v1/blend/hole_blender.py:LMCBlenderHole.blend_layer`
- `lmcache_ascend/v1/blend/hole_blender.py:LMCBlenderHole.blend`

### vLLM forward continuation

Once the covered region has been materialized, normal model execution resumes. The continuation forward only needs to run on the tail after `covered_tokens`, because the reusable and refreshed prefix region has already been handled by the hole-aware load + blend path. In that respect, hole-mode preserves the same division of labor as standard CacheBlend: connector and blender own the reusable prefix, vLLM forward owns the remainder.

Relevant code:

- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole.start_load_kv`
- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole.save_kv_layer`
- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py:LMCacheConnectorV1ImplHole.request_finished`

## 5. File map

Recommended reading order for code review:

1. `lmcache_ascend/v1/hole_types.py`
2. `lmcache_ascend/v1/hole_segment_utils.py`
3. `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py`
4. `lmcache_ascend/v1/blend/hole_blender.py`
5. `lmcache_ascend/v1/npu_hole_connector.py`
6. `lmcache_ascend/v1/lookup_client/lmcache_hole_lookup_client.py`
7. `lmcache_ascend/v1/lookup_client/lmcache_hole_lookup_server.py`
8. `lmcache_ascend/__init__.py`

Purely hole-specific files:

- `lmcache_ascend/v1/hole_types.py`
- `lmcache_ascend/v1/hole_segment_utils.py`
- `lmcache_ascend/integration/vllm/vllm_v1_adapter_hole.py`
- `lmcache_ascend/integration/vllm/lmcache_ascend_hole_connector_v1.py`
- `lmcache_ascend/v1/blend/hole_blender.py`
- `lmcache_ascend/v1/npu_hole_connector.py`
- `lmcache_ascend/v1/lookup_client/lmcache_hole_lookup_client.py`
- `lmcache_ascend/v1/lookup_client/lmcache_hole_lookup_server.py`

Hole-aware extensions to shared code:

- `lmcache_ascend/__init__.py`
  Service-factory patches and runtime extension wiring.

- `lmcache_ascend/v1/rpc_utils.py`
  Local `lookup_pure` service-name widening needed by the hole lookup architecture.

- `lmcache_ascend/v1/blend/`
  Shared blend infrastructure used by both modes. The `blender.py` / `hole_blender.py` pair is where the algorithmic divergence is most visible.

## 6. What's validated

The feature has working code on the current stack and has been exercised beyond import-only smoke checks.

- Single-hole runs on the Musique multi-hop QA dataset with Qwen3-8B have been rerun repeatedly while cleaning up the branch. Recent cleanup commits preserved byte-stable `f1` and hit-rate behavior across the validated baseline configuration.
- The multi-hole CLI path is validated empirically. A first two-hole benchmark run completed successfully and produced a distinct metric profile from the single-hole case, which confirms that the benchmark and runtime path are actually exercising multi-hole behavior rather than collapsing to single-hole semantics.
- The hole vs standard CacheBlend `f1` difference has been characterized on the Musique multi-hop QA dataset with Qwen3-8B. On a matched long-context comparison (`inflate=20`, `until=150`, `recompute_ratio=0.15`) standard CacheBlend reaches `f1=0.2230`; single-hole mode at `hole_pos=4` reaches `f1=0.2200`, with similarly high pass-2 cache hit rates (`0.9975` vs `0.9992`). The residual gap varies modestly with context configuration and traces to two sources: an intrinsic algorithmic difference in how recompute candidates are selected when hole positions are included, and bf16 numerics in the hole-aware materialization path.

## 7. Known limitations and follow-up items

Known follow-up items include refactoring the inline pure-lookup construction in the connector toward a more upstream-aligned shape, and replacing the temporary trace_utils compatibility shim with the proper upstream tracing path. These are noted as design discussion points rather than blockers, and are flagged in the relevant code comments.

## 8. Notes for refactor discussion

The feature is functional and behavior-stable on the current stack. The main remaining design questions are not about whether the feature works, but about how much of the hole-specific structure should be reshaped to match preferred upstream organization.

The two most prominent examples are inline pure-lookup construction in the connector and the temporary trace_utils compatibility shim, both flagged in the relevant code comments. This port intentionally did not attempt heavy refactoring in those areas before maintainer review. The current code is meant to be readable, working, and explicit about where it diverges from upstream assumptions. Refactor scope should be set with maintainer input.
