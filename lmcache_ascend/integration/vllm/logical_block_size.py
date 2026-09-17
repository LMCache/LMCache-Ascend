# SPDX-License-Identifier: Apache-2.0
"""Wrap LMCache ``get_tokens_per_block`` when Ascend ``block_size`` is physical.

Pre-#13242 ``spec.block_size`` is physical slots; the logical span is
``block_size * compress_ratio``. Post-#13242 that field is already logical, so
LMCache core (``spec.block_size``) is correct and this wrap is skipped.

Detection is class-level at patch time: pool math
``max_memory_usage_bytes`` still names ``compress_ratio`` iff ``block_size`` is
physical. Do not ``getattr`` ``storage_block_size`` (GPU inherited property).

``merge()`` is always wrapped to restore ``compress_ratio`` / ``model_version``
dropped by some Ascend spec merges.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import sys

_INSTALLED = False
_GET_TOKENS_WRAPPED = False
_UNPATCHED_GET_TOKENS: Any = None

_ASCEND_LEAF_NAMES = frozenset(
    {
        "AscendMLAAttentionSpec",
        "AscendSlidingWindowMLASpec",
        "AscendSFAIndexerCacheSpec",
        "AscendIndexerKPoolStateSpec",
    }
)


def _is_ascend_leaf(spec: Any) -> bool:
    return any(cls.__name__ in _ASCEND_LEAF_NAMES for cls in type(spec).__mro__)


def _block_size_is_physical(cls: Any | None) -> bool:
    """True when ``cls.max_memory_usage_bytes`` still uses ``compress_ratio``."""
    if cls is None:
        return True
    fn = getattr(cls, "max_memory_usage_bytes", None)
    code = getattr(fn, "__code__", None)
    if code is None:
        return True
    return "compress_ratio" in code.co_names


def _ascend_block_size_is_physical() -> bool:
    """True on pre-#13242 vLLM-Ascend (physical ``block_size``)."""
    try:
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
    except ImportError:
        return True
    return _block_size_is_physical(AscendMLAAttentionSpec)


def tokens_per_block_id(spec: Any) -> int:
    """Logical tokens covered by one block id of ``spec`` (no DCP)."""
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        return max(tokens_per_block_id(leaf) for leaf in inner.values())
    block = int(spec.block_size)
    if not _is_ascend_leaf(spec):
        return block
    ratio = int(
        getattr(spec, "compress_ratio", None)
        or getattr(spec, "tokens_per_state", 1)
        or 1
    )
    return block * ratio


def _restore_dropped_merge_fields(merged: Any, specs: list[Any]) -> Any:
    first = specs[0]
    for name in ("compress_ratio", "model_version"):
        expected = getattr(first, name, None)
        if expected is None:
            continue
        if getattr(merged, name, None) != expected:
            object.__setattr__(merged, name, expected)
    return merged


def _wrap_merge(cls: type) -> None:
    orig = cls.merge
    orig_fn = getattr(orig, "__func__", orig)
    if getattr(orig_fn, "_lmcache_ascend_logical_block_size", False):
        return

    def merge(cls_: type, specs: list[Any]) -> Any:
        merged = orig_fn(cls_, specs)
        return _restore_dropped_merge_fields(merged, specs)

    merge._lmcache_ascend_logical_block_size = True  # type: ignore[attr-defined]
    cls.merge = classmethod(merge)


def install_on(
    leaf_classes: Iterable[type],
    uniform_type_cls: type | None = None,
) -> None:
    """Wrap ``merge()`` on the given spec classes (idempotent)."""
    del uniform_type_cls
    for cls in leaf_classes:
        if hasattr(cls, "merge"):
            _wrap_merge(cls)


def install_logical_block_size() -> bool:
    """Wrap real vLLM-Ascend spec ``merge()`` if the classes are importable."""
    try:
        from vllm_ascend.core.kv_cache_interface import (
            AscendMLAAttentionSpec,
            AscendSlidingWindowMLASpec,
        )
    except ImportError:
        return False
    install_on((AscendMLAAttentionSpec, AscendSlidingWindowMLASpec))
    return True


def _rebind_get_tokens_per_block(orig: Any, wrapped: Any) -> None:
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, "get_tokens_per_block", None) is orig:
                mod.get_tokens_per_block = wrapped
        except Exception:
            continue


def _wrap_get_tokens_per_block() -> None:
    """Replace ``get_tokens_per_block`` with the Ascend-aware span (idempotent)."""
    global _GET_TOKENS_WRAPPED, _UNPATCHED_GET_TOKENS
    if _GET_TOKENS_WRAPPED:
        return
    try:
        import lmcache.integration.vllm.kv_cache_groups as kg
    except ImportError:
        return
    orig = kg.get_tokens_per_block
    if getattr(orig, "_lmcache_ascend_logical_block_size", False):
        _GET_TOKENS_WRAPPED = True
        return

    def get_tokens_per_block(kv_cache_spec: Any, dcp_size: int) -> int:
        install_logical_block_size()
        block = tokens_per_block_id(kv_cache_spec)
        if dcp_size <= 1:
            return block
        if kg._is_attention_spec(kv_cache_spec):
            return block * dcp_size
        return block

    get_tokens_per_block._lmcache_ascend_logical_block_size = True  # type: ignore[attr-defined]
    _UNPATCHED_GET_TOKENS = orig
    _rebind_get_tokens_per_block(orig, get_tokens_per_block)
    _GET_TOKENS_WRAPPED = True


def install_overrides(*, scale_physical: bool | None = None) -> None:
    """Wrap spec ``merge()``; wrap ``get_tokens_per_block`` only if physical.

    ``scale_physical`` overrides the patch-time detector (tests). ``None``
    means detect from vLLM-Ascend. Idempotent: the first decision sticks.
    """
    global _INSTALLED, _GET_TOKENS_WRAPPED
    install_logical_block_size()
    if not _GET_TOKENS_WRAPPED:
        if scale_physical is None:
            scale_physical = _ascend_block_size_is_physical()
        if scale_physical:
            _wrap_get_tokens_per_block()
        else:
            _GET_TOKENS_WRAPPED = True
    _INSTALLED = True
