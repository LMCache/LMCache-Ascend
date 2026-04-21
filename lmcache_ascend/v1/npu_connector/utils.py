# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Union

# Third Party
from lmcache.v1.gpu_connector.utils import permute_to_contiguous
import torch

_KVTupleTwoOrMore = Tuple[torch.Tensor, ...]
_KVLayer = Union[torch.Tensor, _KVTupleTwoOrMore]


def permute_kv_caches_to_contiguous(
    kv_caches: List[_KVLayer],
) -> List[_KVLayer]:
    """Apply :func:`permute_to_contiguous` to each tensor in *kv_caches*.

    Each entry is either a single ``torch.Tensor`` (merged KV) or a tuple of
    two or more tensors (e.g. K/V, or more parts). The returned list has the
    same length and
    structure; tensors are metadata-only permutes where applicable and may
    share storage with the inputs (see upstream ``permute_to_contiguous``).
    """
    results: List[_KVLayer] = []
    for layer in kv_caches:
        if isinstance(layer, torch.Tensor):
            results.append(permute_to_contiguous(layer))
        elif isinstance(layer, tuple):
            if len(layer) < 2:
                raise ValueError(
                    "Tuple KV entries must contain at least two tensors; "
                    f"got len={len(layer)}"
                )
            permuted: list[torch.Tensor] = []
            for t in layer:
                if not isinstance(t, torch.Tensor):
                    raise ValueError(
                        f"Expected torch.Tensor inside KV tuple, got {type(t)}"
                    )
                permuted.append(permute_to_contiguous(t))
            results.append(tuple(permuted))
        else:
            raise ValueError(
                f"Unsupported KV cache entry type: {type(layer)} "
                "(expected Tensor or tuple of Tensors)"
            )
    return results
