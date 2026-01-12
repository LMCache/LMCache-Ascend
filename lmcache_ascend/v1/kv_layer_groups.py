# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from typing import Tuple

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo
import torch

logger = init_logger(__name__)


def patched_hidden_dim_size(self) -> int:
    """Return the size of the hidden dimension in this group."""
    # hidden_dim_size = num_heads * head_size
    if len(self.shape) == 5:
        # MHA
        return self.shape[3] * self.shape[4]
    elif len(self.shape) == 4:
        # NOTE(gingfung): Ascend separated format for KVCaches
        # i.e. a tuple of kv (numblocks, blocksize, heads, headdim)
        #      very unlikely, but potentially MLA with (1, ....)
        if self.shape[0] == 1:
            raise ValueError(f"Invalid shape for hidden dim size: {self.shape}")

        return self.shape[2] * self.shape[3]
    elif len(self.shape) == 3:
        # MLA
        return self.shape[2]
    else:
        raise ValueError(f"Invalid shape: {self.shape}")


def build_kv_layer_groups(self, kv_caches: dict[str, torch.Tensor]) -> None:
    """Build KV layer groups structure by analyzing each layer's shape and dtype.

    Layers with the same shape and dtype are grouped together. This is useful
    because different layers may have different structures (especially the
    last dimension head_size may differ between groups), and different groups
    may have different dtypes.

    If layer groups are already built (non-empty list), this method does nothing.

    Args:
        kv_caches: Dictionary mapping layer names to KV cache tensors.
    """
    # Skip if already built (non-empty list)
    if len(self.kv_layer_groups) > 0:
        return

    if len(kv_caches) == 0:
        logger.debug("No KV caches available, skipping KV layer groups building")
        return

    # Group layers by (shape, dtype) in a single loop
    groups_dict: dict[tuple[torch.Size, torch.dtype], list[tuple[str, int]]] = (
        defaultdict(list)
    )

    for idx, (layer_name, kv_cache) in enumerate(kv_caches.items()):
        logger.debug(f"KVCache Type: {type(kv_cache)} - len: {len(kv_cache)}")
        # NOTE (gingfung): Ascend has tuple of kvcaches
        if isinstance(kv_cache, tuple):
            shape = kv_cache[0].shape
            dtype = kv_cache[0].dtype
        elif isinstance(kv_cache, torch.Tensor):
            shape = kv_cache.shape
            dtype = kv_cache.dtype
        else:
            raise RuntimeError(f"Unknown KVCache type: {type(kv_cache)}")

        key = (shape, dtype)
        groups_dict[key].append((layer_name, idx))

    # Build KVLayerGroupInfo list
    # Sort groups by the first layer index to maintain order
    def _get_first_layer_index(shape_dtype_key):
        """Get the index of the first layer in a layer group."""
        layer_group = groups_dict[
            shape_dtype_key
        ]  # list of (layer_name, layer_index) tuples
        first_layer_info = layer_group[0]  # first (layer_name, layer_index) tuple
        layer_index = first_layer_info[1]  # extract the layer index
        return layer_index

    sorted_keys = sorted(groups_dict.keys(), key=_get_first_layer_index)

    kv_layer_groups: list[KVLayerGroupInfo] = []
    for shape, dtype in sorted_keys:
        layers = groups_dict[(shape, dtype)]
        layer_names, layer_indices = zip(*layers, strict=False)

        group_info = KVLayerGroupInfo(
            layer_names=list(layer_names),
            layer_indices=list(layer_indices),
            shape=shape,
            dtype=dtype,
        )
        kv_layer_groups.append(group_info)

    # Store the built groups
    self.kv_layer_groups = kv_layer_groups

    # Print the group structure
    logger.info("KV layer groups: %s", kv_layer_groups)
