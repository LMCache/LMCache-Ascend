# SPDX-License-Identifier: Apache-2.0
from lmcache_ascend import _build_info

if _build_info.__framework_name__ == "pytorch":
    import sys
    from functools import partial
    import lmcache

    import lmcache_ascend
    import lmcache_ascend.c_ops as ascend_c_ops

    # TODO: Currently we patch all the cuda calls due to effort to port all torch.cuda
    # will disabled torch.jit. This needs to be called before other imports, since some
    # of them have conditional imports based on torch.cuda.is_available()
    from torch_npu.contrib import transfer_to_npu

    sys.modules["lmcache.c_ops"] = ascend_c_ops

    from lmcache_ascend.v1.transfer_channel import (
        CreateTransferChannel as AscendCreateTransferChannel,
        get_correct_device as ascend_get_correct_device
    )

    # Make sure to import before importing init_lmcache_engine, otherwise CreateTransferChannel gets patched
    # after the original version is already imported.
    sys.modules["lmcache.v1.transfer_channel"].CreateTransferChannel = AscendCreateTransferChannel
    sys.modules["lmcache.v1.transfer_channel.transfer_utils"].get_correct_device = ascend_get_correct_device

    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        init_lmcache_engine as ascend_init_lmcache_engine,
    )

    from lmcache_ascend.v1.blend.utils import get_or_create_blender
    from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
    LMCBlenderBuilder.get_or_create = partial(get_or_create_blender, LMCBlenderBuilder)

    import lmcache.integration.vllm.vllm_v1_adapter
    lmcache.integration.vllm.vllm_v1_adapter._init_lmcache_engine = ascend_init_lmcache_engine

    # On OpenEuler and python3.10, the _hash_tokens func hash(None) seems to run into
    # ASLR lead to non-deterministic hashing for builtin hash
    import lmcache.v1.token_database
    from lmcache_ascend.v1.tokens_hash import _hash_tokens
    lmcache.v1.token_database.TokenDatabase._hash_tokens = _hash_tokens

    # Patching this as on some Ascend machines the kernel can set the NUMA node to
    # -1. If propagated in the NUMA mapping, this can cause failures to the caller.
    # The patch sanitizes negative values with None, and is up to the caller to handle it.
    import lmcache.v1.system_detection
    from lmcache_ascend.v1.system_detection import _read_from_sys
    lmcache.v1.system_detection.NUMADetector._read_from_sys = _read_from_sys
