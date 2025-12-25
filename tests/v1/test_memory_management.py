# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
import lmcache_ascend

from lmcache_tests.v1.test_memory_management import (
    test_tensor_allocator,
    test_device_allocators,
    test_inplace_modification,
    test_boundary_alloc,
    test_batched_alloc,
    test_mixed_alloc
)