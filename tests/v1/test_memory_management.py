# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache_tests.v1.test_memory_management import (  # noqa: F401
    test_batched_alloc,
    test_boundary_alloc,
    test_device_allocators,
    test_inplace_modification,
    test_memory_obj_metadata_to_and_from_dict,
    test_mixed_alloc,
    test_pin_monitor_background_thread,
    test_pin_monitor_timeout,
    test_pin_timeout,
    test_tensor_allocator,
    test_tensor_memory_obj_pin_monitor_integration,
)
