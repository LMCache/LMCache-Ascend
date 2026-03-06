# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache_tests.v1.cache_controller.test_registry_tree import (
    TestBatchOperations as UpstreamBatchOperationsTest,
)
from lmcache_tests.v1.cache_controller.test_registry_tree import (
    TestInstanceNodeLocking as UpstreamInstanceNodeLockingTest,
)
from lmcache_tests.v1.cache_controller.test_registry_tree import (
    TestRegistryTreeFineGrainedLocking as UpstreamRegistryTreeFineGrainedLockingTest,
)
from lmcache_tests.v1.cache_controller.test_registry_tree import (
    TestWorkerNodeLocking as UpstreamWorkerNodeLockingTest,
)


class TestWorkerNodeLocking(UpstreamWorkerNodeLockingTest):
    pass


class TestInstanceNodeLocking(UpstreamInstanceNodeLockingTest):
    pass


class TestRegistryTreeFineGrainedLocking(UpstreamRegistryTreeFineGrainedLockingTest):
    pass


class TestBatchOperations(UpstreamBatchOperationsTest):
    pass
