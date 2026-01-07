# SPDX-License-Identifier: Apache-2.0
# Third Party
from lmcache_tests.v1.cache_controller.test_locks import (
    TestFastLockWithTimeout as UpstreamFastLockWithTimeoutTest,
)
from lmcache_tests.v1.cache_controller.test_locks import (
    TestRWLockWithTimeout as UpstreamRWLockWithTimeoutTest,
)


class TestRWLockWithTimeout(UpstreamRWLockWithTimeoutTest):
    pass


class TestFastLockWithTimeout(UpstreamFastLockWithTimeoutTest):
    pass
