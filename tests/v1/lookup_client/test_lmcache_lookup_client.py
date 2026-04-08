# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
from lmcache_tests.v1.lookup_client.test_lmcache_lookup_client import (
    TestLMCacheLookupClientServer as UpstreamLMCacheLookupClientServerTests,
)
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PYTHONHASHSEED") is None,
    reason=(
        "PYTHONHASHSEED must be set for consistent hashing between "
        "LMCacheLookupClient and LMCacheLookupServer. "
        "Run with: PYTHONHASHSEED=0 pytest ..."
    ),
)


class TestLMCacheLookupClientServer(UpstreamLMCacheLookupClientServerTests):
    pass
