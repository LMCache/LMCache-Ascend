# SPDX-License-Identifier: Apache-2.0
# Third Party

import pytest
import torch

# First Party
import lmcache_ascend

from lmcache_tests.v1.test_connector import (
    test_lm_connector,
    test_fs_connector,
    test_redis_connector,
    test_redis_sentinel_connector,
)