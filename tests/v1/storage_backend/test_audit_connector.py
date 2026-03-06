# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401
# Third Party
from lmcache_tests.v1.storage_backend.test_audit_connector import (
    TestAuditConnector as UpstreamAuditConnectorTests,
)
from lmcache_tests.v1.storage_backend.test_audit_connector import (
    event_loop,
    local_cpu_backend,
    log_capture,
    mock_connector,
)


class TestAuditConnector(UpstreamAuditConnectorTests):
    pass
