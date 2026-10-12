# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import importlib.util

# Third Party
import pytest

HANDOFF_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "lmcache_ascend"
    / "v1"
    / "storage_backend"
    / "pd"
    / "handoff.py"
)


def _load_handoff_module():
    spec = importlib.util.spec_from_file_location(
        "pd_handoff_under_test", HANDOFF_MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_pd_handoff_lease_id_uses_reserved_owner_prefix():
    handoff = _load_handoff_module()

    lease_id = handoff.make_pd_handoff_lease_id("request-123")

    assert lease_id == "__lmcache_pd_handoff__:request-123"


@pytest.mark.parametrize("handoff_id", ["", None])
def test_make_pd_handoff_lease_id_rejects_empty_values(handoff_id):
    handoff = _load_handoff_module()

    with pytest.raises(ValueError, match="must not be empty"):
        handoff.make_pd_handoff_lease_id(handoff_id)


def test_split_pd_handoff_request_config_preserves_cache_namespace_and_input():
    handoff = _load_handoff_module()
    original = {
        "lmcache.pd_handoff_id": "handoff-123",
        "lmcache.tag.tenant": "tenant-a",
    }

    handoff_id, sanitized = handoff.split_pd_handoff_request_config(original)

    assert handoff_id == "handoff-123"
    assert sanitized == {"lmcache.tag.tenant": "tenant-a"}
    assert original == {
        "lmcache.pd_handoff_id": "handoff-123",
        "lmcache.tag.tenant": "tenant-a",
    }


def test_split_pd_handoff_request_config_reuses_input_without_control_key():
    handoff = _load_handoff_module()
    original = {"lmcache.tag.tenant": "tenant-a"}

    handoff_id, sanitized = handoff.split_pd_handoff_request_config(original)

    assert handoff_id is None
    assert sanitized is original
