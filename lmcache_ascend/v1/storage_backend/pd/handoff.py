# SPDX-License-Identifier: Apache-2.0
"""Control-plane identifiers for PD pull handoff leases."""

# This value is transported through vLLM ``kv_transfer_params`` and therefore
# appears in LMCache ``request_configs``. It must not affect cache-key hashing.
PD_HANDOFF_REQUEST_CONFIG_KEY = "lmcache.pd_handoff_id"

# Synthetic request owner used between PullReady acknowledgement and the first
# decoder lookup. The prefix keeps it disjoint from real vLLM request IDs.
PD_HANDOFF_LEASE_PREFIX = "__lmcache_pd_handoff__:"


def make_pd_handoff_lease_id(handoff_id: str) -> str:
    """Return the receiver-local synthetic owner for ``handoff_id``."""
    if not handoff_id:
        raise ValueError("PD pull handoff_id must not be empty")
    return f"{PD_HANDOFF_LEASE_PREFIX}{handoff_id}"


def split_pd_handoff_request_config(
    request_configs: dict | None,
) -> tuple[str | None, dict | None]:
    """Extract PD handoff metadata without mutating the caller's dictionary."""
    if not request_configs or PD_HANDOFF_REQUEST_CONFIG_KEY not in request_configs:
        return None, request_configs

    sanitized = dict(request_configs)
    raw_handoff_id = sanitized.pop(PD_HANDOFF_REQUEST_CONFIG_KEY)
    handoff_id = str(raw_handoff_id) if raw_handoff_id is not None else None
    if not handoff_id:
        handoff_id = None
    return handoff_id, sanitized
