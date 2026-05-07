# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import json
import os
import random
import socket
import subprocess
import threading
import time

# Third Party
from lmcache.logging import init_logger
import torch

# First Party
import lmcache_ascend.hcomm_onesided as hcomm_os

# Local
from .hcomm_onesided_protocol import HcommDeviceInfo

logger = init_logger(__name__)

# SoC models that require v1.2 rank table with super_device_id / super_pod_list.
# All other SoCs use the simpler v1.0 format.
# Reference: hixl/src/llm_datadist/common/rank_table_generator.cc (kV2Version)
_V2_SOC_NAMES = frozenset(
    {
        "Ascend910_9391",
        "Ascend910_9381",
        "Ascend910_9392",
        "Ascend910_9382",
        "Ascend910_9372",
        "Ascend910_9362",
    }
)

_HCOMM_INIT_MAX_RETRIES = int(os.environ.get("LMCACHE_HCOMM_INIT_MAX_RETRIES", "3"))
_HCOMM_INIT_BASE_DELAY = 0.5
_HCOMM_INIT_MAX_DELAY = 5.0

_HCOMM_PREPARE_MAX_RETRIES = int(
    os.environ.get("LMCACHE_HCOMM_PREPARE_MAX_RETRIES", "10")
)
_HCOMM_PREPARE_BASE_DELAY = 0.2
_HCOMM_PREPARE_MAX_DELAY = 1.0

# Serialize init_comm_cluster_info / bind_mem / destroy_comm within a process.
# The global one-sided comm registry (g_oneSidedCommHcomInfos) in libhcomm
# is not thread-safe; concurrent calls from the server init thread and the
# client init path corrupt the map and leave zombie comm names that block
# subsequent retries with "comm Name exist" (HCCL_E_PARA).
# NOTE: prepare() must NOT be called under this lock -- it is a blocking
# rendezvous that needs both sides to call concurrently.
_hccl_init_lock = threading.Lock()


def _cleanup_failed_comm(comm: int, mem_handles: List[int]) -> None:
    """Best-effort teardown of a comm whose prepare or bind failed.

    Mirrors the cleanup sequence in ``HcommOneSidedChannel._destroy_peer_comm``
    (unbind all handles, then destroy the comm) so that the comm name is
    removed from ``g_oneSidedCommHcomInfos`` and the next retry can reuse it.

    Must be serialized via ``_hccl_init_lock`` because ``destroy_comm``
    modifies the same thread-unsafe global registry that
    ``init_comm_cluster_info`` writes to.
    """
    with _hccl_init_lock:
        for mh in mem_handles:
            try:
                hcomm_os.unbind_mem(comm, mh)
            except Exception:
                logger.warning("Failed to unbind mem %d from comm %d", mh, comm)
        try:
            hcomm_os.destroy_comm(comm)
        except Exception:
            logger.error("Failed to destroy comm %d", comm)


def _init_comm_and_prepare(
    cluster_json: str,
    comm_name: str,
    rank: int,
    mem_handles: List[int],
) -> int:
    """Blocking helper: init comm, bind mem handles, prepare with two-level retry.

    **Outer loop** (``_HCOMM_INIT_MAX_RETRIES``): retries the full
    init_comm + bind_mem sequence.  On failure the comm is destroyed so
    the name is freed in ``g_oneSidedCommHcomInfos``.

    **Inner loop** (``_HCOMM_PREPARE_MAX_RETRIES``): retries only
    ``prepare()`` on the *same* comm.  The HCCL C++ layer cleans socket
    resources on prepare failure (``CleanSocketResource``) but leaves the
    comm and bound memory intact, so calling ``HcclCommPrepare`` again is
    safe -- it re-runs ``CreateLinkFullmesh`` from scratch.  Keeping the
    comm alive avoids destroying the remote side's in-progress prepare
    and prevents the cascading desynchronization that exhausts retries on
    both sides.

    ``_hccl_init_lock`` serializes only ``init_comm_cluster_info`` and
    ``bind_mem`` (the operations that mutate the thread-unsafe global
    comm registry).  ``prepare()`` runs **outside** the lock.
    """
    last_err: Optional[RuntimeError] = None

    for init_attempt in range(_HCOMM_INIT_MAX_RETRIES):
        comm = None
        try:
            with _hccl_init_lock:
                comm = hcomm_os.init_comm_cluster_info(cluster_json, rank, comm_name)
                for mh in mem_handles:
                    hcomm_os.bind_mem(comm, mh)
        except RuntimeError as e:
            if comm is not None:
                _cleanup_failed_comm(comm, mem_handles)
            last_err = e
            delay = min(
                _HCOMM_INIT_BASE_DELAY * (2**init_attempt),
                _HCOMM_INIT_MAX_DELAY,
            )
            delay *= random.uniform(0.5, 1.5)
            logger.warning(
                "init/bind failed (attempt %d/%d): %s  retrying in %.2fs",
                init_attempt + 1,
                _HCOMM_INIT_MAX_RETRIES,
                e,
                delay,
            )
            time.sleep(delay)
            continue

        # init + bind succeeded -- retry prepare on the same comm
        for prep_attempt in range(_HCOMM_PREPARE_MAX_RETRIES):
            try:
                hcomm_os.prepare(comm, timeout=120)
                return comm
            except RuntimeError as e:
                last_err = e
                delay = min(
                    _HCOMM_PREPARE_BASE_DELAY * (2**prep_attempt),
                    _HCOMM_PREPARE_MAX_DELAY,
                )
                delay *= random.uniform(0.5, 1.5)
                logger.warning(
                    "prepare failed (attempt %d/%d) for comm %s: %s  retrying in %.2fs",
                    prep_attempt + 1,
                    _HCOMM_PREPARE_MAX_RETRIES,
                    comm_name,
                    e,
                    delay,
                )
                time.sleep(delay)

        # prepare retries exhausted -- tear down and start full cycle over
        logger.error(
            "prepare exhausted %d retries for comm %s, destroying comm",
            _HCOMM_PREPARE_MAX_RETRIES,
            comm_name,
        )
        _cleanup_failed_comm(comm, mem_handles)

    raise RuntimeError(
        f"_init_comm_and_prepare failed after "
        f"{_HCOMM_INIT_MAX_RETRIES} init attempts "
        f"(each with {_HCOMM_PREPARE_MAX_RETRIES} prepare retries)"
    ) from last_err


def _is_device_memory(ptr: int) -> bool:
    return hcomm_os.is_device_memory(ptr)


def _find_hccn_tool() -> str:
    """Locate the ``hccn_tool`` binary.

    Returns the first path that exists from common Ascend driver
    locations, or falls back to the bare name (relying on PATH).
    """
    candidates = [
        os.path.join(
            os.environ.get("ASCEND_DRIVER_HOME", "/usr/local/Ascend/driver"),
            "tools",
            "hccn_tool",
        ),
        "/usr/local/Ascend/driver/tools/hccn_tool",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return "hccn_tool"


def _get_device_ip(phy_device_id: int) -> str:
    """Read device IP from /etc/hccn.conf or fall back to hccn_tool.

    The device IP is required when ``HCCL_INTRA_ROCE_ENABLE=1`` (host-
    memory transfers via RoCE).  Without it the rank table omits the
    ``device_ip`` field, causing ``HcclCommInitClusterInfoMemConfig`` to
    reject the table with HCCL_E_PARA.
    """
    hccn_conf = "/etc/hccn.conf"
    if os.path.isfile(hccn_conf):
        key = f"address_{phy_device_id}="
        with open(hccn_conf) as f:
            for line in f:
                if line.startswith(key):
                    return line.strip().split("=", 1)[1]

    hccn_tool = _find_hccn_tool()
    try:
        result = subprocess.run(
            [hccn_tool, "-i", str(phy_device_id), "-ip", "-g"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "ipaddr:" in line:
                return line.split("ipaddr:")[1].strip()
        logger.warning(
            "hccn_tool ran but returned no ipaddr for device %d", phy_device_id
        )
    except FileNotFoundError:
        logger.warning(
            "hccn_tool not found (searched: %s). "
            "Mount /etc/hccn.conf or install hccn_tool to enable "
            "host-memory (RoCE) transfers.",
            hccn_tool,
        )
    except subprocess.TimeoutExpired:
        logger.warning("hccn_tool timed out querying device %d", phy_device_id)
    return ""


def _get_local_device_info() -> HcommDeviceInfo:
    """Gather local device metadata needed for the rank table."""
    device_id = torch.npu.current_device()
    info = hcomm_os.get_device_info(device_id)
    device_ip = _get_device_ip(info["phy_device_id"])
    soc_name = info.get("soc_name", "")
    use_v2 = soc_name in _V2_SOC_NAMES

    result = HcommDeviceInfo(
        server_id=socket.gethostname(),
        phy_device_id=str(info["phy_device_id"]),
        device_ip=device_ip,
        use_v2=use_v2,
    )
    if use_v2:
        result.super_device_id = str(info["super_device_id"])
        result.super_pod_id = str(info["super_pod_id"])
    logger.info(
        "Local device info: soc=%s v2=%s phy_dev=%s ip=%s",
        soc_name,
        use_v2,
        result.phy_device_id,
        result.device_ip,
    )
    return result


def _build_rank_table_json(
    server_info: HcommDeviceInfo,
    server_rank: int,
    client_info: HcommDeviceInfo,
    client_rank: int,
) -> str:
    """Build a rank-table JSON string for HcclCommInitClusterInfoMemConfig."""
    use_v2 = server_info.use_v2 and client_info.use_v2

    servers: dict[str, list[dict]] = {}
    for dev_info, rank_id in [
        (server_info, server_rank),
        (client_info, client_rank),
    ]:
        dev: dict = {
            "device_id": dev_info.phy_device_id,
            "rank_id": str(rank_id),
        }
        if dev_info.device_ip:
            dev["device_ip"] = dev_info.device_ip
        if use_v2:
            dev["super_device_id"] = dev_info.super_device_id
        servers.setdefault(dev_info.server_id, []).append(dev)

    server_list = []
    for sid, devices in servers.items():
        devices.sort(key=lambda d: int(d["rank_id"]))
        server_list.append({"server_id": sid, "device": devices})

    rank_table: dict = {
        "version": "1.2" if use_v2 else "1.0",
        "server_count": str(len(server_list)),
        "server_list": server_list,
        "status": "completed",
    }

    if use_v2:
        pod_map: dict[str, set[str]] = {}
        for dev_info in (server_info, client_info):
            pod_map.setdefault(dev_info.super_pod_id, set()).add(dev_info.server_id)
        super_pod_list = []
        for pod_id, sids in sorted(pod_map.items()):
            super_pod_list.append(
                {
                    "super_pod_id": pod_id,
                    "server_list": [{"server_id": s} for s in sorted(sids)],
                }
            )
        rank_table["super_pod_list"] = super_pod_list

    return json.dumps(rank_table)
