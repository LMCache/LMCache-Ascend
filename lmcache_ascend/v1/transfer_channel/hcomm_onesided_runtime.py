# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import json
import os
import random
import socket
import subprocess
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

_HCOMM_INIT_MAX_RETRIES = int(os.environ.get("LMCACHE_HCOMM_INIT_MAX_RETRIES", "5"))
_HCOMM_INIT_BASE_DELAY = 0.1
_HCOMM_INIT_MAX_DELAY = 5.0


def _init_comm_and_prepare(
    cluster_json: str,
    comm_name: str,
    rank: int,
    mem_handles: List[int],
) -> int:
    """Blocking helper: init comm via cluster-info JSON, bind all mem handles, prepare.

    Retries ``init_comm_cluster_info`` with exponential back-off because
    concurrent calls to HcclCommInitClusterInfoMemConfig from different
    processes on the same node can transiently fail (HCCL error 7).
    """
    last_err: Optional[RuntimeError] = None
    for attempt in range(_HCOMM_INIT_MAX_RETRIES):
        try:
            comm = hcomm_os.init_comm_cluster_info(cluster_json, rank, comm_name)
            break
        except RuntimeError as e:
            last_err = e
            delay = min(
                _HCOMM_INIT_BASE_DELAY * (2**attempt),
                _HCOMM_INIT_MAX_DELAY,
            )
            delay *= random.uniform(0.5, 1.5)
            logger.warning(
                "init_comm_cluster_info failed (attempt %d/%d): %s  retrying in %.2fs",
                attempt + 1,
                _HCOMM_INIT_MAX_RETRIES,
                e,
                delay,
            )
            time.sleep(delay)
    else:
        raise RuntimeError(
            f"init_comm_cluster_info failed after {_HCOMM_INIT_MAX_RETRIES} attempts"
        ) from last_err

    for mh in mem_handles:
        hcomm_os.bind_mem(comm, mh)
    hcomm_os.prepare(comm, timeout=120)
    return comm


def _is_device_memory(ptr: int) -> bool:
    return hcomm_os.is_device_memory(ptr)


def _get_device_ip(phy_device_id: int) -> str:
    """Read device IP from /etc/hccn.conf or fall back to hccn_tool."""
    hccn_conf = "/etc/hccn.conf"
    if os.path.isfile(hccn_conf):
        key = f"address_{phy_device_id}="
        with open(hccn_conf) as f:
            for line in f:
                if line.startswith(key):
                    return line.strip().split("=", 1)[1]
    try:
        result = subprocess.run(
            ["hccn_tool", "-i", str(phy_device_id), "-ip", "-g"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "ipaddr:" in line:
                return line.split("ipaddr:")[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("Failed to get device IP from hccn_tool")
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
