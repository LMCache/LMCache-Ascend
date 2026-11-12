# SPDX-License-Identifier: Apache-2.0
"""Compatibility re-export. The implementation lives in LMCache."""

from lmcache.v1.gpu_connector.gds_backends._npu_host_callback import (
    NpuStreamHostCallbackQueue,
    exception_to_submission_result,
    launch_npu_stream_host_callback,
)

__all__ = [
    "NpuStreamHostCallbackQueue",
    "exception_to_submission_result",
    "launch_npu_stream_host_callback",
]
