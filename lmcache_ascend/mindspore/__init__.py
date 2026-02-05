# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, E501


def _patch_storage_manager():
    # Third Party
    import lmcache.v1.storage_backend.storage_manager as sm_module

    # First Party
    from lmcache_ascend.mindspore.v1.storage_backend.storage_manager import (
        StorageManager__init__,
        allocate_and_copy_objects_310p,
    )

    sm_module.StorageManager.__init__ = StorageManager__init__
    sm_module.StorageManager.allocate_and_copy_objects = allocate_and_copy_objects_310p


def _patch_memory_management():
    # This modification was primarily made due to MindSpore's incomplete support 
    # for certain Tensor operations, which necessitated a switch to NumPy implementation. 
    # The changes stem from the need to manage CPU memory using NumPy arrays.

    # Third Party
    import lmcache.v1.memory_management

    # First Party
    from lmcache_ascend.mindspore.v1.memory_management import (
        NumpyAndTensorMemoryAllocator,
        NumpyAndTensorMemoryObj,
        _allocate_cpu_memory,
    )

    lmcache.v1.memory_management._allocate_cpu_memory = _allocate_cpu_memory
    lmcache.v1.memory_management.TensorMemoryObj = NumpyAndTensorMemoryObj
    lmcache.v1.memory_management.TensorMemoryAllocator = NumpyAndTensorMemoryAllocator


def _patch_storage_backend_interface():
    # This change was made because MindSpore's multi-stream implementation on the 310P machine
    # would cause event resource exhaustion. Therefore, multi-stream is temporarily disabled.
    # If MindSpore later fixes its multi-stream implementation, this restriction can be removed.

    # Third Party
    import lmcache.v1.storage_backend

    # First Party
    from lmcache_ascend.mindspore.v1.storage_backend.abstract_backend import (
        StorageBackendInterface___init__,
    )

    lmcache.v1.storage_backend.StorageBackendInterface.__init__ = (
        StorageBackendInterface___init__
    )


def _patch_mooncake_store_connector():
    # Third Party
    import lmcache.v1.storage_backend.connector.mooncakestore_connector as mooncakestore_connector

    # First Party
    from lmcache_ascend.mindspore.v1.storage_backend.connector.mooncakestore_connector import (
        MooncakeStoreConnector__batch_get_into,
        MooncakeStoreConnector__put_without_metadata,
        MooncakeStoreConnector__register_cpu_buffer,
    )

    mooncakestore_connector.MooncakestoreConnector._register_cpu_buffer = (
        MooncakeStoreConnector__register_cpu_buffer
    )

    mooncakestore_connector.MooncakestoreConnector._batch_get_into = (
        MooncakeStoreConnector__batch_get_into
    )

    mooncakestore_connector.MooncakestoreConnector._put_without_metadata = (
        MooncakeStoreConnector__put_without_metadata
    )


def _patch_sys_detection():
    # Third Party
    import lmcache.v1.system_detection

    # First Party
    from lmcache_ascend.mindspore.v1.system_detection import _read_from_sys

    lmcache.v1.system_detection.NUMADetector._read_from_sys = _read_from_sys


_patch_storage_manager()
_patch_memory_management()
_patch_storage_backend_interface()
_patch_mooncake_store_connector()
_patch_sys_detection()
