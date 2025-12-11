#include "managed_mem.h"
#include <acl/acl.h>
// Only required for old driver version (look at registerHostPtr)
#ifdef PROF_ERROR
    // You can add a pragma message to see this in your build log if you want:
    // #pragma message("Undefining PROF_ERROR from ascend_hal.h before NPU headers")
    #undef PROF_ERROR
#endif

#include <sys/mman.h>
#include "driver/ascend_hal_define.h"
#include "driver/ascend_hal.h"
#include <dlfcn.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "ms_extension.h"
#include "aclnn/opdev/platform.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "ms_extension/api.h"
#include "mem_alloc.h"

py::array create_mmapped_numpy(size_t buffer_size) {
    if (buffer_size <= 0 ) {
        throw std::runtime_error("Buffer size must be positive.");
    }

    int64_t numel = static_cast<int64_t>(buffer_size);

    uintptr_t ptr = alloc_pinned_ptr(buffer_size, 0);
    if (ptr == 0) {
        throw std::runtime_error("Failed to allocate pinned memory."); 
    }

    py::capsule free_mmap_capsule(
        reinterpret_cast<void*>(ptr), // Payload: the ID of the mmap'd region
        [](void* _ptr) {
            free_pinned_ptr(reinterpret_cast<uintptr_t>(_ptr));
        } 
    );

    return py::array(py::dtype::from_args(py::str("uint8")), {numel}, {1}, 
        reinterpret_cast<void*>(ptr), free_mmap_capsule);
}
