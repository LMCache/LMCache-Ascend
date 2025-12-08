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
#include "pinned_mem.h"
#include "ms_extension.h"
#include "aclnn/opdev/platform.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "ms_extension/api.h"

uintptr_t alloc_pinned_ptr(std::size_t buffer_size, unsigned int flags) {
    if (buffer_size <= 0 ) {
        throw std::runtime_error("Buffer size must be positive.");
    }

    int64_t numel = static_cast<int64_t>(buffer_size);

    uintptr_t ptr = lmc::alloc_pinned_mem(buffer_size);
    if (ptr == 0) {
        throw std::runtime_error("Failed to allocate pinned memory."); 
    }

    return ptr;
}

void free_pinned_ptr(uintptr_t ptr) {
    lmc::free_pinned_mem(reinterpret_cast<uintptr_t>(ptr));
}

/*
* This function is potentially slow for the mbind
*/
uintptr_t alloc_pinned_numa_ptr(std::size_t size, int node) {
    return alloc_pinned_ptr(size, 0);
}

void free_pinned_numa_ptr(uintptr_t p, std::size_t size) {
    free_pinned_ptr(p);
}
