#pragma once
#include <iostream>
#include <vector>
#include <pybind11/stl.h>
#include <cstdint> 
#include "driver/ascend_hal_define.h"
#include "driver/ascend_hal.h"
#include "sys/mman.h"
#include <shared_mutex>
#include <mutex>
#include <unordered_map>
#include <string>
#include <cstring>
#include "acl/acl.h"

/*
When aclrtHostRegister is supported,
we could migrate the sharedmemory instantiation with the aclrtHostRegister call instead.
*/
namespace lmc {

constexpr int32_t PROT_FLAGS = static_cast<int32_t>(PROT_READ) | static_cast<int32_t>(PROT_WRITE);
constexpr int32_t MAP_FLAGS = static_cast<int32_t>(MAP_PRIVATE) | static_cast<int32_t>(MAP_ANONYMOUS) | static_cast<int32_t>(MAP_POPULATE);

struct PinnedMemoryRecord {
    uintptr_t ptr;
    uintptr_t devptr;
    size_t buffSize;
    int8_t device;
};

/* We are not responsible for acl init and ctx initialization,
   we assume the user responsible for ctx initialization
 */
class PinnedMemoryManager {
private:
    PinnedMemoryManager();

    // Delete copy constructor and assignment operator
    PinnedMemoryManager(const PinnedMemoryManager&) = delete;
    PinnedMemoryManager& operator=(const PinnedMemoryManager&) = delete;
    PinnedMemoryManager(PinnedMemoryManager&&) = delete;
    PinnedMemoryManager& operator=(PinnedMemoryManager&&) = delete;

    std::unordered_map<uintptr_t, PinnedMemoryRecord> allocatedMap;
    mutable std::shared_mutex mux;
    
    bool innerFree(uintptr_t hostPtr, size_t bufferSize, int8_t device);
    
public:
    static PinnedMemoryManager& GetInstance()
    {
        static PinnedMemoryManager instance;
        return instance;
    }
    ~PinnedMemoryManager();
    
    void* getDevicePtr(void* hostptr);
    uintptr_t allocPinned(size_t bufferSize);
    bool freePinned(uintptr_t hostPtr);
    void freeAll();
};

void* get_device_ptr(void* ptr);
uintptr_t alloc_pinned_mem(size_t bufferSize);
bool free_pinned_mem(uintptr_t hostPtr);
void free_all();
void pinned_memory_deleter(void* ptr);
}
