#include "pinned_mem.h"
#include <string>
#include <exception>
// HACK: undefine the version from ascend_hal.h.
#ifdef PROF_ERROR
    // You can add a pragma message to see this in your build log if you want:
    // #pragma message("Undefining PROF_ERROR from ascend_hal.h before NPU headers")
    #undef PROF_ERROR
#endif

#include <sys/mman.h>
#include "acl/acl.h"
#include "framework_hal.h"
#include "dcmi_management.h"
#include <sys/stat.h>
#include <unistd.h>
#include <memory>

namespace lmc {


// A more robust check for your specific environment.
// numa_available() sometimes does not work in openEuler?
bool is_numa_system_present() {
    struct stat st;
    // Check if the directory that the kernel uses to expose NUMA nodes exists.
    if (stat("/sys/devices/system/node", &st) != 0) {
        return false; // The path doesn't exist.
    }
    // Check if it's actually a directory.
    return S_ISDIR(st.st_mode);
}

// Function to parse a CPU range string (e.g., "144-167")
// and return the first CPU in the range.
int parse_first_cpu(const std::string& cpu_str) {
    try {
        size_t dash_pos = cpu_str.find('-');
        if (dash_pos != std::string::npos) {
            return std::stoi(cpu_str.substr(0, dash_pos));
        }
        // If it's a single CPU, not a range
        return std::stoi(cpu_str);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error("Invalid CPU string format. Could not parse number.");
    } catch (const std::out_of_range& e) {
        throw std::runtime_error("CPU number is out of range.");
    }
}

PinnedMemoryManager::PinnedMemoryManager() {
};

PinnedMemoryManager::~PinnedMemoryManager() {
    this->freeAll();
};

void* PinnedMemoryManager::getDevicePtr(void* hostPtr) {
    if (hostPtr == nullptr) {
        return nullptr;
    }

    const std::shared_lock<std::shared_mutex> lock(this->mux); 
    const uintptr_t hostAddrPtr = reinterpret_cast<uintptr_t>(hostPtr);

    for (const auto& pair: this->allocatedMap) {
        const PinnedMemoryRecord& record = pair.second;

        if (hostAddrPtr >= record.ptr && hostAddrPtr < (record.ptr + record.buffSize)) {
            const size_t offset = hostAddrPtr - record.ptr;

            const uintptr_t deviceAddrPtr = record.devptr + offset;

            return reinterpret_cast<void*>(deviceAddrPtr);
        }
    }

    return nullptr;
}

bool PinnedMemoryManager::innerFree(uintptr_t hostPtr, size_t bufferSize, int8_t device) {
    // since this is a pinned and dev accessible mem
    // we need to halhostunregister first
    auto ret = halHostUnregisterEx(reinterpret_cast<void*>(hostPtr), 
        static_cast<UINT32>(device), HOST_MEM_MAP_DEV_PCIE_TH);

    if (ret != 0) {
        std::cout << "Unable to hal host unregister: "<< ret << std::endl;
        return false;
    }

    auto mret = munmap(reinterpret_cast<void*>(hostPtr), bufferSize);
    if (mret != 0) {
        std::cout << "Unable to unmap memory: "<< ret << std::endl;
        return false;
    }

    return true;
};

bool PinnedMemoryManager::freePinned(uintptr_t hostPtr) {
    const std::unique_lock<std::shared_mutex> lock(this->mux);     

    // make sure this hostptr is in our allocated map
    if (this->allocatedMap.find(hostPtr) == this->allocatedMap.end()) {
        std::cerr << "HostPtr "<<hostPtr << " does not exists." << std::endl;
        return true; 
    }

    auto record = this->allocatedMap.at(hostPtr);
    auto freed = this->innerFree(hostPtr, record.buffSize, record.device);
    if (freed) {
        this->allocatedMap.erase(hostPtr);
        return true;
    }
    return false;
};

uintptr_t PinnedMemoryManager::allocPinned(size_t bufferSize) {
    auto device = framework_hal::GetDeviceIdx();

    uintptr_t hostPtr;
    int adviseErr;

    hostPtr = reinterpret_cast<uintptr_t>(mmap(nullptr, bufferSize, PROT_FLAGS, MAP_FLAGS, -1, 0));
    if ((void*) hostPtr == MAP_FAILED) {
        throw std::runtime_error("Unable to alloc memory with mmap.");
    } 

    adviseErr = madvise(reinterpret_cast<void*>(hostPtr), bufferSize, MADV_HUGEPAGE);

    if (adviseErr != 0) {
        // should be okay to continue
        std::cerr << "Unable to get madvise with HugePages: "<< adviseErr << std::endl;
    } 

    // set to all zeros
    memset(reinterpret_cast<void*>(hostPtr), 0, bufferSize);

    void* devPtr;
    drvError_t drvRet;
    drvRet = halHostRegister(reinterpret_cast<void*>(hostPtr), static_cast<UINT64>(bufferSize),
            HOST_MEM_MAP_DEV_PCIE_TH, static_cast<UINT32>(device), (void**)&devPtr);

    if (drvRet != 0) {
        throw std::runtime_error(std::string("Unable to register host memory with hal: ") + std::to_string(drvRet) + \
         " on device: " + std::to_string(device));
    } 

    auto lockErr = mlock(reinterpret_cast<void*>(hostPtr), bufferSize);
    if (lockErr == -1) {
        std::cerr << "Unable to pin host memory with error code: "<< std::to_string(lockErr) << std::endl;
        // this can happen in non-privileged mode or not enough rlimit, 
        // let's not proceed since we wanted to guarantee pinned
        // because we already alloced, let's free
        this->innerFree(hostPtr, bufferSize, static_cast<int8_t>(device));
        return 0;
    } 

    {
        const std::unique_lock<std::shared_mutex> lock(this->mux);     
        this->allocatedMap.emplace(hostPtr, PinnedMemoryRecord{hostPtr, reinterpret_cast<uintptr_t>(devPtr), bufferSize, static_cast<int8_t>(device)});
    }
    return hostPtr;
};

void PinnedMemoryManager::freeAll() {
    const std::shared_lock<std::shared_mutex> lock(this->mux); 
    if (!this->allocatedMap.empty()) {
        std::cerr << "PinnedMemoryManager::freeAll() called. "
                  << this->allocatedMap.size() 
                  << " block(s) were still allocated. This might indicate "
                  << "that not all PyTorch tensor deleters were invoked." << std::endl;
        
        // Iterate carefully as erasing modifies the map
        // One way is to collect keys then iterate, or use C++17 map::extract
        std::vector<uintptr_t> keys_to_free;
        for (const auto& pair : this->allocatedMap) {
            keys_to_free.push_back(pair.first);
        }

        for (uintptr_t hostPtr : keys_to_free) {
            // No need to check if it exists here, as we are iterating over existing keys
            auto record = this->allocatedMap.at(hostPtr); // Or find again, though at should be safe
            std::cerr << "PinnedMemoryManager::freeAll() freeing hostPtr: " << hostPtr 
                      << " with size: " << record.buffSize << std::endl;
            this->innerFree(hostPtr, record.buffSize, record.device);
            // The map entry will be removed below or after the loop
        }
        this->allocatedMap.clear(); // Clear the map after freeing all elements
    }
};

uintptr_t alloc_pinned_mem(size_t bufferSize) {
    auto& pmm = lmc::PinnedMemoryManager::GetInstance();
    return pmm.allocPinned(bufferSize);
}

void* get_device_ptr(void* ptr) {
    auto& pmm = lmc::PinnedMemoryManager::GetInstance();
    return pmm.getDevicePtr(ptr);
};

bool free_pinned_mem(uintptr_t hostptr) {
    auto& pmm = lmc::PinnedMemoryManager::GetInstance();
    return pmm.freePinned(hostptr);
}

void free_all() {
    auto& pmm = lmc::PinnedMemoryManager::GetInstance();
    pmm.freeAll();
}

void pinned_memory_deleter(void* ptr) {
    if(ptr) {
        free_pinned_mem(reinterpret_cast<uintptr_t>(ptr));
    }
}
} // namespace lmc