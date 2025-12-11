#include "managed_mem.h"
#include <acl/acl.h>
// Only required for old driver version (look at registerHostPtr)
#ifdef PROF_ERROR
    // You can add a pragma message to see this in your build log if you want:
    // #pragma message("Undefining PROF_ERROR from ascend_hal.h before NPU headers")
    #undef PROF_ERROR
#endif
#include <iostream>
#include <string>
#include <sys/mman.h>

#include "torch/torch.h"
#include "torch/extension.h"

#include "exception.h"
#include "framework_hal.h"
#include "managed_mem.h"

constexpr int32_t PROT_FLAGS = static_cast<int32_t>(PROT_READ) | static_cast<int32_t>(PROT_WRITE);
constexpr int32_t MAP_FLAGS = static_cast<int32_t>(MAP_PRIVATE) | static_cast<int32_t>(MAP_ANONYMOUS) | static_cast<int32_t>(MAP_POPULATE);


void swap_tensor_ptr(void* hostPtr, torch::Tensor& original_tensor){
    torch::TensorOptions tensorOpsCpu = torch::TensorOptions()
                                                .dtype(original_tensor.dtype())
                                                .device(original_tensor.device())
                                                .pinned_memory(true);
    int64_t numel = static_cast<int64_t>(original_tensor.nbytes());
    std::vector<int64_t> dims = {numel};
    torch::Tensor new_tensor_from_myptr = torch::from_blob(
        hostPtr, dims, lmc::hal_host_unregister_ptr, tensorOpsCpu);

    original_tensor.set_(new_tensor_from_myptr.storage(), original_tensor.storage_offset(), 
        original_tensor.sizes(), original_tensor.strides());
}

void* register_tensor(torch::Tensor& tensor) {
    torch::Device device = tensor.device();
    if (!device.is_cpu() || !tensor.is_pinned()) {
        TORCH_CHECK(false, "Invalid device. Device must be CPU and tensor must be pinned.");
    }
    auto& hmm = lmc::HostRegisteredMemoryManager::GetInstance();
    size_t tensorSize = tensor.nbytes();
    std::string verString = lmc::get_driver_version();
    if (lmc::is_version_at_least_25(verString)) { // New driver version, supports aclrtHostRegister()
        void* hostPtr = static_cast<void*>(tensor.data_ptr());
        auto record = hmm.registerHostPtr(hostPtr, tensorSize);

        return (void*) record->devptr;
    } else { // Old driver version, does not support aclrtHostRegister(), we have to use HAL.
        // We ask for a new registerd memory and substitute with the previously allocated.
        void* hostPtr;
        // Allocate and register
        hostPtr = mmap(nullptr, tensorSize, PROT_FLAGS, MAP_FLAGS, -1, 0);
        TORCH_CHECK(hostPtr != MAP_FAILED, "Failed to mmap");
        madvise(reinterpret_cast<void*>(hostPtr), tensorSize, MADV_HUGEPAGE);
        auto record = hmm.halRegisterHostPtr(hostPtr, tensorSize);
        if (record == nullptr) {
            munmap(hostPtr, tensorSize);
            TORCH_CHECK(false, "Failed to register memory");
        }
        swap_tensor_ptr((void*) record->ptr, tensor);
        return (void*) record->devptr;
    }
};
