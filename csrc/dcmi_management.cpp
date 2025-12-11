#include "dcmi_management.h"
#include "dlfcn.h"
#include "torch/torch.h"
#include "torch/extension.h"
#include <sstream>
#include <string>

namespace dcmi_ascend {

using pcieInfoV2Func = int(*)(int, int, dcmi_pcie_info_all*);
using initFunc = int(*)(void);

DCMIManager::DCMIManager() {
    this->libHandle_ = dlopen("libdcmi.so", RTLD_LAZY | RTLD_GLOBAL);
    TORCH_CHECK(this->libHandle_ != nullptr, "dlopen libdcmi.so failed");
    auto init_func = reinterpret_cast<initFunc>(dlsym(this->libHandle_, "dcmi_init"));
    auto ret = init_func();
    TORCH_CHECK(ret == 0, "dcmi_init failed, ret = ", ret);
};

DCMIManager::~DCMIManager() {
    if (this->libHandle_ != nullptr) {
        dlclose(this->libHandle_);
    }
};

std::string DCMIManager::getDevicePcieInfoV2(int cardId, int deviceId, dcmi_pcie_info_all* pcieInfo) {
    auto func = reinterpret_cast<pcieInfoV2Func>(dlsym(this->libHandle_, "dcmi_get_device_pcie_info_v2"));
    auto ret = func(cardId, deviceId, pcieInfo);
    TORCH_CHECK(ret == 0, "dcmi_get_device_pcie_info_v2 failed, ret = ", ret);
    std::ostringstream oss;
    oss << std::setfill('0') 
        << std::hex 
        << std::setw(4) << pcieInfo->domain << ":"
        << std::setw(2) << pcieInfo->bdf_busId << ":"
        << std::setw(2) << pcieInfo->bdf_deviceId << "."
        << std::setw(1) << pcieInfo->bdf_funcId;
    return oss.str();
};
}