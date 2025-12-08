#include "dcmi_management.h"
#include <iostream>
#include <string>

namespace lmc {
DCMIManager::DCMIManager() {
    throw std::runtime_error("not support!\n");
};

DCMIManager::~DCMIManager() {
};

std::string DCMIManager::getCPUAffinityFromDeviceId(int8_t cardId, int8_t devId) {
    throw std::runtime_error("not support!\n");
}
}