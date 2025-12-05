#pragma once
#include <string>

namespace lmc {
class DCMIManager {
private:
    DCMIManager();

    // Delete Copy constructor and assignment operator
    DCMIManager(const DCMIManager&) = delete;
    DCMIManager& operator=(const DCMIManager&) = delete;
    DCMIManager(DCMIManager&&) = delete;
    DCMIManager& operator=(DCMIManager&&) = delete;
    
    bool initialized;
    std::string cpuAffinity;

public:
    static DCMIManager& GetInstance()
    {
        static DCMIManager instance;
        return instance;
    }
    ~DCMIManager();
    
    // NOTE: at the moment we assume card and devId are the same.
    // there might be scenario you won't have the same card and devId.
    // we should indeed do this properly ?
    std::string getCPUAffinityFromDeviceId(int8_t cardId, int8_t devId);
};
}
