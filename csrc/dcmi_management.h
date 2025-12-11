#pragma once
#include <string>

namespace dcmi_ascend {

struct dcmi_pcie_info_all {
  unsigned int vendorId;
  unsigned int subvendorId;
  unsigned int deviceId;
  unsigned int subDeviceId;
  int domain;
  unsigned int bdf_busId;
  unsigned int bdf_deviceId;
  unsigned int bdf_funcId;
  unsigned char reserved[32];
};


/*
*  At build time, /usr/local/Ascend/driver is not present, 
*  however, at deployment time, Ascend software stack always requires driver to be mounted
*  hence, we opt to use the dlsym approach with the hopes that dcmi is always present.
*  And, we keep a handle of the it so with a singleton class.
*/
class DCMIManager {
private:
  void* libHandle_;
  DCMIManager();
  
  // Delete Copy constructor and assignment operator
  DCMIManager(const DCMIManager&) = delete;
  DCMIManager& operator=(const DCMIManager&) = delete;
  DCMIManager(DCMIManager&&) = delete;
  DCMIManager& operator=(DCMIManager&&) = delete;
public:
  static DCMIManager& GetInstance(){
    static DCMIManager instance;
    return instance;
  };
  ~DCMIManager();

  std::string getDevicePcieInfoV2(int cardId, int deviceId, dcmi_pcie_info_all* pcieInfo);
};
}