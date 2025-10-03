#include "dcmi_management.h"
#include <stdexcept>
#include "utils.h"
#include <string>

std::string get_npu_pci_bus_id(int device) {
  auto& dcmiManager = dcmi_ascend::DCMIManager::GetInstance();
  dcmi_ascend::dcmi_pcie_info_all pcieInfo;
  // TODO: at the moment, we don't account for multiple dies, where the 0 would have been die Id.
  return dcmiManager.getDevicePcieInfoV2(device, 0, &pcieInfo);
}
