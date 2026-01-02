#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>

namespace py = pybind11;
namespace framework_hal {
// NOTE (Gingfung): This function encapsulates the logic of calling MindSpore's
// Python API The reason we had to do this were because when I directly included
// the mindspore runtime apis for building, it kept erroring with linking
// issues. I was in touch with a couple of MindSpore guys but never really got
// it working. Anyway, we will revisit using Mindspore runtime API once we are
// on 2.7.
int32_t _get_device_id_from_mindspore_hal_python_api_impl() {
  int device_id = -1;

  try {
    py::module_ ms = py::module_::import("mindspore");
    py::module_ hal = ms.attr("hal");
    py::object current_stream_obj = hal.attr("current_stream")();
    py::object device_id_py_obj = current_stream_obj.attr("device_id");
    device_id = device_id_py_obj.cast<int32_t>();

    // NOTE: Mindspore current_device_id starts from 0 no matter what, so we
    // should shift the device id iff users specify the
    // ASCEND_RT_VISIBLE_DEVICES
    const char *env_visible_devices_p =
        std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    if (env_visible_devices_p != nullptr) {
      std::string env_visible_devices = env_visible_devices_p;
      std::vector<uint32_t> list_visible_devices;
      std::stringstream ss(env_visible_devices);
      std::string item;
      while (std::getline(ss, item, ',')) {
        list_visible_devices.push_back(std::stoi(item));
      }
      std::sort(list_visible_devices.begin(), list_visible_devices.end());
      // from what I have seen, there are two cases:
      // 1. no hccl, we just use current_device, even though we have specify the
      // ASCEND_RT_VISIBLE_DEVICES
      // 2. hccl, and we use current_device that seems to be correct
      // for case 2, since the current_device would have been correct anyway,
      // obtaining from the list would be fine. for case 1, we have shifted the
      // device to the RT_VISIBLE_DEVICES, so it should be corrected.
      device_id = list_visible_devices[device_id];
    }

  } catch (const py::error_already_set &e) {
    std::cerr << "C++ Internal: Python error in "
                 "_get_device_id_from_mindspore_hal_python_api_impl: "
              << e.what() << std::endl;
    PyErr_Print(); // Print Python traceback
    device_id = -1;
  } catch (const std::exception &e) {
    std::cerr << "C++ Internal: Standard C++ exception in "
                 "_get_device_id_from_mindspore_hal_python_api_impl: "
              << e.what() << std::endl;
    device_id = -1;
  }
  return device_id;
}

int8_t GetDeviceIdx() {
  // FIXME: should be always within 127 ?
  auto devId = _get_device_id_from_mindspore_hal_python_api_impl();
  if (devId == -1) {
    throw std::runtime_error(
        "Failed to get device ID from MindSpore's Python API.");
  }
  return devId;
};
} // namespace framework_hal