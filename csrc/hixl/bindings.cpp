#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
#define _GLIBCXX_USE_CXX11_ABI 0

#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "hcomm_devva.h"
#include "hixl/hixl.h"
#include "hixl/hixl_types.h"
#include "runtime/mem.h"

namespace py = pybind11;

static void check_status(hixl::Status status, const std::string &error_msg) {
  if (status != hixl::SUCCESS) {
    throw std::runtime_error(error_msg +
                             " | HIXL Error Code: " + std::to_string(status));
  }
}

PYBIND11_MODULE(hixl_npu_comms, m) {
  m.doc() = "pybind11 wrapper for HIXL low-level transfer engine";

  m.def(
      "is_device_memory",
      [](uintptr_t ptr) -> bool {
        rtPointerAttributes_t attributes;
        auto ret =
            rtPointerGetAttributes(&attributes, reinterpret_cast<void *>(ptr));
        if (ret != ACL_SUCCESS) {
          return false; // fallback: assume host memory (safer for HIXL
                        // registration)
        }
        return attributes.memoryType != RT_MEMORY_TYPE_HOST &&
               attributes.memoryType != RT_MEMORY_TYPE_USER;
      },
      py::arg("ptr"),
      "Detect whether a pointer refers to device or host memory via ACL "
      "runtime.");

  m.def(
      "get_dev_va",
      [](int device_id, uintptr_t host_ptr, size_t size) -> py::object {
        void *dev_va = nullptr;
        int ret = hcomm_get_dev_va(
            device_id, reinterpret_cast<void *>(host_ptr), size, &dev_va);
        if (ret != 0) {
          return py::none();
        }
        return py::cast(reinterpret_cast<uintptr_t>(dev_va));
      },
      py::arg("device_id"), py::arg("host_ptr"), py::arg("size"),
      "Get the device VA for host memory already registered via hcomm. "
      "Returns None on failure.");

  // Status codes
  m.attr("SUCCESS") = hixl::SUCCESS;
  m.attr("PARAM_INVALID") = hixl::PARAM_INVALID;
  m.attr("TIMEOUT") = hixl::TIMEOUT;
  m.attr("NOT_CONNECTED") = hixl::NOT_CONNECTED;
  m.attr("ALREADY_CONNECTED") = hixl::ALREADY_CONNECTED;
  m.attr("NOTIFY_FAILED") = hixl::NOTIFY_FAILED;
  m.attr("UNSUPPORTED") = hixl::UNSUPPORTED;
  m.attr("FAILED") = hixl::FAILED;
  m.attr("RESOURCE_EXHAUSTED") = hixl::RESOURCE_EXHAUSTED;

  py::enum_<hixl::MemType>(m, "MemType")
      .value("MEM_DEVICE", hixl::MEM_DEVICE)
      .value("MEM_HOST", hixl::MEM_HOST)
      .export_values();

  py::enum_<hixl::TransferOp>(m, "TransferOp")
      .value("READ", hixl::READ)
      .value("WRITE", hixl::WRITE)
      .export_values();

  py::enum_<hixl::TransferStatus>(m, "TransferStatus")
      .value("WAITING", hixl::TransferStatus::WAITING)
      .value("COMPLETED", hixl::TransferStatus::COMPLETED)
      .value("TIMEOUT", hixl::TransferStatus::TIMEOUT)
      .value("FAILED", hixl::TransferStatus::FAILED)
      .export_values();

  py::class_<hixl::MemDesc>(m, "MemDesc")
      .def(py::init([](uintptr_t addr, size_t len) {
             hixl::MemDesc desc{};
             desc.addr = addr;
             desc.len = len;
             return desc;
           }),
           py::arg("addr"), py::arg("len"))
      .def_readwrite("addr", &hixl::MemDesc::addr)
      .def_readwrite("len", &hixl::MemDesc::len);

  py::class_<hixl::TransferOpDesc>(m, "TransferOpDesc")
      .def(
          py::init([](uintptr_t local_addr, uintptr_t remote_addr, size_t len) {
            return hixl::TransferOpDesc{local_addr, remote_addr, len};
          }),
          py::arg("local_addr"), py::arg("remote_addr"), py::arg("len"))
      .def_readwrite("local_addr", &hixl::TransferOpDesc::local_addr)
      .def_readwrite("remote_addr", &hixl::TransferOpDesc::remote_addr)
      .def_readwrite("len", &hixl::TransferOpDesc::len);

  py::class_<hixl::Hixl>(m, "Hixl")
      .def(py::init<>())
      .def(
          "initialize",
          [](hixl::Hixl &self, const std::string &local_engine,
             const std::map<std::string, std::string> &py_options) {
            std::map<hixl::AscendString, hixl::AscendString> options;
            for (const auto &kv : py_options) {
              options[hixl::AscendString(kv.first.c_str())] =
                  hixl::AscendString(kv.second.c_str());
            }
            check_status(self.Initialize(
                             hixl::AscendString(local_engine.c_str()), options),
                         "HIXL Initialize failed");
          },
          py::arg("local_engine"),
          py::arg("options") = std::map<std::string, std::string>{})
      .def("finalize", &hixl::Hixl::Finalize)
      .def(
          "register_mem",
          [](hixl::Hixl &self, uintptr_t addr, size_t len, hixl::MemType type) {
            hixl::MemDesc desc{};
            desc.addr = addr;
            desc.len = len;
            hixl::MemHandle handle = nullptr;
            check_status(self.RegisterMem(desc, type, handle),
                         "HIXL RegisterMem failed");
            return reinterpret_cast<uintptr_t>(handle);
          },
          py::arg("addr"), py::arg("len"), py::arg("type"))
      .def(
          "deregister_mem",
          [](hixl::Hixl &self, uintptr_t handle) {
            check_status(
                self.DeregisterMem(reinterpret_cast<hixl::MemHandle>(handle)),
                "HIXL DeregisterMem failed");
          },
          py::arg("handle"))
      .def(
          "connect",
          [](hixl::Hixl &self, const std::string &remote_engine,
             int32_t timeout_ms) {
            py::gil_scoped_release release;
            check_status(self.Connect(hixl::AscendString(remote_engine.c_str()),
                                      timeout_ms),
                         "HIXL Connect failed");
          },
          py::arg("remote_engine"), py::arg("timeout_ms") = 30000)
      .def(
          "disconnect",
          [](hixl::Hixl &self, const std::string &remote_engine,
             int32_t timeout_ms) {
            py::gil_scoped_release release;
            check_status(
                self.Disconnect(hixl::AscendString(remote_engine.c_str()),
                                timeout_ms),
                "HIXL Disconnect failed");
          },
          py::arg("remote_engine"), py::arg("timeout_ms") = 5000)
      .def(
          "transfer_sync",
          [](hixl::Hixl &self, const std::string &remote_engine,
             hixl::TransferOp op,
             const std::vector<hixl::TransferOpDesc> &op_descs,
             int32_t timeout_ms) {
            py::gil_scoped_release release;
            check_status(
                self.TransferSync(hixl::AscendString(remote_engine.c_str()), op,
                                  op_descs, timeout_ms),
                "HIXL TransferSync failed");
          },
          py::arg("remote_engine"), py::arg("op"), py::arg("op_descs"),
          py::arg("timeout_ms") = 30000)
      .def(
          "transfer_async",
          [](hixl::Hixl &self, const std::string &remote_engine,
             hixl::TransferOp op,
             const std::vector<hixl::TransferOpDesc> &op_descs) {
            hixl::TransferArgs args{};
            hixl::TransferReq req = nullptr;
            check_status(
                self.TransferAsync(hixl::AscendString(remote_engine.c_str()),
                                   op, op_descs, args, req),
                "HIXL TransferAsync failed");
            return reinterpret_cast<uintptr_t>(req);
          },
          py::arg("remote_engine"), py::arg("op"), py::arg("op_descs"))
      .def(
          "get_transfer_status",
          [](hixl::Hixl &self, uintptr_t req) {
            hixl::TransferStatus status;
            check_status(self.GetTransferStatus(
                             reinterpret_cast<hixl::TransferReq>(req), status),
                         "HIXL GetTransferStatus failed");
            return status;
          },
          py::arg("req"));
}
