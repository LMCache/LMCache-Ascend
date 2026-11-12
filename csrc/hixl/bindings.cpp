#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
#define _GLIBCXX_USE_CXX11_ABI 0

#include <cstdio>
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

namespace {
// C++ data only: safe to throw while the GIL is released. Python exception
// construction happens in the module-local translator after stack unwinding.
class HixlCallError : public std::runtime_error {
public:
  HixlCallError(hixl::Status code, const std::string &operation)
      : std::runtime_error("HIXL " + operation + " failed | HIXL Error Code: " +
                           std::to_string(code)),
        code(code), operation(operation) {}

  const hixl::Status code;
  const std::string operation;
};

void check_status(hixl::Status status, const std::string &operation) {
  if (status != hixl::SUCCESS) {
    throw HixlCallError(status, operation);
  }
}

void check_acl_status(aclError status, const char *operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) +
        " failed | ACL Error Code: " + std::to_string(status));
  }
}
} // namespace

PYBIND11_MODULE(hixl_npu_comms, m) {
  m.doc() = "pybind11 wrapper for HIXL low-level transfer engine";

  // Let pybind11 manage the exception type's lifetime. The custom translator
  // runs before its message-only translator and preserves the native fields.
  static auto &hixl_error = py::register_local_exception<HixlCallError>(
      m, "HixlError", PyExc_RuntimeError);
  py::register_local_exception_translator([](std::exception_ptr error) {
    if (!error) {
      return;
    }
    try {
      std::rethrow_exception(error);
    } catch (const HixlCallError &e) {
      auto instance = py::reinterpret_borrow<py::object>(hixl_error)(e.what());
      instance.attr("code") = e.code;
      instance.attr("operation") = e.operation;
      PyErr_SetObject(hixl_error.ptr(), instance.ptr());
    }
  });

  m.def(
      "get_current_device",
      [] {
        int32_t device = -1;
        check_acl_status(aclrtGetDevice(&device), "aclrtGetDevice");
        return device;
      },
      "Get the current thread's ACL logical device ID.");

  m.def(
      "get_current_context",
      []() -> py::object {
        aclrtContext context = nullptr;
        aclError result = aclrtGetCurrentContext(&context);
        if (result == ACL_ERROR_RT_CONTEXT_NULL) {
          return py::none();
        }
        check_acl_status(result, "aclrtGetCurrentContext");
        if (context == nullptr) {
          return py::none();
        }
        return py::int_(reinterpret_cast<uintptr_t>(context));
      },
      "Borrow the current thread's ACL context; return None if absent.");

  m.def(
      "set_current_context",
      [](uintptr_t context) {
        check_acl_status(
            aclrtSetCurrentContext(reinterpret_cast<aclrtContext>(context)),
            "aclrtSetCurrentContext");
      },
      py::arg("context"), "Set a borrowed ACL context on the current thread.");

  m.def(
      "is_device_memory",
      [](uintptr_t ptr) -> bool {
        rtPointerAttributes_t attributes;
        auto ret =
            rtPointerGetAttributes(&attributes, reinterpret_cast<void *>(ptr));
        if (ret != ACL_SUCCESS) {
          fprintf(stderr,
                  "[hixl] rtPointerGetAttributes(0x%lx) failed "
                  "with code %d, assuming host\n",
                  static_cast<unsigned long>(ptr), static_cast<int>(ret));
          return false;
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
                         "Initialize");
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
            hixl::Status result;
            {
              py::gil_scoped_release release;
              result = self.RegisterMem(desc, type, handle);
            }
            check_status(result, "RegisterMem");
            return reinterpret_cast<uintptr_t>(handle);
          },
          py::arg("addr"), py::arg("len"), py::arg("type"))
      .def(
          "deregister_mem",
          [](hixl::Hixl &self, uintptr_t handle) {
            hixl::Status result;
            {
              py::gil_scoped_release release;
              result =
                  self.DeregisterMem(reinterpret_cast<hixl::MemHandle>(handle));
            }
            check_status(result, "DeregisterMem");
          },
          py::arg("handle"))
      .def(
          "connect",
          [](hixl::Hixl &self, const std::string &remote_engine,
             int32_t timeout_ms) {
            hixl::Status result;
            {
              py::gil_scoped_release release;
              result = self.Connect(hixl::AscendString(remote_engine.c_str()),
                                    timeout_ms);
            }
            check_status(result, "Connect");
          },
          py::arg("remote_engine"), py::arg("timeout_ms") = 30000)
      .def(
          "disconnect",
          [](hixl::Hixl &self, const std::string &remote_engine,
             int32_t timeout_ms) {
            hixl::Status result;
            {
              py::gil_scoped_release release;
              result = self.Disconnect(
                  hixl::AscendString(remote_engine.c_str()), timeout_ms);
            }
            check_status(result, "Disconnect");
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
                "TransferSync");
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
            hixl::Status result;
            {
              py::gil_scoped_release release;
              result =
                  self.TransferAsync(hixl::AscendString(remote_engine.c_str()),
                                     op, op_descs, args, req);
            }
            check_status(result, "TransferAsync");
            return reinterpret_cast<uintptr_t>(req);
          },
          py::arg("remote_engine"), py::arg("op"), py::arg("op_descs"))
      .def(
          "get_transfer_status",
          [](hixl::Hixl &self, uintptr_t req) {
            hixl::TransferStatus status;
            hixl::Status result;
            {
              py::gil_scoped_release release;
              result = self.GetTransferStatus(
                  reinterpret_cast<hixl::TransferReq>(req), status);
            }
            check_status(result, "GetTransferStatus");
            return status;
          },
          py::arg("req"));
}
