// SPDX-License-Identifier: Apache-2.0
#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
#define _GLIBCXX_USE_CXX11_ABI 0

#include <cstdio>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_one_sided_services.h"
#include "hccl/hccl_types.h"
#include "hcomm_devva.h"
#include "runtime/mem.h"
#include "securec.h"

extern "C" {
HcclResult
HcclCommInitClusterInfoMemConfig(const char *clusterInfo, uint32_t rank,
                                 HcclCommConfig *config, HcclComm *comm)
    __attribute__((weak));
}

namespace py = pybind11;

static void check_hccl(HcclResult ret, const char *msg) {
  if (ret != HCCL_SUCCESS) {
    throw std::runtime_error(std::string(msg) +
                             " | HCCL error code: " + std::to_string(ret));
  }
}

PYBIND11_MODULE(hcomm_onesided, m) {
  m.doc() = "pybind11 wrapper for hcomm one-sided service API";

  m.def(
      "is_device_memory",
      [](uintptr_t ptr) -> bool {
        rtPointerAttributes_t attributes;
        auto ret =
            rtPointerGetAttributes(&attributes, reinterpret_cast<void *>(ptr));
        if (ret != ACL_SUCCESS) {
          fprintf(stderr,
                  "[hcomm_onesided] rtPointerGetAttributes(0x%lx) failed "
                  "with code %d, assuming host\n",
                  static_cast<unsigned long>(ptr), static_cast<int>(ret));
          return false;
        }
        bool is_dev = attributes.memoryType != RT_MEMORY_TYPE_HOST &&
                      attributes.memoryType != RT_MEMORY_TYPE_USER;
        return is_dev;
      },
      py::arg("ptr"),
      "Detect whether a pointer refers to device or host memory.");

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

  m.def(
      "get_device_info",
      [](int32_t logic_device_id) -> py::dict {
        int32_t phy_device_id = 0;
        auto ret =
            aclrtGetPhyDevIdByLogicDevId(logic_device_id, &phy_device_id);
        if (ret != ACL_SUCCESS) {
          throw std::runtime_error(
              "aclrtGetPhyDevIdByLogicDevId failed | code: " +
              std::to_string(ret));
        }

        int64_t super_device_id = 0;
        ret = aclrtGetDeviceInfo(static_cast<uint32_t>(logic_device_id),
                                 ACL_DEV_ATTR_SUPER_POD_DEVIDE_ID,
                                 &super_device_id);
        if (ret != ACL_SUCCESS) {
          fprintf(stderr,
                  "[hcomm_onesided] aclrtGetDeviceInfo"
                  "(SUPER_POD_DEVIDE_ID) failed code=%d, defaulting to 0\n",
                  static_cast<int>(ret));
          super_device_id = 0;
        }

        int64_t super_pod_id = 0;
        ret = aclrtGetDeviceInfo(static_cast<uint32_t>(logic_device_id),
                                 ACL_DEV_ATTR_SUPER_POD_ID, &super_pod_id);
        if (ret != ACL_SUCCESS) {
          fprintf(stderr,
                  "[hcomm_onesided] aclrtGetDeviceInfo"
                  "(SUPER_POD_ID) failed code=%d, defaulting to 0\n",
                  static_cast<int>(ret));
          super_pod_id = 0;
        }

        const char *soc = aclrtGetSocName();

        py::dict result;
        result["phy_device_id"] = phy_device_id;
        result["super_device_id"] = super_device_id;
        result["super_pod_id"] = super_pod_id;
        result["soc_name"] = soc ? std::string(soc) : std::string();
        return result;
      },
      py::arg("logic_device_id"),
      "Query physical device ID, super-device ID and super-pod ID "
      "for a given logical device.");

  m.def(
      "get_root_info",
      []() -> py::bytes {
        HcclRootInfo info;
        check_hccl(HcclGetRootInfo(&info), "HcclGetRootInfo failed");
        return py::bytes(info.internal, HCCL_ROOT_INFO_BYTES);
      },
      "Generate an HcclRootInfo on the root rank and return it as raw bytes.");

  m.def(
      "init_comm",
      [](uint32_t n_ranks, py::bytes root_info_bytes, uint32_t rank,
         const std::string &comm_name) -> uintptr_t {
        std::string ri = root_info_bytes;
        if (ri.size() != HCCL_ROOT_INFO_BYTES) {
          throw std::runtime_error("root_info_bytes must be exactly " +
                                   std::to_string(HCCL_ROOT_INFO_BYTES) +
                                   " bytes");
        }

        HcclRootInfo info;
        if (memcpy_s(info.internal, sizeof(info.internal), ri.data(),
                     HCCL_ROOT_INFO_BYTES) != EOK) {
          throw std::runtime_error("memcpy_s failed copying root info");
        }

        HcclComm comm = nullptr;
        {
          py::gil_scoped_release release;
          check_hccl(HcclCommInitRootInfo(n_ranks, &info, rank, &comm),
                     "HcclCommInitRootInfo failed");
        }
        return reinterpret_cast<uintptr_t>(comm);
      },
      py::arg("n_ranks"), py::arg("root_info_bytes"), py::arg("rank"),
      py::arg("comm_name") = "",
      "Initialize an HcclComm from root info. Returns an opaque handle.");

  m.def(
      "init_comm_cluster_info",
      [](const std::string &cluster_json, uint32_t rank,
         const std::string &comm_name) -> uintptr_t {
        if (HcclCommInitClusterInfoMemConfig == nullptr) {
          throw std::runtime_error(
              "HcclCommInitClusterInfoMemConfig is not available in "
              "the loaded libhcomm.so");
        }
        HcclCommConfig config;
        HcclCommConfigInit(&config);

        if (comm_name.empty() || comm_name.size() >= COMM_NAME_MAX_LENGTH) {
          throw std::runtime_error(
              "comm_name must be non-empty and < 128 chars");
        }
        std::snprintf(config.hcclCommName, sizeof(config.hcclCommName), "%s",
                      comm_name.c_str());

        HcclComm comm = nullptr;
        {
          py::gil_scoped_release release;
          check_hccl(HcclCommInitClusterInfoMemConfig(cluster_json.c_str(),
                                                      rank, &config, &comm),
                     "HcclCommInitClusterInfoMemConfig failed");
        }
        return reinterpret_cast<uintptr_t>(comm);
      },
      py::arg("cluster_json"), py::arg("rank"), py::arg("comm_name"),
      "Initialize an HcclComm from a rank-table JSON string (in-memory). "
      "comm_name is required and used as the communicator identifier.");

  m.def(
      "destroy_comm",
      [](uintptr_t comm_handle) {
        HcclComm comm = reinterpret_cast<HcclComm>(comm_handle);
        check_hccl(HcclCommDestroy(comm), "HcclCommDestroy failed");
      },
      py::arg("comm"), "Destroy an HcclComm.");

  m.def(
      "register_global_mem",
      [](uintptr_t addr, uint64_t size, bool is_device) -> uintptr_t {
        HcclMem mem;
        mem.type = is_device ? HCCL_MEM_TYPE_DEVICE : HCCL_MEM_TYPE_HOST;
        mem.addr = reinterpret_cast<void *>(addr);
        mem.size = size;
        void *handle = nullptr;
        check_hccl(HcclRegisterGlobalMem(&mem, &handle),
                   "HcclRegisterGlobalMem failed");
        return reinterpret_cast<uintptr_t>(handle);
      },
      py::arg("addr"), py::arg("size"), py::arg("is_device"),
      "Register memory at process level. Returns a mem handle.");

  m.def(
      "deregister_global_mem",
      [](uintptr_t mem_handle) {
        check_hccl(
            HcclDeregisterGlobalMem(reinterpret_cast<void *>(mem_handle)),
            "HcclDeregisterGlobalMem failed");
      },
      py::arg("mem_handle"), "Deregister memory at process level.");

  m.def(
      "bind_mem",
      [](uintptr_t comm_handle, uintptr_t mem_handle) {
        HcclComm comm = reinterpret_cast<HcclComm>(comm_handle);
        check_hccl(HcclCommBindMem(comm, reinterpret_cast<void *>(mem_handle)),
                   "HcclCommBindMem failed");
      },
      py::arg("comm"), py::arg("mem_handle"),
      "Bind a registered memory handle to a communicator.");

  m.def(
      "unbind_mem",
      [](uintptr_t comm_handle, uintptr_t mem_handle) {
        HcclComm comm = reinterpret_cast<HcclComm>(comm_handle);
        check_hccl(
            HcclCommUnbindMem(comm, reinterpret_cast<void *>(mem_handle)),
            "HcclCommUnbindMem failed");
      },
      py::arg("comm"), py::arg("mem_handle"),
      "Unbind a memory handle from a communicator.");

  m.def(
      "prepare",
      [](uintptr_t comm_handle, int timeout) {
        HcclComm comm = reinterpret_cast<HcclComm>(comm_handle);
        HcclPrepareConfig config;
        config.topoType = HCCL_TOPO_FULLMESH;
        config.rsvd0 = 0;
        config.rsvd1 = 0;
        config.rsvd2 = 0;
        {
          py::gil_scoped_release release;
          check_hccl(HcclCommPrepare(comm, &config, timeout),
                     "HcclCommPrepare failed");
        }
      },
      py::arg("comm"), py::arg("timeout") = 120,
      "Run full-mesh prepare (blocking collective, GIL released).");

  py::class_<HcclOneSideOpDesc>(m, "OpDesc")
      .def(py::init([](uintptr_t local_addr, uintptr_t remote_addr,
                       uint64_t num_bytes) {
             HcclOneSideOpDesc d;
             d.localAddr = reinterpret_cast<void *>(local_addr);
             d.remoteAddr = reinterpret_cast<void *>(remote_addr);
             d.count = num_bytes;
             d.dataType = HCCL_DATA_TYPE_UINT8;
             return d;
           }),
           py::arg("local_addr"), py::arg("remote_addr"), py::arg("num_bytes"));

  m.def(
      "batch_put",
      [](uintptr_t comm_handle, uint32_t remote_rank,
         std::vector<HcclOneSideOpDesc> &descs, uintptr_t stream) {
        HcclComm comm = reinterpret_cast<HcclComm>(comm_handle);
        rtStream_t s = reinterpret_cast<rtStream_t>(stream);
        check_hccl(HcclBatchPut(comm, remote_rank, descs.data(),
                                static_cast<uint32_t>(descs.size()), s),
                   "HcclBatchPut failed");
      },
      py::arg("comm"), py::arg("remote_rank"), py::arg("op_descs"),
      py::arg("stream"),
      "Submit a batch of one-sided PUT (write) operations on the given "
      "stream.");

  m.def(
      "batch_get",
      [](uintptr_t comm_handle, uint32_t remote_rank,
         std::vector<HcclOneSideOpDesc> &descs, uintptr_t stream) {
        HcclComm comm = reinterpret_cast<HcclComm>(comm_handle);
        rtStream_t s = reinterpret_cast<rtStream_t>(stream);
        check_hccl(HcclBatchGet(comm, remote_rank, descs.data(),
                                static_cast<uint32_t>(descs.size()), s),
                   "HcclBatchGet failed");
      },
      py::arg("comm"), py::arg("remote_rank"), py::arg("op_descs"),
      py::arg("stream"),
      "Submit a batch of one-sided GET (read) operations on the given "
      "stream.");

  m.attr("ROOT_INFO_BYTES") = HCCL_ROOT_INFO_BYTES;
}
