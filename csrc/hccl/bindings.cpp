#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
// Force build with old ABI for now since HCCL depends on old ABI.
// Should be fine as long as we don't link to pytorch or any other libraries 
// that expose functions with std::string or std::vector as arguments
#define _GLIBCXX_USE_CXX11_ABI 0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "hccl_agent.h"

namespace py = pybind11;

// Helper to convert HcclResult to a Python exception
void check_hccl_result(HcclResult result, const std::string& error_msg) {
    if (result != HCCL_SUCCESS) {
        throw std::runtime_error(error_msg + " | HCCL Error Code: " + std::to_string(result));
    }
}

PYBIND11_MODULE(hccl_npu_comms, m) {
    m.doc() = "pybind11 wrapper for HcclAgent";

    py::class_<ServerMeta>(m, "ServerMeta")
        .def(py::init<>())
        .def_readwrite("dev_id", &ServerMeta::devId)
        .def_readwrite("ipv4_addr", &ServerMeta::ipv4Addr)
        .def_readwrite("listen_port", &ServerMeta::listenPort)
        .def_property("tag_ctrl",
            [](const ServerMeta &s) { return py::bytes(s.tagCtrl, TAG_SIZE); },
            [](ServerMeta &s, const py::bytes &b) {
                py::buffer_info info(py::buffer(b).request());
                if (static_cast<size_t>(info.size) != TAG_SIZE) throw std::runtime_error("tag_ctrl must be 32 bytes");
                memcpy(s.tagCtrl, info.ptr, TAG_SIZE);
            })
        .def_property("tag_data",
            [](const ServerMeta &s) { return py::bytes(s.tagData, TAG_SIZE); },
            [](ServerMeta &s, const py::bytes &b) {
                py::buffer_info info(py::buffer(b).request());
                if (static_cast<size_t>(info.size) != TAG_SIZE) throw std::runtime_error("tag_data must be 32 bytes");
                memcpy(s.tagData, info.ptr, TAG_SIZE);
            })
        .def(py::pickle(
            [](const ServerMeta &s) { // __getstate__
                return py::make_tuple(
                    s.devId, s.ipv4Addr, s.listenPort,
                    py::bytes(s.tagCtrl, TAG_SIZE), py::bytes(s.tagData, TAG_SIZE)
                );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 5) throw std::runtime_error("Invalid state for ServerMeta unpickling!");
                
                ServerMeta s;
                s.devId = t[0].cast<int32_t>();
                s.ipv4Addr = t[1].cast<uint32_t>();
                s.listenPort = t[2].cast<uint32_t>();
                
                py::buffer_info ctrl_info(py::buffer(t[3].cast<py::bytes>()).request());
                if (static_cast<size_t>(ctrl_info.size) != TAG_SIZE) throw std::runtime_error("Invalid tag_ctrl size in pickle state");
                memcpy(s.tagCtrl, ctrl_info.ptr, TAG_SIZE);
                
                py::buffer_info data_info(py::buffer(t[4].cast<py::bytes>()).request());
                if (static_cast<size_t>(data_info.size) != TAG_SIZE) throw std::runtime_error("Invalid tag_data size in pickle state");
                memcpy(s.tagData, data_info.ptr, TAG_SIZE);

                return s;
            }
        ));

    py::class_<ClientMeta>(m, "ClientMeta")
        .def(py::init<>())
        .def_readwrite("dev_id", &ClientMeta::devId)
        .def_readwrite("ipv4_addr", &ClientMeta::ipv4Addr)
        .def(py::pickle(
            [](const ClientMeta &c) { // __getstate__
                return py::make_tuple(c.devId, c.ipv4Addr);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state for ClientMeta unpickling!");
                
                ClientMeta c;
                c.devId = t[0].cast<int32_t>();
                c.ipv4Addr = t[1].cast<uint32_t>();
                
                return c;
            }
        ));
        
    py::class_<hccl::TransportMem::RmaMemDesc>(m, "RmaMemDesc")
        .def(py::init<>())
        .def_readwrite("local_rank_id", &hccl::TransportMem::RmaMemDesc::localRankId)
        .def_readwrite("remote_rank_id", &hccl::TransportMem::RmaMemDesc::remoteRankId)
        .def_property("mem_desc",
            [](const hccl::TransportMem::RmaMemDesc &h) { return py::bytes(h.memDesc, hccl::TRANSPORT_EMD_ESC_SIZE); },
            [](hccl::TransportMem::RmaMemDesc &h, const py::bytes &b) {
                py::buffer_info info(py::buffer(b).request());
                if (static_cast<size_t>(info.size) > hccl::TRANSPORT_EMD_ESC_SIZE) throw std::runtime_error("mem_desc data is too large");
                memcpy(h.memDesc, info.ptr, info.size);
                if (static_cast<size_t>(info.size) < hccl::TRANSPORT_EMD_ESC_SIZE) {
                    memset(h.memDesc + info.size, 0, hccl::TRANSPORT_EMD_ESC_SIZE - info.size);
                }
            })
        .def(py::pickle(
            [](const hccl::TransportMem::RmaMemDesc &h) { // __getstate__
                return py::make_tuple(h.localRankId, py::bytes(h.memDesc, hccl::TRANSPORT_EMD_ESC_SIZE));
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state for RmaMemDesc unpickling!");
                
                hccl::TransportMem::RmaMemDesc h;
                h.localRankId = t[0].cast<uint32_t>();
                
                py::buffer_info info(py::buffer(t[1].cast<py::bytes>()).request());
                if (static_cast<size_t>(info.size) != hccl::TRANSPORT_EMD_ESC_SIZE) throw std::runtime_error("Invalid mem_desc size in pickle state");
                memcpy(h.memDesc, info.ptr, info.size);
                
                return h;
            }
        ));

    py::class_<HcclWriteOp>(m, "HcclWriteOp")
        .def(py::init([](uintptr_t src, uintptr_t dst, uint64_t s) {
                return HcclWriteOp{reinterpret_cast<void*>(src), reinterpret_cast<void*>(dst), s};
            }),
            py::arg("src"), py::arg("dst"), py::arg("s")
        )
        .def_property("src_addr",
            [](const HcclWriteOp &op) { return reinterpret_cast<uintptr_t>(op.srcAddr); },
            [](HcclWriteOp &op, uintptr_t addr) { op.srcAddr = reinterpret_cast<void*>(addr); })
        .def_property("dst_addr",
            [](const HcclWriteOp &op) { return reinterpret_cast<uintptr_t>(op.dstAddr); },
            [](HcclWriteOp &op, uintptr_t addr) { op.dstAddr = reinterpret_cast<void*>(addr); })
        .def_readwrite("size", &HcclWriteOp::size);

    py::class_<HcclReadOp>(m, "HcclReadOp")
        .def(py::init([](uintptr_t src, uintptr_t dst, uint64_t s) {
                return HcclReadOp{reinterpret_cast<void*>(src), reinterpret_cast<void*>(dst), s};
            }),
            py::arg("src"), py::arg("dst"), py::arg("s")
        )
        .def_property("src_addr",
            [](const HcclReadOp &op) { return reinterpret_cast<uintptr_t>(op.srcAddr); },
            [](HcclReadOp &op, uintptr_t addr) { op.srcAddr = reinterpret_cast<void*>(addr); })
        .def_property("dst_addr",
            [](const HcclReadOp &op) { return reinterpret_cast<uintptr_t>(op.dstAddr); },
            [](HcclReadOp &op, uintptr_t addr) { op.dstAddr = reinterpret_cast<void*>(addr); })
        .def_readwrite("size", &HcclReadOp::size);

    py::class_<HcclAgent, std::shared_ptr<HcclAgent>>(m, "HcclAgent")
        .def_static("get_instance", [](uint32_t deviceId) {
            std::shared_ptr<HcclAgent> agent;
            check_hccl_result(HcclAgent::GetInstance(deviceId, agent), "Failed to get HcclAgent instance");
            return agent;
        }, py::arg("device_id"))
        .def("init", [](HcclAgent &self) { check_hccl_result(self.Init(), "HcclAgent Init failed"); })
        .def("register_mem", [](HcclAgent &self, uintptr_t dev_ptr, uint64_t size) {
            hccl::TransportMem::RmaMemDesc mem_handle{};
            check_hccl_result(self.RegisterMem(reinterpret_cast<void*>(dev_ptr), size, mem_handle), "Failed to register memory");
            return mem_handle;
        }, py::arg("dev_ptr"), py::arg("size"))
        .def("deregister_mem", [](HcclAgent &self, uintptr_t dev_ptr) { check_hccl_result(self.DeregisterMem(reinterpret_cast<void*>(dev_ptr)), "Failed to deregister memory"); }, py::arg("dev_ptr"))
        .def("import_mem", [](HcclAgent &self, uintptr_t conn_handle, const hccl::TransportMem::RmaMemDesc& remote_mem_handle) {
            check_hccl_result(self.ImportMem(reinterpret_cast<HcclConn>(conn_handle), remote_mem_handle), "Failed to import memory");
        }, py::arg("conn_handle"), py::arg("remote_mem_handle"))
        .def("get_client_meta", [](HcclAgent &self) {
            ClientMeta meta{};
            check_hccl_result(self.GetClientMeta(meta), "Failed to get client metadata");
            return meta;
        })
        .def("get_server_meta", [](HcclAgent &self) {
            ServerMeta meta{};
            check_hccl_result(self.GetServerMeta(meta), "Failed to get server metadata");
            return meta;
        })
        .def("accept", [](HcclAgent &self, const ClientMeta& client_meta, const ServerMeta& server_meta) {
            // TODO: added for now, maybe not the cleanest
            py::gil_scoped_release release;

            HcclConn conn = nullptr;
            check_hccl_result(self.Accept(client_meta, server_meta, conn), "Accept failed");
            return reinterpret_cast<uintptr_t>(conn);
        }, py::arg("client_meta"), py::arg("server_meta"))
        .def("connect", [](HcclAgent &self, const ServerMeta& server_meta) {
            // TODO: added for now, maybe not the cleanest
            py::gil_scoped_release release;
            
            HcclConn conn = nullptr;
            check_hccl_result(self.Connect(server_meta, conn), "Connect failed");
            return reinterpret_cast<uintptr_t>(conn);
        }, py::arg("server_meta"))
        .def("write_batch", [](HcclAgent &self, uintptr_t conn_handle, const std::vector<HcclWriteOp>& writes, uintptr_t stream_ptr) {
            check_hccl_result(self.WriteBatch(reinterpret_cast<HcclConn>(conn_handle), writes, reinterpret_cast<aclrtStream>(stream_ptr)), "Failed to write batch");
        }, py::arg("conn_handle"), py::arg("writes"), py::arg("stream"))
        .def("read_batch", [](HcclAgent &self, uintptr_t conn_handle, const std::vector<HcclReadOp>& reads, uintptr_t stream_ptr) {
            check_hccl_result(self.ReadBatch(reinterpret_cast<HcclConn>(conn_handle), reads, reinterpret_cast<aclrtStream>(stream_ptr)), "Failed to write batch");
        }, py::arg("conn_handle"), py::arg("writes"), py::arg("stream"));
}