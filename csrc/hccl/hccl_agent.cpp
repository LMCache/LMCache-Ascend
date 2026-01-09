#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
// Force build with old ABI for now since HCCL depends on old ABI.
// Should be fine as long as we don't link to pytorch or any other libraries 
// that expose functions with std::string or std::vector as arguments
#define _GLIBCXX_USE_CXX11_ABI 0

#include "hccl_agent.h"
#include "hccl_utils.h"

#include <iostream>
#include <random>
#include <cstring>
#include <thread>
#include <chrono>

#include "runtime/dev.h" 

std::shared_ptr<HcclAgent> HcclAgent::instances[MAX_LOCAL_DEVICES];
std::mutex HcclAgent::instanceMutex;

HcclResult HcclAgent::GetInstance(uint32_t deviceId, std::shared_ptr<HcclAgent>& outAgent) {
    if (deviceId >= MAX_LOCAL_DEVICES) {
        return HCCL_E_PARA;
    }
    std::lock_guard<std::mutex> lock(instanceMutex);
    if (instances[deviceId] == nullptr) {
        instances[deviceId] = std::shared_ptr<HcclAgent>(new HcclAgent(deviceId));
    }
    outAgent = instances[deviceId];
    return HCCL_SUCCESS;
}

HcclAgent::~HcclAgent() {
    std::lock_guard<std::mutex> lock(agentMutex_);
    conns_.clear();
    nicServerSocket_.reset();
    
    for (auto const& pair : registeredMems_) {
        HcclMemDereg(&pair.second.handle);
    }
    registeredMems_.clear();

    if (notifyPool_) {
        notifyPool_->Destroy();
    }
    if (dispatcher_) {
        HcclDispatcherDestroy(dispatcher_);
    }
    if (nicNetDevCtx_) {
        HcclNetCloseDev(nicNetDevCtx_);
    }
}

HcclResult HcclAgent::Init() {
    std::lock_guard<std::mutex> lock(agentMutex_);
    if (state_ == State::INITIALIZED) {
        return HCCL_SUCCESS;
    }
    if (state_ == State::ERROR) {
        return HCCL_E_INTERNAL;
    }

    ACL_CHECK(rtGetDevicePhyIdByIndex(devId_, &phyId_));
    HCCL_CHECK(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, phyId_, devId_, false));
    HCCL_CHECK(GetLocalIpv4(phyId_, localIp_));
    HCCL_CHECK(HcclNetOpenDev(&nicNetDevCtx_, NicType::DEVICE_NIC_TYPE, phyId_, devId_, localIp_));
    
    nicServerSocket_ = std::make_shared<hccl::HcclSocket>(nicNetDevCtx_, 0);
    HCCL_CHECK(nicServerSocket_->Init());
    HCCL_CHECK(nicServerSocket_->Listen());
    
    HCCL_CHECK(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, phyId_, &dispatcher_));
    
    notifyPool_ = std::make_unique<hccl::NotifyPool>();
    HCCL_CHECK(notifyPool_->Init(phyId_));

    state_ = State::INITIALIZED;
    return HCCL_SUCCESS;
}

HcclResult HcclAgent::RegisterMem(void* devPtr, uint64_t size, hccl::TransportMem::RmaMemDesc& memHandle) {
    std::lock_guard<std::mutex> lock(agentMutex_);
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;

    auto it = registeredMems_.find(devPtr);
    if (it != registeredMems_.end()) {
        it->second.refCount++;
    } else {
        HcclMem mem;
        mem.addr = devPtr;
        mem.size = size;
        mem.type = HCCL_MEM_TYPE_DEVICE;

        HcclBuf buf;
        HCCL_CHECK(HcclMemReg(nicNetDevCtx_, &mem, &buf));
        registeredMems_[devPtr] = {buf, 1};
    }

    char *desc = nullptr;
    uint64_t desc_len = 0;
    HCCL_CHECK(HcclMemExport(&registeredMems_[devPtr].handle, &desc, &desc_len));
    
    memset(memHandle.memDesc, 0, hccl::TRANSPORT_EMD_ESC_SIZE);
    memcpy(memHandle.memDesc, desc, desc_len);

    memHandle.localRankId = devId_;
    return HCCL_SUCCESS;
}

HcclResult HcclAgent::DeregisterMem(void* devPtr) {
    std::lock_guard<std::mutex> lock(agentMutex_);
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;

    auto it = registeredMems_.find(devPtr);
    if (it == registeredMems_.end()) {
        return HCCL_E_NOT_FOUND; // Not registered
    }

    it->second.refCount--;
    if (it->second.refCount == 0) {
        HCCL_CHECK(HcclMemDereg(&it->second.handle));
        registeredMems_.erase(it);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAgent::ImportMem(HcclConn conn, const hccl::TransportMem::RmaMemDesc& remoteMemHandle) {
    std::lock_guard<std::mutex> lock(agentMutex_);
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;
    
    auto it = conns_.find(conn);
    if (it == conns_.end()) {
        return HCCL_E_INTERNAL;
    }

    hccl::TransportMem::RmaMem remoteMem;
    auto transportMem = it->second;
    HCCL_CHECK(transportMem->EnableMemAccess(remoteMemHandle, remoteMem));
    return HCCL_SUCCESS;
}

HcclResult HcclAgent::GetClientMeta(ClientMeta& clientMeta) {
    std::lock_guard<std::mutex> lock(agentMutex_);
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;

    clientMeta.devId = devId_;
    clientMeta.ipv4Addr = localIp_.GetBinaryAddress().addr.s_addr;
    return HCCL_SUCCESS;
}

HcclResult HcclAgent::GetServerMeta(ServerMeta& serverMeta) {
    std::lock_guard<std::mutex> lock(agentMutex_);
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;

    serverMeta.devId = devId_;
    serverMeta.ipv4Addr = localIp_.GetBinaryAddress().addr.s_addr;
    serverMeta.listenPort = nicServerSocket_->GetLocalPort();
    FillRandom(serverMeta.tagCtrl, TAG_SIZE);
    FillRandom(serverMeta.tagData, TAG_SIZE);
    return HCCL_SUCCESS;
}

HcclResult HcclAgent::Accept(const ClientMeta& clientMeta, const ServerMeta& serverMeta, HcclConn &conn) {
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;

    std::vector<SocketWlistInfo> wlistInfoVecCtrl;
    SocketWlistInfo wlistInfoCtrl = {};
    wlistInfoCtrl.connLimit = 1;
    memcpy(&wlistInfoCtrl.tag[0], serverMeta.tagCtrl, TAG_SIZE);
    wlistInfoCtrl.remoteIp.addr.s_addr = clientMeta.ipv4Addr;
    wlistInfoVecCtrl.push_back(wlistInfoCtrl);
    std::string tagCtrl(serverMeta.tagCtrl, TAG_SIZE);

    std::shared_ptr<hccl::HcclSocket> hccl_ctrl_socket;
    HCCL_CHECK(nicServerSocket_->AddWhiteList(wlistInfoVecCtrl));
    HCCL_CHECK(nicServerSocket_->Accept(tagCtrl, hccl_ctrl_socket));

    // Accept data socket
    std::vector<SocketWlistInfo> wlistInfoVecData;
    SocketWlistInfo wlistInfoData = {};
    wlistInfoData.connLimit = 1;
    memcpy(&wlistInfoData.tag[0], serverMeta.tagData, TAG_SIZE);
    wlistInfoData.remoteIp.addr.s_addr = clientMeta.ipv4Addr;
    wlistInfoVecData.push_back(wlistInfoData);
    std::string tagData(serverMeta.tagData, TAG_SIZE);

    std::shared_ptr<hccl::HcclSocket> hccl_data_socket;
    HCCL_CHECK(nicServerSocket_->AddWhiteList(wlistInfoVecData));
    HCCL_CHECK(nicServerSocket_->Accept(tagData, hccl_data_socket));
    
    std::shared_ptr<hccl::TransportMem> transportMem;
    HCCL_CHECK(CreateTransportMem(clientMeta.devId, transportMem));
    HCCL_CHECK(transportMem->SetDataSocket(hccl_data_socket));
    HCCL_CHECK(transportMem->SetSocket(hccl_ctrl_socket));
    HCCL_CHECK(transportMem->Connect(120)); // Timeout 120 seconds

    std::lock_guard<std::mutex> lock(agentMutex_);
    conn = transportMem.get();
    auto result = conns_.emplace(conn, transportMem);
    if (!result.second) {
        // This should ideally not happen if conn pointers are unique
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAgent::Connect(const ServerMeta& serverMeta, HcclConn &conn) {
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;
    
    hccl::HcclIpAddress remoteDevIp(serverMeta.ipv4Addr);

    std::string tagCtrl(serverMeta.tagCtrl, TAG_SIZE);

    auto hccl_ctrl_socket = std::make_shared<hccl::HcclSocket>(tagCtrl, nicNetDevCtx_, remoteDevIp, serverMeta.listenPort, hccl::HcclSocketRole::SOCKET_ROLE_CLIENT);
    HCCL_CHECK(hccl_ctrl_socket->Init());
    HCCL_CHECK(hccl_ctrl_socket->Connect());
    HCCL_CHECK(ConnectSocket(hccl_ctrl_socket));

    std::string tagData(serverMeta.tagData, TAG_SIZE);

    auto hccl_data_socket = std::make_shared<hccl::HcclSocket>(tagData, nicNetDevCtx_, remoteDevIp, serverMeta.listenPort, hccl::HcclSocketRole::SOCKET_ROLE_CLIENT);
    HCCL_CHECK(hccl_data_socket->Init());
    HCCL_CHECK(hccl_data_socket->Connect());
    HCCL_CHECK(ConnectSocket(hccl_data_socket));
    
    std::shared_ptr<hccl::TransportMem> transportMem;
    HCCL_CHECK(CreateTransportMem(serverMeta.devId, transportMem));
    HCCL_CHECK(transportMem->SetDataSocket(hccl_data_socket));
    HCCL_CHECK(transportMem->SetSocket(hccl_ctrl_socket));
    HCCL_CHECK(transportMem->Connect(120));

    std::lock_guard<std::mutex> lock(agentMutex_);
    conn = transportMem.get();
    auto result = conns_.emplace(conn, transportMem);
    if (!result.second) {
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAgent::WriteBatch(HcclConn conn, const std::vector<HcclWriteOp>& writes, aclrtStream stream) {
    std::shared_ptr<hccl::TransportMem> transportMem;
    {
        std::lock_guard<std::mutex> lock(agentMutex_);
        if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;
        
        auto it = conns_.find(conn);
        if (it == conns_.end()) {
            return HCCL_E_INTERNAL;
        }
        transportMem = it->second;
    }

    for (const auto& write_op : writes) {
        hccl::TransportMem::RmaOpMem localMem;
        localMem.addr = write_op.srcAddr;
        localMem.size = write_op.size;

        hccl::TransportMem::RmaOpMem remoteMem;
        remoteMem.addr = write_op.dstAddr;
        remoteMem.size = write_op.size;
        HCCL_CHECK(transportMem->Write(remoteMem, localMem, stream));
    }
    
    HCCL_CHECK(transportMem->AddOpFence(stream));
    return HCCL_SUCCESS;
}

HcclResult HcclAgent::ReadBatch(HcclConn conn, const std::vector<HcclReadOp>& reads, aclrtStream stream) {
    std::shared_ptr<hccl::TransportMem> transportMem;
    {
        std::lock_guard<std::mutex> lock(agentMutex_);
        if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;
        
        auto it = conns_.find(conn);
        if (it == conns_.end()) {
            return HCCL_E_INTERNAL;
        }
        transportMem = it->second;
    }

    for (const auto& read_op : reads) {
        hccl::TransportMem::RmaOpMem localMem;
        localMem.addr = read_op.dstAddr;
        localMem.size = read_op.size;

        hccl::TransportMem::RmaOpMem remoteMem;
        remoteMem.addr = read_op.srcAddr;
        remoteMem.size = read_op.size;
        HCCL_CHECK(transportMem->Read(localMem, remoteMem, stream));
    }
    
    HCCL_CHECK(transportMem->AddOpFence(stream));
    return HCCL_SUCCESS;
}

HcclResult HcclAgent::CreateTransportMem(uint32_t remoteRankId, std::shared_ptr<hccl::TransportMem>& transportMem) {
    hccl::TransportMem::AttrInfo attrInfo; 
    attrInfo.localRankId = devId_;
    attrInfo.remoteRankId = remoteRankId;
    attrInfo.sdid = 0xFFFFFFFF;
    attrInfo.serverId = 0; // Always 0 at the moment
    // TODO: make configurable
    attrInfo.trafficClass = 132;
    attrInfo.serviceLevel = 4;
    
    transportMem = hccl::TransportMem::Create(hccl::TransportMem::TpType::ROCE, notifyPool_, nicNetDevCtx_, dispatcher_, attrInfo);
    if (transportMem == nullptr) {
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAgent::ConnectSocket(std::shared_ptr<hccl::HcclSocket>& socket) {
    hccl::HcclSocketStatus status;
    const int timeout_ms = 120000;
    const int sleep_ms = 10;
    int elapsed_ms = 0;

    do {
        status = socket->GetStatus();
        if (status == hccl::HcclSocketStatus::SOCKET_OK) {
            return HCCL_SUCCESS;
        }
        if (status == hccl::HcclSocketStatus::SOCKET_ERROR) {
            return HCCL_E_INTERNAL;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        elapsed_ms += sleep_ms;
    } while (elapsed_ms < timeout_ms);

    return HCCL_E_TIMEOUT;
}