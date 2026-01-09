#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <array>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_network_pub.h"
#include "hccl/hccl_common.h"
#include "hccl/hccl_socket.h"
#include "hccl/dispatcher.h"
#include "hccl/notify_pool.h"
#include "hccl/transport_mem.h"
#include "hccl/hccl_mem.h"

constexpr uint32_t MAX_LOCAL_DEVICES = 16;
constexpr uint32_t TAG_SIZE = 32;

struct ServerMeta {
    int32_t devId;
    uint32_t ipv4Addr;
    uint32_t listenPort;
    char tagCtrl[TAG_SIZE];
    char tagData[TAG_SIZE];
};

struct ClientMeta {
    int32_t devId;
    uint32_t ipv4Addr;
};

typedef void *HcclConn;

typedef struct HcclWriteOp {
    void* srcAddr;
    void *dstAddr;
    uint64_t size;
} HcclWriteOp;

typedef struct HcclReadOp {
    void* srcAddr;
    void *dstAddr;
    uint64_t size;
} HcclReadOp;

class HcclAgent {
public:
    static HcclResult GetInstance(uint32_t deviceId, std::shared_ptr<HcclAgent>& outAgent);

    HcclAgent(const HcclAgent&) = delete;
    HcclAgent& operator=(const HcclAgent&) = delete;
    HcclAgent(HcclAgent&&) = delete;
    HcclAgent& operator=(HcclAgent&&) = delete;

    ~HcclAgent();

    HcclResult Init();

    HcclResult RegisterMem(void* devPtr, uint64_t size, hccl::TransportMem::RmaMemDesc& memHandle);
    HcclResult DeregisterMem(void* devPtr);
    HcclResult ImportMem(HcclConn conn, const hccl::TransportMem::RmaMemDesc& remoteMemHandle);

    HcclResult GetClientMeta(ClientMeta& clientMeta);
    HcclResult GetServerMeta(ServerMeta& serverMeta);

    HcclResult Accept(const ClientMeta& clientMeta, const ServerMeta& serverMeta, HcclConn &conn);
    HcclResult Connect(const ServerMeta& serverMeta, HcclConn &conn);

    HcclResult WriteBatch(HcclConn conn, const std::vector<HcclWriteOp>& writes, aclrtStream stream);
    HcclResult ReadBatch(HcclConn conn, const std::vector<HcclReadOp>& reads, aclrtStream stream);
private:
    explicit HcclAgent(uint32_t devId) : devId_(devId), state_(State::CREATED) {};

    HcclResult CreateTransportMem(uint32_t remoteRankId, std::shared_ptr<hccl::TransportMem>& transportMem);
    static HcclResult ConnectSocket(std::shared_ptr<hccl::HcclSocket>& socket);

    enum class State { CREATED, INITIALIZED, ERROR };

    static std::shared_ptr<HcclAgent> instances[MAX_LOCAL_DEVICES];
    static std::mutex instanceMutex;

    uint32_t devId_;
    State state_;
    uint32_t phyId_;
    hccl::HcclIpAddress localIp_;

    HcclNetDevCtx nicNetDevCtx_{nullptr};
    std::shared_ptr<hccl::HcclSocket> nicServerSocket_;
    HcclDispatcher dispatcher_{nullptr};
    std::unique_ptr<hccl::NotifyPool> notifyPool_;

    std::mutex agentMutex_;

    std::unordered_map<HcclConn, std::shared_ptr<hccl::TransportMem>> conns_;

    struct MemInfo {
        HcclBuf handle;
        uint64_t refCount;
    };
    std::unordered_map<void*, MemInfo> registeredMems_;
};