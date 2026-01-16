#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
// Force build with old ABI for now since HCCL depends on old ABI.
// Should be fine as long as we don't link to pytorch or any other libraries 
// that expose functions with std::string or std::vector as arguments
#define _GLIBCXX_USE_CXX11_ABI 0

#include "hccl_utils.h"

#include <random>
#include <vector>

#include "hccl/hccl_socket.h"

void FillRandom(char* buffer, size_t length)
{
    if (length == 0 || buffer == nullptr) {
        return;
    }

    const char* characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const size_t characters_size = 62; // 10 + 26 + 26 = 62

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, characters_size - 1);

    for (size_t i = 0; i < length; ++i) {
        buffer[i] = characters[dis(gen)];
    }

    buffer[length] = '\0';
}

HcclResult GetLocalIpv4(uint32_t phyId, hccl::HcclIpAddress &localIp)
{
    std::vector<hccl::HcclIpAddress> ips;
    
    HCCL_CHECK(hrtRaGetDeviceIP(phyId, ips));

    for (const auto& ip : ips) {
        if (ip.GetFamily() == AF_INET) {
            localIp = ip;
            return HCCL_SUCCESS;
        }
    }

    return HCCL_E_NOT_FOUND;
}