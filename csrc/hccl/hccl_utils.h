#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_network_pub.h"
#include "hccl/hccl_types.h"

#define ACL_CHECK(answer)                                                      \
  do {                                                                         \
    aclError ret = (answer);                                                   \
    if (ret != ACL_SUCCESS) {                                                  \
      return HCCL_E_INTERNAL;                                                  \
    }                                                                          \
  } while (0)

#define HCCL_CHECK(answer)                                                     \
  do {                                                                         \
    HcclResult ret = (answer);                                                 \
    if (ret != HCCL_SUCCESS) {                                                 \
      return ret;                                                              \
    }                                                                          \
  } while (0)

// Fills a buffer with random alphanumeric characters
void FillRandom(char *buffer, size_t length);

// Retrieves the IPv4 address for a specific physical device ID
HcclResult GetLocalIpv4(uint32_t phyId, hccl::HcclIpAddress &localIp);