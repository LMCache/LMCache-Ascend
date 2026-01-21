/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: rma buffer common interface
 */

#ifndef HCCL_RMA_BUFFER_H
#define HCCL_RMA_BUFFER_H

#include <memory>

#include "hccl/hccl_common.h"
#include "hccl/transport_mem.h"

namespace hccl {
class RmaBuffer {
public:
  void *GetAddr() const { return addr; }

  uint64_t GetSize() const { return size; }

  RmaMemType GetMemType() const { return memType; }

  void *GetDevAddr() const { return devAddr; }

  const HcclNetDevCtx GetNetDevCtx() const { return netDevCtx; }

protected:
  virtual ~RmaBuffer() = default;
  const HcclNetDevCtx netDevCtx{nullptr};
  void *addr{nullptr};
  u64 size{0};
  void *devAddr{nullptr};
  RmaMemType memType{RmaMemType::TYPE_NUM};

  RmaBuffer(const RmaBuffer &) = delete;
  RmaBuffer &operator=(const RmaBuffer &) = delete;
};
} // namespace hccl
#endif //  HCCL_RMA_BUFFER_H