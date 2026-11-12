#ifndef FUSED_ROPE_BASE_H
#define FUSED_ROPE_BASE_H

#include "kernel_operator.h"

namespace FusedRope {
using namespace AscendC;
using AscendC::Duplicate;
using AscendC::HardEvent;

template <typename T> class FusedRopeBase {
public:
  __aicore__ inline FusedRopeBase(){};
  __aicore__ inline void
  InitData(uint64_t coreNumUse, uint64_t numTokens, uint64_t numHeads,
           uint64_t headSize, uint64_t rotaryDim, uint64_t kLeadingDimension,
           uint64_t isNeoxStyle, uint64_t frontCore, uint64_t tailCore,
           uint64_t numTokensFrontCoreEachLoop,
           uint64_t numTokensTailCoreEachLoop, uint64_t numTokensEachFrontCore,
           uint64_t numTokensEachTailCore, uint64_t loopTimeEachFrontCore,
           uint64_t loopTimeEachTailCore, uint64_t numTokensFrontCoreLastLoop,
           uint64_t numTokensTailCoreLastLoop);

  __aicore__ inline void SetPositionRange(bool enabled, uint64_t oldStart,
                                          uint64_t newStart, uint64_t slots) {
    usePositionRange = enabled;
    oldPositionStart = oldStart;
    newPositionStart = newStart;
    positionSlots = slots;
  }

  __aicore__ inline uint64_t OldPosition(GlobalTensor<uint64_t>& positions,
                                         uint64_t localOffset,
                                         uint64_t blockOffset) const {
    if (usePositionRange) {
      return oldPositionStart + (blockOffset + localOffset) % positionSlots;
    }
    return positions.GetValue(localOffset);
  }

  __aicore__ inline uint64_t NewPosition(GlobalTensor<uint64_t>& positions,
                                         uint64_t localOffset,
                                         uint64_t blockOffset) const {
    if (usePositionRange) {
      return newPositionStart + (blockOffset + localOffset) % positionSlots;
    }
    return positions.GetValue(localOffset);
  }

protected:
  uint32_t blockIdx_;
  uint64_t coreNumUse;
  uint64_t numTokens;
  uint64_t numHeads;
  uint64_t rotaryDim;
  uint64_t headSize;
  uint64_t kLeadingDimension;
  uint64_t frontCore;
  uint64_t tailCore;
  uint64_t numTokensEachFrontCore;
  uint64_t numTokensEachTailCore;
  uint64_t isNeoxStyle;
  uint64_t loopTimeCurrentCore{0};
  uint64_t numTokensEachLoopCurrentCore{0};
  uint64_t numTokensLastLoopCurrentCore{0};
  bool usePositionRange{false};
  uint64_t oldPositionStart{0};
  uint64_t newPositionStart{0};
  uint64_t positionSlots{1};
};

template <typename T>
__aicore__ inline void FusedRopeBase<T>::InitData(
    uint64_t coreNumUse, uint64_t numTokens, uint64_t numHeads,
    uint64_t headSize, uint64_t rotaryDim, uint64_t kLeadingDimension,
    uint64_t isNeoxStyle, uint64_t frontCore, uint64_t tailCore,
    uint64_t numTokensFrontCoreEachLoop, uint64_t numTokensTailCoreEachLoop,
    uint64_t numTokensEachFrontCore, uint64_t numTokensEachTailCore,
    uint64_t loopTimeEachFrontCore, uint64_t loopTimeEachTailCore,
    uint64_t numTokensFrontCoreLastLoop, uint64_t numTokensTailCoreLastLoop) {
  blockIdx_ = AscendC::GetBlockIdx();

  this->coreNumUse = coreNumUse;
  this->numTokens = numTokens;
  this->numHeads = numHeads;
  this->rotaryDim = rotaryDim;
  this->headSize = headSize;
  this->kLeadingDimension = kLeadingDimension;
  this->frontCore = frontCore;
  this->tailCore = tailCore;
  this->numTokensEachFrontCore = numTokensEachFrontCore;
  this->numTokensEachTailCore = numTokensEachTailCore;
  this->isNeoxStyle = isNeoxStyle;

  loopTimeCurrentCore =
      (blockIdx_ < frontCore) ? loopTimeEachFrontCore : loopTimeEachTailCore;
  numTokensEachLoopCurrentCore = (blockIdx_ < frontCore)
                                     ? numTokensFrontCoreEachLoop
                                     : numTokensTailCoreEachLoop;
  numTokensLastLoopCurrentCore = (blockIdx_ < frontCore)
                                     ? numTokensFrontCoreLastLoop
                                     : numTokensTailCoreLastLoop;
}

} // namespace FusedRope

#endif