#ifndef FUSED_ROPE_BASE_H
#define FUSED_ROPE_BASE_H

#include "kernel_operator.h"

namespace FusedRope {
using namespace AscendC;
using AscendC::Duplicate;
using AscendC::HardEvent;

template <typename T>
class FusedRopeBase
{
public:
    __aicore__ inline FusedRopeBase(){};
    __aicore__ inline void InitData(uint64_t coreNumUse, uint64_t numTokens,
                    uint64_t numHeads, uint64_t headSize, uint64_t rotaryDim,
                    uint64_t kLeadingDimension, uint64_t isNeoxStyle,
                    uint64_t frontCore, uint64_t tailCore,
                    uint64_t numTokensFrontCoreEachLoop, uint64_t numTokensTailCoreEachLoop,
                    uint64_t numTokensEachFrontCore, uint64_t numTokensEachTailCore,
                    uint64_t loopTimeEachFrontCore, uint64_t loopTimeEachTailCore,
                    uint64_t numTokensFrontCoreLastLoop, uint64_t numTokensTailCoreLastLoop);

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
};

template <typename T>
__aicore__ inline void FusedRopeBase<T>::InitData(uint64_t coreNumUse, uint64_t numTokens,
                    uint64_t numHeads, uint64_t headSize, uint64_t rotaryDim,
                    uint64_t kLeadingDimension, uint64_t isNeoxStyle,
                    uint64_t frontCore, uint64_t tailCore,
                    uint64_t numTokensFrontCoreEachLoop, uint64_t numTokensTailCoreEachLoop,
                    uint64_t numTokensEachFrontCore, uint64_t numTokensEachTailCore,
                    uint64_t loopTimeEachFrontCore, uint64_t loopTimeEachTailCore,
                    uint64_t numTokensFrontCoreLastLoop, uint64_t numTokensTailCoreLastLoop){
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

    loopTimeCurrentCore = (blockIdx_ < frontCore) ? loopTimeEachFrontCore : loopTimeEachTailCore;
    numTokensEachLoopCurrentCore = (
        blockIdx_ < frontCore) ? numTokensFrontCoreEachLoop : numTokensTailCoreEachLoop;
    numTokensLastLoopCurrentCore = (
        blockIdx_ < frontCore) ? numTokensFrontCoreLastLoop : numTokensTailCoreLastLoop;
}

}

#endif