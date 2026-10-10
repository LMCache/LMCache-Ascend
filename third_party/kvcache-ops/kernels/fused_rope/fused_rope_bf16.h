#ifndef FUSED_ROPE_BF16_H
#define FUSED_ROPE_BF16_H

#include "fused_rope_base.h"

namespace FusedRope {
using namespace AscendC;

template <typename T>
class FusedRopeFP16 : public FusedRopeBase<T>
{
public:
    __aicore__ inline FusedRopeFP16(){};
    __aicore__ inline void Init(
        GM_ADDR oldPosition, GM_ADDR newPosition, GM_ADDR keyIn, GM_ADDR cosSinCache, GM_ADDR keyOut,
        uint64_t coreNumUse, uint64_t numTokens, uint64_t numHeads,
        uint64_t headSize, uint64_t rotaryDim, uint64_t kLeadingDimension,
        uint64_t isNeoxStyle, uint64_t frontCore, uint64_t tailCore,
        uint64_t numTokensFrontCoreEachLoop, uint64_t numTokensTailCoreEachLoop,
        uint64_t numTokensEachFrontCore, uint64_t numTokensEachTailCore,
        uint64_t loopTimeEachFrontCore, uint64_t loopTimeEachTailCore,
        uint64_t numTokensFrontCoreLastLoop, uint64_t numTokensTailCoreLastLoop, TPipe* pipe);

    __aicore__ inline void Process();
    __aicore__ inline void Compute(uint64_t index, uint64_t loopN);

protected:

    __aicore__ inline void ReverseRope(
        uint64_t index, uint64_t loopN, LocalTensor<float>& inLocal,
        LocalTensor<float>& reverseQ, LocalTensor<float>& cosSin,
        LocalTensor<float>& negOne, LocalTensor<float>& inCosSin,
        LocalTensor<T>& inQueueCosSinCacheBeforeCastLocal,
        GlobalTensor<uint64_t>& oldPositionIdGM, GlobalTensor<T>& cosSinCacheGM,
        uint32_t* dstShape, uint32_t* srcShape, uint32_t* dstShape4Negone);

    __aicore__ inline void Rope(
        uint64_t index, uint64_t loopN, LocalTensor<float>& inLocal,
        LocalTensor<float>& reverseQ, LocalTensor<float>& cosSin,
        LocalTensor<float>& oneNeg, LocalTensor<float>& inCosSin,
        LocalTensor<T>& inQueueCosSinCacheBeforeCastLocal, LocalTensor<T> inQueCalLocal,
        LocalTensor<float>& temp1Local, LocalTensor<uint32_t>& offsetLocal,
        GlobalTensor<uint64_t>& newPositionIdGM, GlobalTensor<T>& cosSinCacheGM,
        uint32_t* dstShape, uint32_t* srcShape, uint32_t* dstShape4Negone);

    static constexpr uint64_t BLOCK_SIZE = 32;
    static constexpr uint64_t BUFFER_NUM = 1;
    static constexpr uint64_t ELE_NUM_FP32 = 8;
    static constexpr uint64_t ELE_NUM_FP16 = 16;
    static constexpr uint64_t MASK = 64;
    uint64_t blockOffset;
    uint16_t headBlockLen{0};
    uint16_t rotaryBlockLen{0};
    uint16_t calBlockLen{0};
    uint16_t calBlockLenFP16{0};
    uint64_t kSize;
    uint64_t numHeadsMax;

    GlobalTensor<uint64_t> oldPositionIdGM;
    GlobalTensor<uint64_t> newPositionIdGM;
    GlobalTensor<T> keyInGM;
    GlobalTensor<T> cosSinCacheGM;
    GlobalTensor<T> keyGM;

    TBuf<TPosition::VECCALC> inQQue, inQueueCosSinCache;
    TBuf<TPosition::VECCALC> outQue;
    TBuf<TPosition::VECCALC> reverseBuf, negOneBuf, oneNegBuf, cosSinBuf, temp1, offsetBuf, inQueCalBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQQueBeforeCast, inQueueCosSinCacheBeforeCast;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueAfterCast;
};


template <typename T>
__aicore__ inline void FusedRopeFP16<T>::Init(
        GM_ADDR oldPositionId, GM_ADDR newPositionId, GM_ADDR keyIn, GM_ADDR cosSinCache, GM_ADDR keyOut, 
        uint64_t coreNumUse, uint64_t numTokens, uint64_t numHeads,
        uint64_t headSize, uint64_t rotaryDim, uint64_t kLeadingDimension,
        uint64_t isNeoxStyle, uint64_t frontCore, uint64_t tailCore,
        uint64_t numTokensFrontCoreEachLoop, uint64_t numTokensTailCoreEachLoop,
        uint64_t numTokensEachFrontCore, uint64_t numTokensEachTailCore,
        uint64_t loopTimeEachFrontCore, uint64_t loopTimeEachTailCore,
        uint64_t numTokensFrontCoreLastLoop, uint64_t numTokensTailCoreLastLoop, TPipe* pipe)
{

    this->InitData(coreNumUse, numTokens, numHeads, headSize, rotaryDim,
                    kLeadingDimension, isNeoxStyle, frontCore, tailCore,
                    numTokensFrontCoreEachLoop, numTokensTailCoreEachLoop,
                    numTokensEachFrontCore, numTokensEachTailCore,
                    loopTimeEachFrontCore, loopTimeEachTailCore,
                    numTokensFrontCoreLastLoop, numTokensTailCoreLastLoop);

    headBlockLen = static_cast<uint16_t>(this->headSize / ELE_NUM_FP32);
    rotaryBlockLen = static_cast<uint16_t>(this->rotaryDim / ELE_NUM_FP32);
    calBlockLenFP16 = static_cast<uint16_t>(this->rotaryDim / 2 / ELE_NUM_FP16);
    calBlockLen = rotaryBlockLen / 2;

    if (this->blockIdx_ < this->frontCore) {
        blockOffset = this->numTokensEachFrontCore * this->blockIdx_;
    } else {
        blockOffset = this->numTokensEachFrontCore * (this->frontCore) +
                      (this->blockIdx_ - this->frontCore) * this->numTokensEachTailCore;
    }

    oldPositionIdGM.SetGlobalBuffer((__gm__ uint64_t*)oldPositionId + blockOffset);
    newPositionIdGM.SetGlobalBuffer((__gm__ uint64_t*)newPositionId + blockOffset);
    keyInGM.SetGlobalBuffer((__gm__ T*)keyIn + blockOffset * this->kLeadingDimension);
    cosSinCacheGM.SetGlobalBuffer((__gm__ T*)cosSinCache);
    keyGM.SetGlobalBuffer((__gm__ T*)keyOut + blockOffset * this->numHeads * this->headSize);
    numHeadsMax = this->numHeads;
    
    pipe->InitBuffer(
        inQQue, this->numTokensEachLoopCurrentCore * numHeadsMax * this->rotaryDim * sizeof(float));
    pipe->InitBuffer(inQueueCosSinCache, this->rotaryDim * sizeof(float));
    pipe->InitBuffer(
        reverseBuf, this->numTokensEachLoopCurrentCore * numHeadsMax * this->rotaryDim * sizeof(float));
    pipe->InitBuffer(
        negOneBuf, this->numTokensEachLoopCurrentCore * numHeadsMax * this->rotaryDim * sizeof(float));
    pipe->InitBuffer(
        oneNegBuf, this->numTokensEachLoopCurrentCore * numHeadsMax * this->rotaryDim * sizeof(float));
    pipe->InitBuffer(
        cosSinBuf, this->numTokensEachLoopCurrentCore * numHeadsMax * this->rotaryDim * sizeof(float));
    pipe->InitBuffer(
        inQueCalBuf, this->numTokensEachLoopCurrentCore * numHeadsMax * this->rotaryDim * sizeof(T));
    pipe->InitBuffer(
        inQQueBeforeCast, BUFFER_NUM,
        this->numTokensEachLoopCurrentCore * numHeadsMax * this->headSize * sizeof(T));
    pipe->InitBuffer(inQueueCosSinCacheBeforeCast, BUFFER_NUM, this->rotaryDim * sizeof(T));
    pipe->InitBuffer(
        outQueAfterCast, BUFFER_NUM,
        this->numTokensEachLoopCurrentCore * numHeadsMax * this->headSize * sizeof(T));

    if (this->isNeoxStyle == 0) {
        // GPT-J style
        pipe->InitBuffer(offsetBuf, this->rotaryDim * sizeof(uint32_t));
        pipe->InitBuffer(
            temp1, this->numTokensEachLoopCurrentCore * numHeadsMax * this->rotaryDim * sizeof(float));
    } else {
        pipe->InitBuffer(offsetBuf, 0 * sizeof(uint32_t));
        pipe->InitBuffer(temp1, 0 * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void FusedRopeFP16<T>::ReverseRope(
        uint64_t index, uint64_t loopN, LocalTensor<float>& inLocal,
        LocalTensor<float>& reverseQ, LocalTensor<float>& cosSin,
        LocalTensor<float>& negOne, LocalTensor<float>& inCosSin,
        LocalTensor<T>& inQueueCosSinCacheBeforeCastLocal,
        GlobalTensor<uint64_t>& oldPositionIdGM, GlobalTensor<T>& cosSinCacheGM,
        uint32_t* dstShape, uint32_t* srcShape, uint32_t* dstShape4Negone) 
{
    // x_half
    DataCopy(
        reverseQ, inLocal[this->rotaryDim / 2],
        {static_cast<uint16_t>(loopN * this->numHeads), calBlockLen, calBlockLen, calBlockLen});
    PipeBarrier<PIPE_ALL>();
    DataCopy(
        reverseQ[this->rotaryDim / 2], inLocal,
        {static_cast<uint16_t>(loopN * this->numHeads), calBlockLen, calBlockLen, calBlockLen});
    PipeBarrier<PIPE_ALL>();

    // reverse rope
    // [1.0, ..., 1.0, -1.0, ..., -1.0]
    float one = 1.0;
    float none = -1.0;
    Duplicate<float>(negOne, one, this->rotaryDim / 2);
    Duplicate<float>(negOne[this->rotaryDim / 2], none, this->rotaryDim / 2);
    Broadcast<float, 2, 0, false>(negOne[this->rotaryDim], negOne, dstShape4Negone, srcShape);
    PipeBarrier<PIPE_ALL>();

    // old cos
    uint64_t localStartAddr = 0;
    for (uint32_t i = 0; i < loopN; ++i) {
        uint64_t offsetPos = this->numTokensEachLoopCurrentCore * index + i;
        uint64_t pos = oldPositionIdGM.GetValue(offsetPos);
        PipeBarrier<PIPE_ALL>();
        DataCopy(
            inQueueCosSinCacheBeforeCastLocal, cosSinCacheGM[pos * this->rotaryDim],
            {1, calBlockLenFP16, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(
            inCosSin, inQueueCosSinCacheBeforeCastLocal, AscendC::RoundMode::CAST_NONE,
            static_cast<uint16_t>(this->rotaryDim));
        PipeBarrier<PIPE_ALL>();
        DataCopy(inCosSin[this->rotaryDim / 2], inCosSin, {1, calBlockLen, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Broadcast<float, 2, 0, false>(cosSin[localStartAddr], inCosSin, dstShape, srcShape);
        localStartAddr += this->numHeads * this->rotaryDim;
    }

    PipeBarrier<PIPE_ALL>();
    Mul(inLocal, cosSin, inLocal, loopN * this->numHeads * this->rotaryDim); // x × cosθ_old
    PipeBarrier<PIPE_ALL>();
    Mul(reverseQ, negOne, reverseQ, loopN * this->numHeads * this->rotaryDim);  // x_half × (-1)
    PipeBarrier<PIPE_ALL>();

    // old sin
    localStartAddr = 0;
    for (uint32_t i = 0; i < loopN; ++i) {
        uint64_t offsetPos = this->numTokensEachLoopCurrentCore * index + i;
        uint64_t pos = oldPositionIdGM.GetValue(offsetPos);
        PipeBarrier<PIPE_ALL>();
        DataCopy(
            inQueueCosSinCacheBeforeCastLocal, cosSinCacheGM[pos * this->rotaryDim + this->rotaryDim / 2],
            {1, calBlockLenFP16, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(
            inCosSin, inQueueCosSinCacheBeforeCastLocal, AscendC::RoundMode::CAST_NONE,
            static_cast<uint16_t>(this->rotaryDim));
        PipeBarrier<PIPE_ALL>();
        DataCopy(inCosSin[this->rotaryDim / 2], inCosSin, {1, calBlockLen, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Broadcast<float, 2, 0, false>(cosSin[localStartAddr], inCosSin, dstShape, srcShape);
        localStartAddr += this->numHeads * this->rotaryDim;
    }
    PipeBarrier<PIPE_ALL>();
    Mul(reverseQ, cosSin, reverseQ, loopN * this->numHeads * this->rotaryDim); // x_half × (-1) × sinθ_old
    PipeBarrier<PIPE_ALL>();
    Add(inLocal, reverseQ, inLocal, loopN * this->numHeads * this->rotaryDim);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void FusedRopeFP16<T>::Rope(
        uint64_t index, uint64_t loopN, LocalTensor<float>& inLocal,
        LocalTensor<float>& reverseQ, LocalTensor<float>& cosSin,
        LocalTensor<float>& oneNeg, LocalTensor<float>& inCosSin,
        LocalTensor<T>& inQueueCosSinCacheBeforeCastLocal, LocalTensor<T> inQueCalLocal,
        LocalTensor<float>& temp1Local, LocalTensor<uint32_t>& offsetLocal,
        GlobalTensor<uint64_t>& newPositionIdGM, GlobalTensor<T>& cosSinCacheGM,
        uint32_t* dstShape, uint32_t* srcShape, uint32_t* dstShape4Negone) 
{
    // x_half
    DataCopy(
        reverseQ, inLocal[this->rotaryDim / 2],
        {static_cast<uint16_t>(loopN * this->numHeads), calBlockLen, calBlockLen, calBlockLen});
    PipeBarrier<PIPE_ALL>();
    DataCopy(
        reverseQ[this->rotaryDim / 2], inLocal,
        {static_cast<uint16_t>(loopN * this->numHeads), calBlockLen, calBlockLen, calBlockLen});
    PipeBarrier<PIPE_ALL>();

    // rope
    // [-1.0, ..., -1.0, 1.0, ..., 1.0]
    float one = 1.0;
    float none = -1.0;
    Duplicate<float>(oneNeg, none, this->rotaryDim / 2);
    Duplicate<float>(oneNeg[this->rotaryDim / 2], one, this->rotaryDim / 2);
    Broadcast<float, 2, 0, false>(oneNeg[this->rotaryDim], oneNeg, dstShape4Negone, srcShape);
    PipeBarrier<PIPE_ALL>();

    // new cos
    uint64_t localStartAddr = 0;
    for (uint32_t i = 0; i < loopN; ++i) {
        uint64_t offsetPos = this->numTokensEachLoopCurrentCore * index + i;
        uint64_t pos = newPositionIdGM.GetValue(offsetPos);
        PipeBarrier<PIPE_ALL>();
        DataCopy(
            inQueueCosSinCacheBeforeCastLocal, cosSinCacheGM[pos * this->rotaryDim],
            {1, calBlockLenFP16, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(
            inCosSin, inQueueCosSinCacheBeforeCastLocal, AscendC::RoundMode::CAST_NONE,
            static_cast<uint16_t>(this->rotaryDim));
        PipeBarrier<PIPE_ALL>();
        DataCopy(inCosSin[this->rotaryDim / 2], inCosSin, {1, calBlockLen, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Broadcast<float, 2, 0, false>(cosSin[localStartAddr], inCosSin, dstShape, srcShape);
        localStartAddr += this->numHeads * this->rotaryDim;
    }

    Mul(inLocal, cosSin, inLocal, loopN * this->numHeads * this->rotaryDim);
    PipeBarrier<PIPE_V>();
    Mul(reverseQ, oneNeg, reverseQ, loopN * this->numHeads * this->rotaryDim);
    PipeBarrier<PIPE_V>();

    // new sin
    localStartAddr = 0;
    for (uint32_t i = 0; i < loopN; ++i) {
        uint64_t offsetPos = this->numTokensEachLoopCurrentCore * index + i;
        uint64_t pos = newPositionIdGM.GetValue(offsetPos);
        PipeBarrier<PIPE_ALL>();
        DataCopy(
            inQueueCosSinCacheBeforeCastLocal, cosSinCacheGM[pos * this->rotaryDim + this->rotaryDim / 2],
            {1, calBlockLenFP16, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(
            inCosSin, inQueueCosSinCacheBeforeCastLocal, AscendC::RoundMode::CAST_NONE,
            static_cast<uint16_t>(this->rotaryDim));
        PipeBarrier<PIPE_V>();
        DataCopy(inCosSin[this->rotaryDim / 2], inCosSin, {1, calBlockLen, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Broadcast<float, 2, 0, false>(cosSin[localStartAddr], inCosSin, dstShape, srcShape);
        localStartAddr += this->numHeads * this->rotaryDim;
    }

    PipeBarrier<PIPE_V>();
    Mul(reverseQ, cosSin, reverseQ, loopN * this->numHeads * this->rotaryDim);
    PipeBarrier<PIPE_V>();
    Add(inLocal, reverseQ, inLocal, loopN * this->numHeads * this->rotaryDim);
    PipeBarrier<PIPE_V>();

    if (this->isNeoxStyle == 0) {
        for (uint32_t i = 0; i < this->rotaryDim / 2; i++) {
            offsetLocal.SetValue(i * 2, i * 4);
            offsetLocal.SetValue(i * 2 + 1, (this->rotaryDim / 2 + i) * 4);
        }

        for (uint32_t i = 0; i < loopN * this->numHeads; i++) {
            Gather(
                temp1Local[i * this->rotaryDim], inLocal[i * this->rotaryDim], offsetLocal, (uint32_t)0,
                this->rotaryDim);
            PipeBarrier<PIPE_V>();
        }
        #if ASCEND_AICORE_ARCH >= 220
            Cast(
                inQueCalLocal, temp1Local, AscendC::RoundMode::CAST_RINT,
                static_cast<uint16_t>(loopN * this->numHeads * this->rotaryDim));
        #else
            Cast(
                inQueCalLocal, temp1Local, AscendC::RoundMode::CAST_NONE,
                static_cast<uint16_t>(loopN * this->numHeads * this->rotaryDim));
        #endif
        PipeBarrier<PIPE_V>();
    } else {
        #if ASCEND_AICORE_ARCH >= 220
            Cast(
                inQueCalLocal, inLocal, AscendC::RoundMode::CAST_RINT,
                static_cast<uint16_t>(loopN * this->numHeads * this->rotaryDim));
        #else
            Cast(
                inQueCalLocal, inLocal, AscendC::RoundMode::CAST_NONE,
                static_cast<uint16_t>(loopN * this->numHeads * this->rotaryDim));
        #endif
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void FusedRopeFP16<T>::Compute(uint64_t index, uint64_t loopN)
{
    kSize = this->numHeads * this->headSize;
    uint64_t keyInOffset = index * this->numTokensEachLoopCurrentCore * this->kLeadingDimension;
    uint64_t offsetK = index * this->numTokensEachLoopCurrentCore * kSize;
    uint32_t dstShape[2] = {static_cast<uint32_t>(this->numHeadsMax), static_cast<uint32_t>(this->rotaryDim)};
    uint32_t dstShape4Negone[2] = {
        static_cast<uint32_t>(loopN * this->numHeadsMax), static_cast<uint32_t>(this->rotaryDim)};
    uint32_t srcShape[2] = {1, static_cast<uint32_t>(this->rotaryDim)};

    LocalTensor<T> inQQueBeforeCastLocal = inQQueBeforeCast.AllocTensor<T>();
    LocalTensor<T> inQueueCosSinCacheBeforeCastLocal = inQueueCosSinCacheBeforeCast.AllocTensor<T>();
    LocalTensor<T> outQueAfterCastLocal = outQueAfterCast.AllocTensor<T>();
    LocalTensor<float> inLocal = inQQue.Get<float>();
    LocalTensor<float> inCosSin = inQueueCosSinCache.Get<float>();
    LocalTensor<float> reverseQ = reverseBuf.Get<float>();
    LocalTensor<float> negOne = negOneBuf.Get<float>();
    LocalTensor<float> oneNeg = oneNegBuf.Get<float>();
    LocalTensor<float> cosSin = cosSinBuf.Get<float>();
    LocalTensor<T> inQueCalLocal = inQueCalBuf.Get<T>();
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    LocalTensor<float> temp1Local = temp1.Get<float>();

    DataCopy(
        inQQueBeforeCastLocal, keyInGM[keyInOffset],
        {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->numHeads * headBlockLen / 2),
         static_cast<uint16_t>(this->kLeadingDimension / ELE_NUM_FP16 - this->kSize / ELE_NUM_FP16), 0});
    PipeBarrier<PIPE_ALL>();
    DataCopy(
        inQueCalLocal, inQQueBeforeCastLocal,
        {static_cast<uint16_t>(loopN * this->numHeads), static_cast<uint16_t>(rotaryBlockLen / 2),
         static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2), 0});
    PipeBarrier<PIPE_ALL>();

    if (this->isNeoxStyle == 0) {
        // GPT-J style
        Cast(
            temp1Local, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
            static_cast<uint16_t>(loopN * this->numHeads * this->rotaryDim));
        PipeBarrier<PIPE_V>();
        uint64_t rsv = 0;
        for (uint32_t i = 0; i < loopN * this->numHeads; i++) {
            GatherMask(
                inLocal[i * this->rotaryDim], temp1Local[i * this->rotaryDim], static_cast<uint8_t>(1), true,
                this->rotaryDim, {1, 1, 8, 0}, rsv);
            PipeBarrier<PIPE_ALL>();
            GatherMask(
                inLocal[i * this->rotaryDim + this->rotaryDim / 2], temp1Local[i * this->rotaryDim],
                static_cast<uint8_t>(2), true, this->rotaryDim, {1, 1, 8, 0}, rsv);
            PipeBarrier<PIPE_ALL>();
        }
    } else {
        Cast(
            inLocal, inQueCalLocal, AscendC::RoundMode::CAST_NONE,
            static_cast<uint16_t>(loopN * this->numHeads * this->rotaryDim));
        PipeBarrier<PIPE_V>();
    }

    PipeBarrier<PIPE_V>();
    ReverseRope(index, loopN, inLocal, reverseQ, cosSin, negOne, inCosSin, 
                inQueueCosSinCacheBeforeCastLocal, oldPositionIdGM, cosSinCacheGM,
                dstShape, srcShape, dstShape4Negone);
    PipeBarrier<PIPE_V>();
    Rope(index, loopN, inLocal, reverseQ, cosSin, oneNeg, inCosSin,
         inQueueCosSinCacheBeforeCastLocal, inQueCalLocal, temp1Local, offsetLocal,
         newPositionIdGM, cosSinCacheGM, dstShape, srcShape, dstShape4Negone);
    PipeBarrier<PIPE_V>();

    if (this->headSize != this->rotaryDim) {
        DataCopy(
            outQueAfterCastLocal, inQueCalLocal,
            {static_cast<uint16_t>(loopN * this->numHeads), static_cast<uint16_t>(rotaryBlockLen / 2), 0,
             static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2)});
        PipeBarrier<PIPE_ALL>();
        DataCopy(
            outQueAfterCastLocal[this->rotaryDim], inQQueBeforeCastLocal[this->rotaryDim],
            {static_cast<uint16_t>(loopN * this->numHeads),
             static_cast<uint16_t>(headBlockLen / 2 - rotaryBlockLen / 2), static_cast<uint16_t>(rotaryBlockLen / 2),
             static_cast<uint16_t>(rotaryBlockLen / 2)});
        PipeBarrier<PIPE_ALL>();
    } else {
        DataCopy(
            outQueAfterCastLocal, inQueCalLocal,
            {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->numHeads * headBlockLen / 2), 0, 0});
        PipeBarrier<PIPE_ALL>();
    }

    DataCopy(
        keyGM[offsetK], outQueAfterCastLocal,
        {static_cast<uint16_t>(loopN), static_cast<uint16_t>(this->numHeads * headBlockLen / 2), 0, 0});
    PipeBarrier<PIPE_ALL>();

    inQQueBeforeCast.FreeTensor(inQQueBeforeCastLocal);
    inQueueCosSinCacheBeforeCast.FreeTensor(inQueueCosSinCacheBeforeCastLocal);
    outQueAfterCast.FreeTensor(outQueAfterCastLocal);
}


template <typename T>
__aicore__ inline void FusedRopeFP16<T>::Process()
{
    for (uint64_t n = 0; n < this->loopTimeCurrentCore - 1; n++) {
        Compute(n, this->numTokensEachLoopCurrentCore);
    }
    if (this->numTokensLastLoopCurrentCore == 0) {
        Compute(this->loopTimeCurrentCore - 1, this->numTokensEachLoopCurrentCore);
    } else {
        Compute(this->loopTimeCurrentCore - 1, this->numTokensLastLoopCurrentCore);
    }
}
}

#endif