#include "pos_kernels.h"
#include "tiling/platform/platform_ascendc.h"
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
namespace py = pybind11;

namespace {

constexpr uint32_t CORE_ALLOC_TOKENS_THRESHOLD = 4000;
constexpr uint32_t CORE_NUM_SMALL_TOKENS = 8;
constexpr uint32_t CORE_NUM_LARGE_TOKENS = 16;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;

constexpr int64_t UB_SIZE_910 = static_cast<int64_t>(192) * 1024;
constexpr int64_t UB_SIZE_310 = static_cast<int64_t>(252) * 1024;

constexpr uint32_t FP32_DTYPE_SIZE = 4;

struct TilingParams {
  uint64_t coreNumUse = 0;
  uint64_t numTokens = 0;
  uint64_t numHeads = 0;
  uint64_t headSize = 0;
  uint64_t rotaryDim = 0;
  uint64_t kLeadingDimension = 0;
  uint64_t isNeoxStyle = 0;
  uint64_t frontCore = 0;
  uint64_t tailCore = 0;
  uint64_t numTokensFrontCoreEachLoop = 0;
  uint64_t numTokensTailCoreEachLoop = 0;
  uint64_t numTokensEachFrontCore = 0;
  uint64_t numTokensEachTailCore = 0;
  uint64_t loopTimeEachFrontCore = 0;
  uint64_t loopTimeEachTailCore = 0;
  uint64_t numTokensFrontCoreLastLoop = 0;
  uint64_t numTokensTailCoreLastLoop = 0;
  uint64_t tilingKey = 0;
  int64_t ubSize = UB_SIZE_910;
};
} // namespace

void GetDtypeInfo(const at::Tensor &key, uint64_t &tilingKey) {
  auto ascendType = vllm_ascend::get_dtype_from_torch(key.scalar_type());
  tilingKey = static_cast<uint64_t>(ascendType);

  if (ascendType != kvcache_ops::AscendType::BF16 &&
      ascendType != kvcache_ops::AscendType::FP16 &&
      ascendType != kvcache_ops::AscendType::FP32) {
    throw std::invalid_argument("Unavailable tensor type for RoPE Tiling (only "
                                "support BF16/FP16/FP32).");
  }
}

uint64_t CalculateTargetCoreNum(const char *socName, uint64_t availableAivCore,
                                uint64_t numTokens) {
  bool is310Platform = (strstr(socName, "310") != nullptr);
  if (is310Platform || numTokens <= CORE_ALLOC_TOKENS_THRESHOLD) {
    return std::min(availableAivCore,
                    static_cast<uint64_t>(CORE_NUM_SMALL_TOKENS));
  } else {
    return std::min(availableAivCore,
                    static_cast<uint64_t>(CORE_NUM_LARGE_TOKENS));
  }
}

uint64_t GetPlatformUbSize(const char *socName) {
  if (strstr(socName, "310") != nullptr) {
    return UB_SIZE_310;
  } else {
    return UB_SIZE_910;
  }
}

void ComputeTilingParams(TilingParams &params, const at::Tensor &key,
                         const at::Tensor &cosSinCache) {
  params.numTokens = static_cast<uint64_t>(key.size(DIM_0));
  params.kLeadingDimension = static_cast<uint64_t>(key.size(DIM_1));
  params.numHeads = params.kLeadingDimension / params.headSize;

  uint64_t cosSinDimNum = static_cast<uint64_t>(cosSinCache.dim());
  params.rotaryDim = static_cast<uint64_t>(cosSinCache.size(cosSinDimNum - 1));

  const char *socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);

  const uint64_t availableAivCore =
      static_cast<uint64_t>(ascendcPlatform->GetCoreNumAiv());
  const uint64_t targetCoreNum =
      CalculateTargetCoreNum(socName, availableAivCore, params.numTokens);

  params.coreNumUse = targetCoreNum;

  uint64_t ubSizePlatform = 0;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                  ubSizePlatform);
  params.ubSize = GetPlatformUbSize(socName);
  uint64_t maxUbSize =
      std::min(ubSizePlatform, static_cast<uint64_t>(params.ubSize));

  GetDtypeInfo(key, params.tilingKey);

  uint64_t totalDataNum = params.numTokens;
  params.frontCore = (totalDataNum % targetCoreNum != 0)
                         ? (totalDataNum % targetCoreNum)
                         : targetCoreNum;
  params.tailCore =
      (totalDataNum <= targetCoreNum) ? 0 : (targetCoreNum - params.frontCore);
  uint64_t blockDim = params.frontCore + params.tailCore;
  params.coreNumUse = blockDim;

  uint64_t numHeadsMax = params.numHeads;

  uint64_t perTokenUbSize =
      params.isNeoxStyle == 1
          ? (numHeadsMax * (params.rotaryDim * 9 + params.headSize) *
             FP32_DTYPE_SIZE)
          : (numHeadsMax * (params.rotaryDim * 11 + params.headSize) *
             FP32_DTYPE_SIZE);
  uint64_t maxNPerLoopForUb =
      (perTokenUbSize == 0) ? 0 : (maxUbSize / perTokenUbSize);
  maxNPerLoopForUb = std::max(maxNPerLoopForUb, static_cast<uint64_t>(1));

  params.numTokensEachFrontCore =
      (totalDataNum + targetCoreNum - 1) / targetCoreNum;
  params.loopTimeEachFrontCore =
      (params.numTokensEachFrontCore + maxNPerLoopForUb - 1) / maxNPerLoopForUb;
  params.numTokensFrontCoreEachLoop = (params.loopTimeEachFrontCore == 1)
                                          ? params.numTokensEachFrontCore
                                          : maxNPerLoopForUb;
  params.numTokensFrontCoreLastLoop =
      (params.loopTimeEachFrontCore == 1)
          ? 0
          : (params.numTokensEachFrontCore -
             params.numTokensFrontCoreEachLoop *
                 (params.loopTimeEachFrontCore - 1));

  params.numTokensEachTailCore =
      (totalDataNum <= targetCoreNum) ? 0 : (totalDataNum / targetCoreNum);
  params.loopTimeEachTailCore =
      (params.numTokensEachTailCore + maxNPerLoopForUb - 1) / maxNPerLoopForUb;
  params.numTokensTailCoreEachLoop = (params.loopTimeEachTailCore <= 1)
                                         ? params.numTokensEachTailCore
                                         : maxNPerLoopForUb;
  params.numTokensTailCoreLastLoop =
      (params.loopTimeEachTailCore == 1)
          ? 0
          : (params.numTokensEachTailCore -
             params.numTokensTailCoreEachLoop *
                 (params.loopTimeEachTailCore - 1));
}

void rotary_embedding_k_fused(torch::Tensor &oldPositions,
                              torch::Tensor &newPositions, torch::Tensor &key,
                              int64_t headSize, torch::Tensor &cosSinCache,
                              bool isNeox) {

  TilingParams params;
  params.headSize = static_cast<uint64_t>(headSize);
  params.isNeoxStyle = static_cast<uint64_t>(isNeox);
  params.kLeadingDimension = static_cast<uint64_t>(key.size(DIM_1));

  ComputeTilingParams(params, key, cosSinCache);

  uint8_t *oldPositionsPtr =
      get_kernel_ptr<uint8_t, torch::Tensor>(oldPositions);
  uint8_t *newPositionsPtr =
      get_kernel_ptr<uint8_t, torch::Tensor>(newPositions);
  uint8_t *keyPtr = get_kernel_ptr<uint8_t, torch::Tensor>(key);
  uint8_t *cosSinCachePtr = get_kernel_ptr<uint8_t, torch::Tensor>(cosSinCache);
  uint8_t *keyOutPtr = keyPtr;

  auto aclStream = c10_npu::getCurrentNPUStream().stream();

  at_npu::native::OpCommand cmd;
  cmd.Name("fused_rope");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::rotary_embedding_kernel_dispatch(
        params.coreNumUse, aclStream, oldPositionsPtr, newPositionsPtr, keyPtr,
        cosSinCachePtr, keyOutPtr, params.numTokens, params.numHeads,
        params.headSize, params.rotaryDim, params.kLeadingDimension,
        params.isNeoxStyle, params.frontCore, params.tailCore,
        params.numTokensFrontCoreEachLoop, params.numTokensTailCoreEachLoop,
        params.numTokensEachFrontCore, params.numTokensEachTailCore,
        params.loopTimeEachFrontCore, params.loopTimeEachTailCore,
        params.numTokensFrontCoreLastLoop, params.numTokensTailCoreLastLoop,
        params.tilingKey);
    return 0;
  });
  cmd.Run();
}
