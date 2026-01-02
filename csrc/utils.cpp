#include "utils.h"
#include "dcmi_management.h"
#include <stdexcept>
#include <string>

namespace vllm_ascend {
kvcache_ops::AscendType get_dtype_from_torch(at::ScalarType scalarType) {
  if (scalarType == at::ScalarType::Float) {
    return kvcache_ops::AscendType::FP32;
  } else if (scalarType == at::ScalarType::BFloat16) {
    return kvcache_ops::AscendType::BF16;
  } else if (scalarType == at::ScalarType::Half) {
    return kvcache_ops::AscendType::FP16;
  } else if (scalarType == at::ScalarType::Long) {
    return kvcache_ops::AscendType::INT64;
  } else if (scalarType == at::ScalarType::Int) {
    return kvcache_ops::AscendType::INT32;
  } else {
    TORCH_CHECK(false, "ScalarType not supported.");
  }
}
} // namespace vllm_ascend
