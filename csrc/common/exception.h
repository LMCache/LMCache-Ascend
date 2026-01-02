#pragma once
#include <sstream>
#include <string>

#if defined(USE_MINDSPORE) && (USE_MINDSPORE)
#define LMCACHE_ASCEND_CHECK(cond, ...)                                        \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::stringstream ss;                                                    \
      ss << __VA_ARGS__;                                                       \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0);
#else
#include "torch/extension.h"
#define LMCACHE_ASCEND_CHECK(...) TORCH_CHECK(__VA_ARGS__)
#endif
