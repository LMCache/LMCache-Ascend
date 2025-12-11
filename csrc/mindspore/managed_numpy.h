#pragma once
#include <shared_mutex>
#include <map>
#include "ms_extension/api.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "ms_extension.h"
#include "aclnn/opdev/platform.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "ms_extension/api.h"

namespace py = pybind11;

py::array create_mmapped_numpy(size_t buffer_size);
