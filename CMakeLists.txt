# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)
# project information
project(c_ops)

set(CMAKE_CXX_STANDARD 17)
set(LMC_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(SOC_VERSION ${SOC_VERSION})
set(ARCH ${ARCH})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARCH_SUBDIR "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ARCH_SUBDIR "x86_64-linux")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_subdirectory(third_party/kvcache-ops)
add_subdirectory(csrc)


set(TORCH_LIBS_DIR "${TORCH_PATH}/lib")

target_link_options(c_ops PRIVATE
  "-Wl,-rpath,$ORIGIN:$ORIGIN/lib"
  "-Wl,-rpath,${LMC_INSTALL_PATH}" 
)

target_link_directories(
  c_ops
  PRIVATE
  ${TORCH_LIBS_DIR}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_CANN_PACKAGE_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib
)

target_link_libraries(
  c_ops
  PUBLIC
  ${TORCH_LIBRARIES}
  libtorch_npu.so
  cache_kernels
  ascendcl
  platform
  ascend_hal
  tiling_api
)


install(TARGETS c_ops cache_kernels DESTINATION ${LMC_INSTALL_PATH})