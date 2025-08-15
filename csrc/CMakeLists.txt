include(utils.cmake)
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11")
find_package(pybind11 REQUIRED)

message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

file(GLOB SRC_FILES
${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

find_package(Torch REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/experiment/ascend_hal
  ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/experiment/platform
  ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/experiment/ascend_hal
)


set(
  INCLUDES
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/experiment/ascend_hal
)

set(PYMODULE_FILES 
    ${SRC_FILES}
)

pybind11_add_module(c_ops ${PYMODULE_FILES})
