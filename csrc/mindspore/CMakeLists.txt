set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11")
find_package(pybind11 REQUIRED)

file(GLOB SRC_FILES
${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${MINDSPORE_PATH}
  ${MINDSPORE_PATH}/include
  ${MINDSPORE_PATH}/include/third_party
  ${MINDSPORE_PATH}/include/third_party/robin_hood_hashing
  ${MINDSPORE_PATH}/include/third_party/securec/include
  ${MINDSPORE_PATH}/include/mindspore/core/include
  ${MINDSPORE_PATH}/include/mindspore/core/mindrt/include
  ${MINDSPORE_PATH}/include/mindspore/core/mindrt
  ${MINDSPORE_PATH}/include/mindspore/ops
  ${MINDSPORE_PATH}/include/mindspore/ops/kernel/include
  ${MINDSPORE_PATH}/include/mindspore/ccsrc
  ${MINDSPORE_PATH}/include/mindspore/ccsrc/include
  ${MINDSPORE_PATH}/include/mindspore/ccsrc/minddata/mindrecord/include
  ${MINDSPORE_PATH}/lib
  ${MINDSPORE_PATH}/lib/plugin
  ${ASCEND_CANN_PACKAGE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/ascendc/include
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include/experiment/platform
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include/experiment/ascend_hal
)

set(PYMODULE_FILES 
    ${SRC_FILES}
)

pybind11_add_module(c_ops ${PYMODULE_FILES})

target_compile_definitions(c_ops PRIVATE
ENABLE_FAST_HASH_TABLE=1
-DUSE_MINDSPORE=1
)
