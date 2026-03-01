find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

set(HIXL_SRC_FILES
    bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/hcomm_devva.cc
)

pybind11_add_module(hixl_npu_comms ${HIXL_SRC_FILES})

target_include_directories(hixl_npu_comms PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include/experiment/msprof
)

target_link_directories(hixl_npu_comms PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hixl_npu_comms PRIVATE
    cann_hixl
    ascendcl
    runtime
    hccl
    pthread
)

target_compile_definitions(hixl_npu_comms PRIVATE
    FUNC_VISIBILITY
)

install(TARGETS hixl_npu_comms DESTINATION ${LMC_INSTALL_PATH})
