find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

set(HCCL_SRC_FILES 
    hccl_utils.cpp
    hccl_agent.cpp
    bindings.cpp
)

pybind11_add_module(hccl_npu_comms ${HCCL_SRC_FILES})

target_include_directories(hccl_npu_comms PRIVATE
    ${HCOMM_SRC_PATH}/src/pub_inc
    ${HCOMM_SRC_PATH}/src/pub_inc/inner
    ${HCOMM_SRC_PATH}/include
    ${HCOMM_SRC_PATH}/include/hccl
    ${HCOMM_SRC_PATH}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_directories(hccl_npu_comms PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/lib64
)

target_link_libraries(hccl_npu_comms PRIVATE
    hccl_plf
    ascendcl
    runtime
    pthread
)

target_compile_definitions(hccl_npu_comms PRIVATE
    OPEN_BUILD_PROJECT
)

install(TARGETS hccl_npu_comms DESTINATION ${LMC_INSTALL_PATH})