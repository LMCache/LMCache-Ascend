find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

set(HCCL_SRC_FILES 
    hccl_utils.cpp
    hccl_agent.cpp
    bindings.cpp
)

pybind11_add_module(hccl_npu_comms ${HCCL_SRC_FILES})

target_include_directories(hccl_npu_comms PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util/error_manager
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

target_link_directories(hccl_npu_comms PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/lib64
)

target_link_libraries(hccl_npu_comms PRIVATE
    hccl_plf
    ascendcl
    hccl
    runtime
    pthread
)

target_compile_definitions(hccl_npu_comms PRIVATE
    OPEN_BUILD_PROJECT
)

install(TARGETS hccl_npu_comms DESTINATION ${LMC_INSTALL_PATH})