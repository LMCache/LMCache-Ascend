find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

set(HCOMM_ONESIDED_SRC_FILES
    bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/hcomm_devva.cc
)

pybind11_add_module(hcomm_onesided ${HCOMM_ONESIDED_SRC_FILES})

target_include_directories(hcomm_onesided PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc/runtime
)

target_link_directories(hcomm_onesided PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hcomm_onesided PRIVATE
    ascendcl
    runtime
    hccl
    hcomm
    pthread
)

target_compile_definitions(hcomm_onesided PRIVATE
    FUNC_VISIBILITY
)

install(TARGETS hcomm_onesided DESTINATION ${LMC_INSTALL_PATH})
