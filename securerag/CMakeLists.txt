cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

add_subdirectory(test)

# dnnl need -std=c++17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MARK: SGX
set(SGX_HW OFF)
set(SGX_MODE Debug)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(SGX REQUIRED)

if(SGX_HW AND SGX_MODE STREQUAL "Release")
    set(LDS Enclave/Enclave.lds)
else()
    set(LDS Enclave/Enclave_debug.lds)
endif()

set(SGX_INCLUDE /opt/intel/sgxsdk/include)
# set(SGX_LIBRARY_PATH /opt/intel/sgxsdk/lib64)


# MARK: Common
set(COMMON common)
set(COMMON_SRC
    common/utils.cpp
    common/model/func.cpp
    common/model/data.cpp
)
add_library(${COMMON} SHARED ${COMMON_SRC})
include_directories(
    common
    common/model
    common/eigen
    ${SGX_INCLUDE}
)
target_compile_definitions(${COMMON} PRIVATE EIGEN=1)
install(TARGETS ${COMMON} DESTINATION ../dev)

# MARK: Enclave
set(TRUSTED_LIB trusted_lib)
set(EDL_SEARCH_PATHS
    Enclave
)
set(E_SRCS
    Enclave/Enclave.cpp
)
set(T_SRCS
    Enclave/TrustedLibrary/Libcxx.cpp
    Enclave/TrustedLibrary/mask.cpp
)
include_directories(
    Enclave/TrustedLibrary
)
add_trusted_library(${TRUSTED_LIB}
    SRCS ${T_SRCS} ${COMMON_SRC} # compile sgx api.
    EDL Enclave/Enclave.edl
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
)
target_include_directories(${TRUSTED_LIB} PRIVATE sgx_include)
target_compile_definitions(${TRUSTED_LIB} PRIVATE SGX=1)
target_compile_definitions(${TRUSTED_LIB} PRIVATE EIGEN=1)
add_enclave_library(enclave
    SRCS ${E_SRCS}
    TRUSTED_LIBS ${TRUSTED_LIB} 
    EDL Enclave/Enclave.edl
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
    LDSCRIPT ${LDS}
)
enclave_sign(enclave
    KEY Enclave/Enclave_private.pem
    CONFIG Enclave/Enclave.config.xml
)
install(FILES $<TARGET_FILE_DIR:enclave>/enclave.signed.so DESTINATION ../dev)

# MARK: APP
set(SRCS
    App/App.cpp
    App/TrustedLibrary/Libcxx.cpp
    App/TrustedLibrary/Model.cpp
)
include_directories(
    APP
    App/TrustedLibrary
)
add_untrusted_executable(SGXRAG
    SRCS ${SRCS}
    EDL Enclave/Enclave.edl
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
)
target_link_libraries(SGXRAG)
add_dependencies(SGXRAG enclave-sign)
set_target_properties(SGXRAG PROPERTIES EXCLUDE_FROM_ALL true)

set(SGXRAGLIB sgxrag)
add_untrusted_library(${SGXRAGLIB} SHARED
    SRCS ${SRCS}
    EDL Enclave/Enclave.edl
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
)
target_link_libraries(${SGXRAGLIB} ${COMMON}) # link non-sgx api.
add_dependencies(${SGXRAGLIB} enclave-sign)
install(TARGETS ${SGXRAGLIB} DESTINATION ../dev)
