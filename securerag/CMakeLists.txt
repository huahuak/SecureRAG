cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

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

# MARK: Common
set(COMMON common)
set(COMMON_SRC
    common/utils.cpp
    common/model/func.cpp
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
set(EDL_SEARCH_PATHS
    Enclave
)
set(E_SRCS
    Enclave/Enclave.cpp
)
set(T_SRCS
    Enclave/TrustedLibrary/Libcxx.cpp
)
add_trusted_library(trusted_lib
    SRCS ${T_SRCS} ${COMMON_SRC}
    EDL Enclave/Enclave.edl
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
)
target_compile_definitions(trusted_lib PRIVATE SGX=1)
add_enclave_library(enclave
    SRCS ${E_SRCS}
    TRUSTED_LIBS trusted_lib
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
target_link_libraries(SGXRAG ${COMMON})
add_dependencies(SGXRAG enclave-sign)
set_target_properties(SGXRAG PROPERTIES EXCLUDE_FROM_ALL true)

set(SGXRAGLIB sgxrag)
add_untrusted_library(${SGXRAGLIB} SHARED
    SRCS ${SRCS}
    EDL Enclave/Enclave.edl
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
)
target_link_libraries(${SGXRAGLIB} ${COMMON})
add_dependencies(${SGXRAGLIB} enclave-sign)
install(TARGETS ${SGXRAGLIB} DESTINATION ../dev)
