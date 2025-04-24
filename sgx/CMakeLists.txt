cmake_minimum_required(VERSION 3.5.0)
project(SGX VERSION 0.1.0 LANGUAGES C CXX)

find_package(SGX REQUIRED)

# APP
set(APP_SRCS
    app/app.cpp
    app/libcxx.cpp
)

set(SGXRAGLIB sgxrag)
add_untrusted_library(${SGXRAGLIB} SHARED
    SRCS ${SRCS}
    EDL ${ENCLAVE_EDL}
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
)
target_link_libraries(${SGXRAGLIB} ${CORE})
target_link_directories(${SGXRAGLIB} PRIVATE
    app
)
add_dependencies(${SGXRAGLIB} enclave-sign)

# ENCLAVE
set(ENCLAVE_SRCS
    enclave/enclave.cpp
    enclave/libcxx.cpp
    enclave/mask.cpp
)
add_enclave_library(enclave
    SRCS ${ENCLAVE_SRCS}
    TRUSTED_LIBS  ${SGX_CORE}
    EDL ${ENCLAVE_EDL}
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
    LDSCRIPT ${LDS}
)
add_dependencies(enclave ${SGX_CORE})
target_include_directories(enclave PRIVATE
    enclave
    ${CMAKE_SOURCE_DIR}/include/core
)
enclave_sign(enclave
    KEY enclave/Enclave_private.pem
    CONFIG enclave/Enclave.config.xml
)