cmake_minimum_required(VERSION 3.5.0)
project(SecureRAG VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CUDA_ARCHITECTURES 86)
# FindSGX.cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# SGX
set(SGX_HW ON)
set(SGX_MODE Debug)
if(SGX_HW AND SGX_MODE STREQUAL "Release")
    get_filename_component(LDS "sgx/enclave/Enclave.lds" ABSOLUTE)
else()
    get_filename_component(LDS "sgx/enclave/Enclave_debug.lds" ABSOLUTE)
endif()
set(SGX_INCLUDE /opt/intel/sgxsdk/include)
get_filename_component(ENCLAVE_EDL "sgx/enclave/Enclave.edl" ABSOLUTE)
get_filename_component(EDL_SEARCH_PATHS "sgx/enclave" ABSOLUTE)
get_filename_component(INCLUDE "include" ABSOLUTE)
get_filename_component(LIB "lib" ABSOLUTE)

# module
set(CORE core)
set(SGX_CORE sgx_core)
add_subdirectory(core)

set(SGX_RAG_LIB sgxrag)
add_subdirectory(sgx)

set(TORCH_SECURERAG_LIB torch_securerag)
add_subdirectory(ext)

# install
install(TARGETS
    ${CORE} ${SGX_CORE} ${SGX_RAG_LIB} ${TORCH_SECURERAG_LIB}
    DESTINATION
    ${LIB}
)
