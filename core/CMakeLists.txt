cmake_minimum_required(VERSION 3.5.0)
project(Core VERSION 0.1.0 LANGUAGES C CXX)

# sgx
find_package(SGX REQUIRED)

# src
set(SRCS
    utils.cpp
    operator.cpp
    model/data.cpp
    model/func.cpp
)
set(HEADER_DIRS
    .
    model
    ${INCLUDE}/eigen
)
include_directories(${HEADER_DIRS})
set(HEADERS
    constant.h
    operator.h
    utils.h
    model/func.h
    model/data.h
)
get_filename_component(PUBLIC_INCLUDE "include" ABSOLUTE)
install(FILES ${HEADERS} DESTINATION ${PUBLIC_INCLUDE})

# CORE
add_library(${CORE} SHARED ${SRCS})
target_compile_definitions(${CORE} PRIVATE EIGEN=1)
target_include_directories(${CORE} PUBLIC ./include)
target_include_directories(${CORE} PRIVATE 
    ${INCLUDE}/dnnl
)
target_link_directories(${CORE} PRIVATE ${LIB}/dnnl)
target_link_libraries(${CORE} PRIVATE dnnl)

# SGX_CORE
add_trusted_library(${SGX_CORE}
    SRCS ${SRCS}  
    EDL ${ENCLAVE_EDL}
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
)
# TODO: tmp need move public header to include
target_compile_definitions(${SGX_CORE} PRIVATE SGX=1)
target_compile_definitions(${SGX_CORE} PRIVATE EIGEN=1)
target_include_directories(${SGX_CORE} PRIVATE 
    ${INCLUDE}/sgx
    ${INCLUDE}/core
)