cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

project(Ext LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND
    ${Python_EXECUTABLE} -c
    "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
find_package(Torch REQUIRED)

set(SRCS
    operator.cpp
)

add_library(${TORCH_SECURERAG_LIB} SHARED ${SRCS})
target_link_libraries(${TORCH_SECURERAG_LIB} PRIVATE 
    ${TORCH_LIBRARIES}
    Python::Python
    ${SGX_RAG_LIB}
)

include_directories(
    ${INCLUDE}/sgx
    ${INCLUDE}/core
)