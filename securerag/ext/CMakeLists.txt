cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

project(SecureRAGExtension LANGUAGES CXX)

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

add_library(${PROJECT_NAME} SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} Python::Python)

# link sgx library
include_directories(
    ../App
    ../common/model
)
target_link_libraries(${PROJECT_NAME} PRIVATE sgxrag)

install(TARGETS ${PROJECT_NAME}
    DESTINATION ../dev
)
