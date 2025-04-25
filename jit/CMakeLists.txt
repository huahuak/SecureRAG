cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(model_infer)

find_package(SGX REQUIRED)
find_package(Torch REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(SRCS
    main.cpp
)

add_executable(model_infer main.cpp)
target_link_libraries(model_infer
    ${TORCH_LIBRARIES}
    Python::Python
)

add_trusted_library(sgx_model_infer 
    SRCS ${SRCS}  
    EDL ${ENCLAVE_EDL}
    EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
)
target_link_libraries(sgx_model_infer
    ${TORCH_LIBRARIES}
    Python::Python
)
