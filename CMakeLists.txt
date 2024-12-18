cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

project(SecureRAG)
add_subdirectory(securerag)
add_subdirectory(securerag/ext)

SET(CMAKE_INSTALL_PREFIX ".")