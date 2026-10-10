cmake_minimum_required(VERSION 3.15)

project(Null VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

# core IR is header-only
add_library(null_core INTERFACE)

target_include_directories(null_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# onnx frontend
add_library(null_frontend STATIC
    src/frontend/protobuf.cpp
    src/frontend/onnx.cpp
)
target_link_libraries(null_frontend PUBLIC null_core)

add_executable(null_cli src/main.cpp)

target_link_libraries(null_cli PRIVATE null_core)

# tests
enable_testing()

# one executable per suite, built from tests/test_<suite>.cpp
set(TEST_SUITES tensor node graph protobuf onnx)

foreach(suite IN LISTS TEST_SUITES)
    add_executable(test_${suite} tests/test_main.cpp tests/test_${suite}.cpp)
    target_link_libraries(test_${suite} PRIVATE null_core null_frontend)
    target_compile_definitions(test_${suite} PRIVATE NULL_TEST_SUITE="${suite}")
    add_test(NAME ${suite} COMMAND test_${suite})
endforeach()


