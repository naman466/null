cmake_minimum_required(VERSION 3.15)

project(Null VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
add_library(null_core ${CORE_SOURCES})

target_include_directories(null_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(null_cli src/main.cpp)

target_link_libraries(null_cli PRIVATE null_core)


