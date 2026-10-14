cmake_minimum_required(VERSION 3.15)

project(Null VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

# undefined behaviour traps and bounds-checked standard containers
# trap mode needs no sanitizer runtime, so it also works with mingw
option(NULL_SANITIZE "Build with UB sanitizer (trap mode) and libstdc++ assertions" OFF)
if(NULL_SANITIZE)
    add_compile_options(-fsanitize=undefined -fsanitize-trap=all -fno-omit-frame-pointer)
    add_compile_definitions(_GLIBCXX_ASSERTIONS)
endif()

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
set(TEST_SUITES tensor node graph protobuf onnx adversarial fixtures)

foreach(suite IN LISTS TEST_SUITES)
    add_executable(test_${suite} tests/test_main.cpp tests/test_${suite}.cpp)
    target_link_libraries(test_${suite} PRIVATE null_core null_frontend)
    target_compile_definitions(test_${suite} PRIVATE
        NULL_TEST_SUITE="${suite}"
        NULL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    add_test(NAME ${suite} COMMAND test_${suite})
endforeach()

# mutation fuzzer, a short fixed-seed run is part of ctest
# longer runs: fuzz_frontend <iterations> <rng_seed>
add_executable(fuzz_frontend tests/fuzz_frontend.cpp)
target_link_libraries(fuzz_frontend PRIVATE null_core null_frontend)
target_include_directories(fuzz_frontend PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
add_test(NAME fuzz COMMAND fuzz_frontend 20000 1)
set_tests_properties(fuzz PROPERTIES TIMEOUT 120)
