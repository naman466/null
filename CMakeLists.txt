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

# analyses over the IR
add_library(null_analysis STATIC
    src/analysis/shape_inference.cpp
)
target_link_libraries(null_analysis PUBLIC null_core)

add_executable(null_cli src/main.cpp)

target_link_libraries(null_cli PRIVATE null_core null_frontend)

# tests
enable_testing()

# one executable per suite, built from tests/test_<suite>.cpp
set(TEST_SUITES tensor node graph protobuf onnx adversarial fixtures dump shape)

foreach(suite IN LISTS TEST_SUITES)
    add_executable(test_${suite} tests/test_main.cpp tests/test_${suite}.cpp)
    target_link_libraries(test_${suite} PRIVATE null_core null_frontend null_analysis)
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

# cli end to end: run null_cli on the fixtures, check exit code and output
set(FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
function(add_cli_test name expect_exit expect_output)
    add_test(NAME cli_${name}
        COMMAND ${CMAKE_COMMAND}
            "-DEXE=$<TARGET_FILE:null_cli>"
            "-DARGS=${ARGN}"
            "-DEXPECT_EXIT=${expect_exit}"
            "-DEXPECT_OUTPUT=${expect_output}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_cli.cmake")
endfunction()

add_cli_test(dump           0 "4  /softmax: Softmax.logits. -> probs: float32.[?], 3." "${FIXTURES}/mlp.onnx" --dump)
add_cli_test(summary        0 "graph 'linear': 2 nodes, 2 initializers, 1 input, 1 output, ok" "${FIXTURES}/linear.onnx")
add_cli_test(unsupported    2 "unsupported operators: LSTM .2 nodes., Shape .1 node." "${FIXTURES}/unsupported_ops.onnx")
add_cli_test(old_opset      2 "opset 11 is not supported" "${FIXTURES}/opset11.onnx")
add_cli_test(missing_file   2 "cannot open 'does-not-exist.onnx'" does-not-exist.onnx)
add_cli_test(no_args        1 "Usage: nullc")
add_cli_test(unknown_option 1 "unknown option '--bogus'" --bogus)
add_cli_test(two_models     1 "only one model can be given" a.onnx b.onnx)
add_cli_test(help           0 "Usage: nullc" --help)
