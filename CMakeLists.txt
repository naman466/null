cmake_minimum_required(VERSION 3.15)

project(Null VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# the runtime and generated code are plain C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

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

# c runtime used by generated code, header-only
add_library(null_runtime INTERFACE)
target_include_directories(null_runtime INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/runtime")
# softmax uses expf; on windows it is in the c runtime, elsewhere in libm
if(UNIX)
    target_link_libraries(null_runtime INTERFACE m)
endif()

# onnx frontend
add_library(null_frontend STATIC
    src/frontend/protobuf.cpp
    src/frontend/onnx.cpp
)
target_link_libraries(null_frontend PUBLIC null_core)

# analyses over the IR
add_library(null_analysis STATIC
    src/analysis/shape_inference.cpp
    src/analysis/specialize.cpp
)
target_link_libraries(null_analysis PUBLIC null_core)

# c code generation. the runtime header is embedded in the compiler as a
# string, so generated model.c files can carry it and need nothing else;
# editing the header re-runs configure, which regenerates the string
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/null_runtime.h")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/runtime/null_runtime.h" NULL_RUNTIME_TEXT)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/runtime_source.cpp"
    "// generated from runtime/null_runtime.h by CMakeLists.txt, do not edit
"
    "#include \"backend/emit_c.h\"
"
    "namespace null {
"
    "const char* runtime_source() {
"
    "    return R\"NULLRT(" "${NULL_RUNTIME_TEXT}" ")NULLRT\";
"
    "}
"
    "} // namespace null
")

add_library(null_backend STATIC
    src/backend/emit_c.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/generated/runtime_source.cpp"
)
target_link_libraries(null_backend PUBLIC null_core null_analysis null_runtime)

add_executable(null_cli src/main.cpp)

target_link_libraries(null_cli PRIVATE null_core null_frontend null_analysis null_backend)

# tests
enable_testing()

# one executable per suite, built from tests/test_<suite>.cpp
set(TEST_SUITES tensor node graph protobuf onnx adversarial fixtures dump shape pytorch emit)

foreach(suite IN LISTS TEST_SUITES)
    add_executable(test_${suite} tests/test_main.cpp tests/test_${suite}.cpp)
    target_link_libraries(test_${suite} PRIVATE null_core null_frontend null_analysis null_backend)
    target_compile_definitions(test_${suite} PRIVATE
        NULL_TEST_SUITE="${suite}"
        NULL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
        NULL_RUNTIME_HEADER="${CMAKE_CURRENT_SOURCE_DIR}/runtime/null_runtime.h")
    add_test(NAME ${suite} COMMAND test_${suite})
endforeach()

# runtime kernels, tested as c99 and as c++
add_executable(test_runtime_c tests/runtime/test_kernels.c)
target_link_libraries(test_runtime_c PRIVATE null_runtime)
add_test(NAME runtime_c COMMAND test_runtime_c)

add_executable(test_runtime_cpp tests/runtime/test_kernels_cpp.cpp)
target_link_libraries(test_runtime_cpp PRIVATE null_runtime)
target_include_directories(test_runtime_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime")
add_test(NAME runtime_cpp COMMAND test_runtime_cpp)

# the runtime tests compare floats bit for bit, so no fused multiply-add:
# a fused a*b+c rounds once instead of twice and changes the last bit
target_compile_options(test_runtime_c PRIVATE -ffp-contract=off)
target_compile_options(test_runtime_cpp PRIVATE -ffp-contract=off)

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
add_cli_test(summary        0 "graph 'linear': 2 nodes, 2 initializers, 1 input, 1 output, all shapes known, ok" "${FIXTURES}/linear.onnx")
add_cli_test(dynamic_input  2 "input 'x' has a dynamic shape float32.[?], 1, 4, 4.*--batch N" "${FIXTURES}/mlp.onnx")
add_cli_test(batch          0 "graph 'mlp': .*all shapes known, ok" "${FIXTURES}/mlp.onnx" --batch 4)
add_cli_test(batch_equals   0 "all shapes known, ok" "${FIXTURES}/mlp.onnx" --batch=1)
add_cli_test(input_shape    0 "all shapes known, ok" "${FIXTURES}/mlp.onnx" --input-shape x=2,1,4,4)
add_cli_test(input_shape_bad_dim 2 "cannot set input 'x' .* dim 2 is fixed at 4" "${FIXTURES}/mlp.onnx" --input-shape x=2,1,5,4)
add_cli_test(input_shape_not_input 2 "'y' is not a graph input" "${FIXTURES}/mlp.onnx" --input-shape y=1)
add_cli_test(bad_batch      1 "--batch needs a positive integer, got '0'" "${FIXTURES}/mlp.onnx" --batch 0)
add_cli_test(bad_input_shape 1 "--input-shape needs NAME=D0,D1" "${FIXTURES}/mlp.onnx" --input-shape oops)
add_cli_test(batch_no_effect 0 "note: --batch had no effect" "${FIXTURES}/linear.onnx" --batch 3)
add_cli_test(dump_batch     0 "4  /softmax: Softmax.logits. -> probs: float32.4, 3." "${FIXTURES}/mlp.onnx" --dump --batch 4)
add_cli_test(dump_dynamic_note 0 "6 tensors still have dynamic dims" "${FIXTURES}/mlp.onnx" --dump)
add_cli_test(no_shape_rule  2 "shape inference: node 'Conv_0' .Conv.: no shape inference rule" "${FIXTURES}/cnn.onnx")
add_cli_test(dump_no_shape_rule 0 "warning: shape inference stopped" "${FIXTURES}/cnn.onnx" --dump)
add_cli_test(unsupported    2 "unsupported operators: LSTM .2 nodes., Shape .1 node." "${FIXTURES}/unsupported_ops.onnx")
add_cli_test(old_opset      2 "opset 11 is not supported" "${FIXTURES}/opset11.onnx")
add_cli_test(missing_file   2 "cannot open 'does-not-exist.onnx'" does-not-exist.onnx)
add_cli_test(no_args        1 "Usage: nullc")
add_cli_test(unknown_option 1 "unknown option '--bogus'" --bogus)
add_cli_test(two_models     1 "only one model can be given" a.onnx b.onnx)
add_cli_test(help           0 "Usage: nullc" --help)

# generated C: nullc -o on every model it can compile, then the C compiler
# must accept the file as C99 with every warning an error. the output folder
# is created at configure time, so any of these tests can run alone
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_models")
foreach(model relu linear mlp initializers_as_inputs
              pytorch/mlp__legacy_dyn pytorch/mlp__dynamo_dyn pytorch/mlp_view__legacy)
    string(REPLACE "/" "_" test_name "${model}")
    add_test(NAME codegen_${test_name}
        COMMAND ${CMAKE_COMMAND}
            "-DNULLC=$<TARGET_FILE:null_cli>"
            "-DMODEL=${FIXTURES}/${model}.onnx"
            "-DOUT=${CMAKE_CURRENT_BINARY_DIR}/generated_models"
            "-DCC=${CMAKE_C_COMPILER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/compile_generated.cmake")
endforeach()

add_cli_test(codegen_unsupported_op 2 "code generation: node 'Clip_4' .Clip.: no code generation for this operator yet"
             "${FIXTURES}/optionals.onnx" -o "${CMAKE_CURRENT_BINARY_DIR}/generated_models/optionals.c")
add_cli_test(output_needs_name 1 "-o needs a file name" "${FIXTURES}/mlp.onnx" -o)
add_cli_test(output_written 0 "wrote .*mlp_cli.c" "${FIXTURES}/mlp.onnx" --batch 1 -o "${CMAKE_CURRENT_BINARY_DIR}/generated_models/mlp_cli.c")

# end to end: compile each model with nullc -o, build the generated C with a
# driver, run it on recorded inputs and compare with onnx runtime's outputs
# (tests/fixtures/e2e/*.vectors, written by make_e2e_vectors.py)
foreach(model relu linear mlp initializers_as_inputs
              pytorch/mlp__legacy_dyn pytorch/mlp__dynamo_dyn pytorch/mlp_view__legacy)
    string(REPLACE "/" "__" vectors "${model}")
    string(REPLACE "/" "_" test_name "${model}")
    add_test(NAME e2e_${test_name}
        COMMAND ${CMAKE_COMMAND}
            "-DNULLC=$<TARGET_FILE:null_cli>"
            "-DMODEL=${FIXTURES}/${model}.onnx"
            "-DVECTORS=${FIXTURES}/e2e/${vectors}.vectors"
            "-DDRIVER=${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/run_vectors.c"
            "-DOUT=${CMAKE_CURRENT_BINARY_DIR}/e2e"
            "-DCC=${CMAKE_C_COMPILER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_e2e.cmake")
endforeach()
