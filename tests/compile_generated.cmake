# compiles a model with nullc -o and then compiles the generated C strictly:
# C99, all warnings as errors. proves the generated file is valid,
# self-contained C on its own
#
#   cmake -DNULLC=<null_cli> -DMODEL=<model.onnx> -DOUT=<dir> -DCC=<c compiler> -P compile_generated.cmake

file(MAKE_DIRECTORY "${OUT}")
get_filename_component(name "${MODEL}" NAME_WE)
set(c_file "${OUT}/${name}.c")

execute_process(
    COMMAND "${NULLC}" "${MODEL}" --batch 1 -o "${c_file}"
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "nullc failed (${code}):\n${out}${err}")
endif()

execute_process(
    COMMAND "${CC}" -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -c "${c_file}" -o "${OUT}/${name}.o"
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "generated C does not compile cleanly:\n${out}${err}")
endif()
