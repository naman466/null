# end-to-end: compile a model with nullc -o, build the generated C with the
# vectors driver, and check its answers against onnx runtime's recorded ones
#
#   cmake -DNULLC=<null_cli> -DMODEL=<model.onnx> -DVECTORS=<file.vectors>
#         -DDRIVER=<run_vectors.c> -DOUT=<dir> -DCC=<c compiler> -P run_e2e.cmake

file(MAKE_DIRECTORY "${OUT}")
get_filename_component(name "${VECTORS}" NAME_WE)
set(c_file "${OUT}/${name}.c")
set(exe "${OUT}/${name}_e2e${CMAKE_EXECUTABLE_SUFFIX}")
if(WIN32)
    set(exe "${OUT}/${name}_e2e.exe")
endif()

execute_process(
    COMMAND "${NULLC}" "${MODEL}" --batch 1 -o "${c_file}"
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "nullc failed (${code}):\n${out}${err}")
endif()

# -ffp-contract=off: the same arithmetic on every compiler (see the runtime header)
set(libm "")
if(UNIX)
    set(libm "-lm")
endif()
execute_process(
    COMMAND "${CC}" -std=c99 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic -Werror
            "${c_file}" "${DRIVER}" -o "${exe}" ${libm}
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "building the generated model failed:\n${out}${err}")
endif()

execute_process(
    COMMAND "${exe}" "${VECTORS}"
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
message(STATUS "${out}${err}")
if(NOT code EQUAL 0)
    message(FATAL_ERROR "generated model disagrees with onnx runtime:\n${out}${err}")
endif()
