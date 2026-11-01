# end-to-end: compile a model with nullc -o, build the generated C with the
# vectors driver, and check its answers against onnx runtime's recorded ones.
# the model is compiled twice, optimised (the default) and with --no-opt:
# both must agree with onnx runtime, and with each other bit for bit
#
#   cmake -DNULLC=<null_cli> -DMODEL=<model.onnx> -DVECTORS=<file.vectors>
#         -DDRIVER=<run_vectors.c> -DOUT=<dir> -DCC=<c compiler> -P run_e2e.cmake

file(MAKE_DIRECTORY "${OUT}")
get_filename_component(name "${VECTORS}" NAME_WE)

# -ffp-contract=off: the same arithmetic on every compiler (see the runtime header)
set(libm "")
if(UNIX)
    set(libm "-lm")
endif()

foreach(variant opt noopt)
    set(c_file "${OUT}/${name}_${variant}.c")
    set(exe "${OUT}/${name}_${variant}_e2e${CMAKE_EXECUTABLE_SUFFIX}")
    if(WIN32)
        set(exe "${OUT}/${name}_${variant}_e2e.exe")
    endif()
    set(flags "")
    if(variant STREQUAL "noopt")
        set(flags --no-opt)
    endif()

    execute_process(
        COMMAND "${NULLC}" "${MODEL}" --batch 1 ${flags} -o "${c_file}"
        RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "nullc ${flags} failed (${code}):\n${out}${err}")
    endif()

    execute_process(
        COMMAND "${CC}" -std=c99 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic -Werror
                "${c_file}" "${DRIVER}" -o "${exe}" ${libm}
        RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "building the generated model (${variant}) failed:\n${out}${err}")
    endif()

    execute_process(
        COMMAND "${exe}" "${VECTORS}" "${OUT}/${name}_${variant}.out"
        RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
    message(STATUS "${variant}: ${out}${err}")
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "generated model (${variant}) disagrees with onnx runtime:\n${out}${err}")
    endif()
endforeach()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}/${name}_opt.out" "${OUT}/${name}_noopt.out"
    RESULT_VARIABLE code)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "the optimised and unoptimised builds give different outputs")
endif()
message(STATUS "optimised and unoptimised outputs are identical")
