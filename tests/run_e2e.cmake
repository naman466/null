# end-to-end: compile a model with nullc -o, build the generated C with the
# vectors driver, and check its answers against onnx runtime's recorded ones.
# the model is compiled twice, optimised with its memory planned (the default)
# and with --no-opt --no-plan:
# both must agree with onnx runtime, and with each other bit for bit (unless
# the optimiser folded a BatchNorm, which changes the arithmetic)
#
#   cmake -DNULLC=<nullc> -DMODEL=<model.onnx> -DVECTORS=<file.vectors>
#         -DDRIVER=<run_vectors.c> -DOUT=<dir> -DCC=<c compiler>
#         [-DTOLERANCE=<atol>,<rtol>] -P run_e2e.cmake

file(MAKE_DIRECTORY "${OUT}")
get_filename_component(name "${VECTORS}" NAME_WE)

# -ffp-contract=off: the same arithmetic on every compiler (see the runtime header)
set(tolerance "")
if(TOLERANCE)
    string(REPLACE "," ";" parts "${TOLERANCE}")
    list(GET parts 0 atol)
    list(GET parts 1 rtol)
    set(tolerance "-DATOL=${atol}" "-DRTOL=${rtol}")
endif()

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
        set(flags --no-opt --no-plan)
    endif()

    execute_process(
        COMMAND "${NULLC}" "${MODEL}" --batch 1 ${flags} -o "${c_file}"
        RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "nullc ${flags} failed (${code}):\n${out}${err}")
    endif()
    # folding a BatchNorm into a Conv is the one pass that changes the
    # arithmetic (new weights, rounded once): nullc's report says when it ran
    if(variant STREQUAL "opt" AND out MATCHES "fold-batchnorm [1-9]")
        set(changes_arithmetic TRUE)
    endif()

    execute_process(
        COMMAND "${CC}" -std=c99 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic -Werror
                ${tolerance} "${c_file}" "${DRIVER}" -o "${exe}" ${libm}
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

if(changes_arithmetic)
    message(STATUS "a BatchNorm was folded into its Conv, which changes the arithmetic: "
                   "both builds were held to onnx runtime's answers instead of to each other")
    return()
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}/${name}_opt.out" "${OUT}/${name}_noopt.out"
    RESULT_VARIABLE code)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "the optimised and unoptimised builds give different outputs")
endif()
message(STATUS "optimised and unoptimised outputs are identical")
