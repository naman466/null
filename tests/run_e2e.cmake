# end-to-end: compile a model with nullc -o, build the generated C with the
# vectors driver, and check its answers against onnx runtime's recorded ones.
# the model is compiled twice, optimised with its memory planned and the
# blocked kernels (the default), and with --no-opt --no-plan
# --reference-kernels: both must agree with onnx runtime, and with each other
# bit for bit. when the optimiser folded a BatchNorm, which changes the
# arithmetic, a third build, optimised but with --reference-kernels, must
# give the first one's bits instead
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

# build and run one variant of the model: nullc with `flags`, the C with the
# driver, the result checked against onnx runtime and written to <name>_<variant>.out
function(run_variant variant)
    set(flags ${ARGN})
    set(c_file "${OUT}/${name}_${variant}.c")
    set(exe "${OUT}/${name}_${variant}_e2e${CMAKE_EXECUTABLE_SUFFIX}")
    if(WIN32)
        set(exe "${OUT}/${name}_${variant}_e2e.exe")
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
        set(changes_arithmetic TRUE PARENT_SCOPE)
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
endfunction()

function(same_outputs a b what)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}/${name}_${a}.out" "${OUT}/${name}_${b}.out"
        RESULT_VARIABLE code)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "${what} give different outputs")
    endif()
    message(STATUS "${what} give identical outputs")
endfunction()

run_variant(opt)
run_variant(noopt --no-opt --no-plan --reference-kernels)

if(NOT changes_arithmetic)
    same_outputs(opt noopt "the optimised build and the unoptimised one with reference kernels")
    return()
endif()

# the folded BatchNorm leaves the kernels to check: the same optimised graph
# with the reference loops must give the same bits as with the blocked ones
message(STATUS "a BatchNorm was folded into its Conv, which changes the arithmetic: "
               "both builds were held to onnx runtime's answers instead of to each other")
run_variant(ref --reference-kernels)
same_outputs(opt ref "the blocked and the reference kernels")
