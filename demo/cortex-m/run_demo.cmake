# the Cortex-M demo: compile a model with nullc, build it with the demo's
# startup code for qemu's mps2-an386 board (a Cortex-M4 with a hardware
# float unit), run it under qemu on recorded inputs, check it against onnx
# runtime's outputs, and check that its outputs are bit for bit those of the
# same generated C built for the host
#
#   cmake -DNULLC=<nullc> -DMODEL=<model.onnx> -DVECTORS=<file.vectors>
#         -DARM_GCC=<arm-none-eabi-gcc> -DQEMU=<qemu-system-arm>
#         -DCC=<host c compiler> -DDRIVER=<tests/e2e/run_vectors.c>
#         -DOUT=<dir> -P run_demo.cmake

get_filename_component(demo "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
# qemu runs in OUT, where the outputs file appears, so every path is made absolute
foreach(var NULLC MODEL VECTORS DRIVER OUT)
    get_filename_component(${var} "${${var}}" ABSOLUTE)
endforeach()
file(MAKE_DIRECTORY "${OUT}")
set(c_file "${OUT}/model.c")
set(elf "${OUT}/model.elf")

execute_process(
    COMMAND "${NULLC}" "${MODEL}" --batch 1 -o "${c_file}"
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "nullc failed (${code}):\n${out}${err}")
endif()

# -ffp-contract=off as everywhere: the Cortex-M4 has fused multiply-add, and
# allowing it would change the last bits. no C library: -nostdlib, with
# libgcc for 64-bit integer division
execute_process(
    COMMAND "${ARM_GCC}" -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
            -std=c99 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic -Werror
            -ffreestanding -nostdlib -T "${demo}/link.ld" "-I${demo}"
            "-DVECTORS_FILE=\"${VECTORS}\""
            "${demo}/startup.c" "${demo}/main.c" "${demo}/vectors.S" "${c_file}" -lgcc -o "${elf}"
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "building for the Cortex-M failed:\n${out}${err}")
endif()

get_filename_component(arm_bin "${ARM_GCC}" DIRECTORY)
find_program(ARM_SIZE arm-none-eabi-size HINTS "${arm_bin}")
if(ARM_SIZE)
    execute_process(COMMAND "${ARM_SIZE}" "${elf}" OUTPUT_VARIABLE sizes)
    message(STATUS "image sizes (text holds the code and the weights; bss the arena and the I/O buffers):\n${sizes}")
endif()

file(REMOVE "${OUT}/lenet_cortex_m.out")
execute_process(
    COMMAND "${QEMU}" -machine mps2-an386 -nographic -monitor none
            -semihosting-config enable=on,target=native -kernel "${elf}"
    WORKING_DIRECTORY "${OUT}"
    TIMEOUT 300
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
message(STATUS "${out}${err}")
if(NOT code EQUAL 0)
    message(FATAL_ERROR "the model on the Cortex-M disagrees with onnx runtime, or did not finish (${code})")
endif()

# the same generated C on the host, through the end-to-end driver
set(libm "")
if(UNIX)
    set(libm "-lm")
endif()
set(host_exe "${OUT}/model_host${CMAKE_EXECUTABLE_SUFFIX}")
if(WIN32)
    set(host_exe "${OUT}/model_host.exe")
endif()
execute_process(
    COMMAND "${CC}" -std=c99 -O2 -ffp-contract=off "${c_file}" "${DRIVER}" -o "${host_exe}" ${libm}
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "building the model for the host failed:\n${out}${err}")
endif()
execute_process(
    COMMAND "${host_exe}" "${VECTORS}" "${OUT}/host.out"
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "the model on the host disagrees with onnx runtime:\n${out}${err}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}/lenet_cortex_m.out" "${OUT}/host.out"
    RESULT_VARIABLE code)
if(NOT code EQUAL 0)
    message(FATAL_ERROR "the Cortex-M and the host give different bits")
endif()
message(STATUS "the Cortex-M's outputs are bit for bit the host's")
