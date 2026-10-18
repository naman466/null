# runs the cli once and checks both its exit code and its output
# (ctest's PASS_REGULAR_EXPRESSION alone ignores the exit code)
#
#   cmake -DEXE=<null_cli> "-DARGS=a;b" -DEXPECT_EXIT=<code> "-DEXPECT_OUTPUT=<regex>" -P run_cli.cmake

execute_process(
    COMMAND ${EXE} ${ARGS}
    RESULT_VARIABLE code
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
)
set(all "${out}${err}")

if(NOT code EQUAL EXPECT_EXIT)
    message(FATAL_ERROR "exit code ${code}, expected ${EXPECT_EXIT}\n--- output ---\n${all}")
endif()
if(NOT all MATCHES "${EXPECT_OUTPUT}")
    message(FATAL_ERROR "output does not match '${EXPECT_OUTPUT}'\n--- output ---\n${all}")
endif()
