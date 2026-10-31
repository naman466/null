/*
 * tests for runtime/null_runtime.h, one file per part of the runtime
 * compiled twice: as C99 (test_runtime_c) and as C++ (test_runtime_cpp),
 * so the header is proven usable from both generated C and the compiler
 */
#include "check.h"

#include "test_core.c"
#include "test_elementwise.c"
#include "test_matmul.c"
#include "test_softmax.c"
#include "test_conv.c"
#include "test_pool.c"
#include "test_reduce.c"

int main(void) {
    run_core_tests();
    run_elementwise_tests();
    run_matmul_tests();
    run_softmax_tests();
    run_conv_tests();
    run_pool_tests();
    run_reduce_tests();

#ifdef __cplusplus
    printf("=== runtime (C++) ===\n");
#else
    printf("=== runtime (C) ===\n");
#endif
    printf("Results: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
