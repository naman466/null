/*
 * the runtime tests' checks, shared by the test_<part>.c files. they are all
 * included into test_kernels.c, so this is one translation unit
 */
#ifndef NULL_RUNTIME_TEST_CHECK_H
#define NULL_RUNTIME_TEST_CHECK_H

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "null_runtime.h"

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            printf("  [FAIL] %s:%d: %s\n", __func__, __LINE__, #cond);           \
        }                                                                        \
    } while (0)

/* bitwise float equality, so -0.0 and NaN payloads are compared exactly */
static int same_bits(float a, float b) {
    return memcmp(&a, &b, sizeof a) == 0;
}

static int all_equal(const float* got, const float* want, int64_t n) {
    int64_t i;
    for (i = 0; i < n; ++i)
        if (!same_bits(got[i], want[i])) return 0;
    return 1;
}

#endif /* NULL_RUNTIME_TEST_CHECK_H */
