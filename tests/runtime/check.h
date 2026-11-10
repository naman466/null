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

/* the same values: every bit equal, except that any NaN equals any NaN.
   ieee 754 leaves open which NaN an operation on NaNs returns, and x86 gives
   the first operand's, so a compiler that swaps the operands of a + b (which
   it may) changes a NaN's sign or payload and nothing else */
static int all_same_values(const float* got, const float* want, int64_t n) {
    int64_t i;
    for (i = 0; i < n; ++i)
        if (!same_bits(got[i], want[i]) && !(isnan(got[i]) && isnan(want[i]))) return 0;
    return 1;
}

/* a fixed-seed generator, so random tests are the same on every run */
static uint32_t g_rng = 12345u;
static uint32_t rand_u32(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng >> 8;
}
static int64_t rand_int(int64_t lo, int64_t hi) {    /* in [lo, hi] */
    return lo + (int64_t)(rand_u32() % (uint32_t)(hi - lo + 1));
}

/* mostly values in [-2, 2]; with `special` set, about one in 64 is NaN,
   +-infinity or -0.0, to show the kernels treat them alike */
static float rand_value(int special) {
    const uint32_t r = rand_u32();
    if (special && r % 64 == 0) {
        switch ((r >> 6) % 4) {
            case 0: return (float)NAN;
            case 1: return (float)INFINITY;
            case 2: return -(float)INFINITY;
            default: return -0.0f;
        }
    }
    return (float)(r % 4096) / 1024.0f - 2.0f + (float)(r % 7) * 1e-4f;
}

#endif /* NULL_RUNTIME_TEST_CHECK_H */
