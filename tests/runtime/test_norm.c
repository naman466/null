/* tests for runtime/null/norm.h: layer normalization; included by test_kernels.c */

static void test_layernorm_by_hand(void) {
    /* [1, 2, 3, 4]: mean 2.5, variance 1.25; then a scale and a shift per position */
    float x[4] = {1, 2, 3, 4}, y[4];
    float one[4] = {1, 1, 1, 1}, scale[4] = {1, 2, -1, 0.5f}, bias[4] = {0, 1, 0, -1};
    float want[4];
    int i;
    for (i = 0; i < 4; ++i) want[i] = (float)((i - 1.5) / sqrt(1.25));
    null_layernorm_f32(x, one, NULL, y, 1, 4, 0.0f);
    CHECK(all_equal(y, want, 4));
    for (i = 0; i < 4; ++i) want[i] = (float)((i - 1.5) / sqrt(1.25) * scale[i] + bias[i]);
    null_layernorm_f32(x, scale, bias, y, 1, 4, 0.0f);
    CHECK(all_equal(y, want, 4));
}

static void test_layernorm_rows_and_epsilon(void) {
    /* rows are independent; a constant row has no spread, so epsilon alone
       keeps the division finite and the output is exactly the bias */
    float x[6] = {5, 5, 5, -1, 0, 1}, y[6];
    float one[3] = {1, 1, 1}, bias[3] = {0.25f, 0, -2};
    float want_row1[3];
    int i;
    for (i = 0; i < 3; ++i) want_row1[i] = (float)((i - 1) / sqrt(2.0 / 3.0 + (double)1e-5f) + bias[i]);
    null_layernorm_f32(x, one, bias, y, 2, 3, 1e-5f);
    CHECK(all_equal(y, bias, 3));
    CHECK(all_equal(y + 3, want_row1, 3));
}

static void test_layernorm_large_offset(void) {
    /* 1e7 + [1, 2, 3, 4] is exact in float, and normalises to the same values
       as [1, 2, 3, 4]: summing in double keeps the spread that a float
       sum of squares would lose */
    float x[4] = {10000001.0f, 10000002.0f, 10000003.0f, 10000004.0f}, base[4] = {1, 2, 3, 4};
    float one[4] = {1, 1, 1, 1}, y[4], want[4];
    null_layernorm_f32(base, one, NULL, want, 1, 4, 1e-5f);
    null_layernorm_f32(x, one, NULL, y, 1, 4, 1e-5f);
    CHECK(all_equal(y, want, 4));
}

static void test_layernorm_special_values(void) {
    /* a NaN or an infinity makes its whole row NaN, and only its row; in place works */
    float x[6] = {1, NAN, 3, 1, INFINITY, 3}, one[3] = {1, 1, 1};
    float good[3] = {4, 5, 6}, want[3];
    int i;
    null_layernorm_f32(good, one, NULL, want, 1, 3, 0.0f);
    null_layernorm_f32(x, one, NULL, x, 2, 3, 0.0f);
    for (i = 0; i < 6; ++i) CHECK(x[i] != x[i]);
    null_layernorm_f32(good, one, NULL, good, 1, 3, 0.0f);
    CHECK(all_equal(good, want, 3));
}

static void run_norm_tests(void) {
    test_layernorm_by_hand();
    test_layernorm_rows_and_epsilon();
    test_layernorm_large_offset();
    test_layernorm_special_values();
}
