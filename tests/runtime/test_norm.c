/* tests for runtime/null/norm.h: layer and batch normalization; included by test_kernels.c */

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

static void test_batchnorm_by_hand(void) {
    /* x[2, 2, 3]: two images, two channels of three values. each channel has
       its own recorded mean and variance; the result is the formula in double,
       rounded once */
    float x[12] = {1, 2, 3, 4, 5, 6, -1, 0, 1, 10, 20, 30}, y[12], want[12];
    float scale[2] = {2.0f, 0.5f}, bias[2] = {1.0f, -1.0f}, mean[2] = {2.0f, 15.0f}, var[2] = {4.0f, 0.25f};
    int b, k, i;
    for (b = 0; b < 2; ++b)
        for (k = 0; k < 2; ++k)
            for (i = 0; i < 3; ++i) {
                int j = (b * 2 + k) * 3 + i;
                want[j] = (float)((x[j] - (double)mean[k]) / sqrt((double)var[k] + (double)1e-5f) * scale[k] + bias[k]);
            }
    null_batchnorm_f32(x, scale, bias, mean, var, y, 2, 2, 3, 1e-5f);
    CHECK(all_equal(y, want, 12));
    /* exact values with epsilon 0: channel 0 is (x - 2) / 2 * 2 + 1 = x - 1 */
    null_batchnorm_f32(x, scale, bias, mean, var, y, 2, 2, 3, 0.0f);
    CHECK(y[0] == 0.0f && y[1] == 1.0f && y[2] == 2.0f);
    CHECK(y[3] == (4 - 15) / 0.5f * 0.5f - 1);
    /* in place */
    null_batchnorm_f32(x, scale, bias, mean, var, x, 2, 2, 3, 1e-5f);
    CHECK(all_equal(x, want, 12));
}

static void test_batchnorm_layouts(void) {
    /* [N, C] (BatchNorm1d, inner = 1) and [N, C, H, W] give per-channel results
       that do not depend on where in the image a value sits */
    float x2[6] = {1, 2, 3, 4, 5, 6}, y2[6];
    float one[3] = {1, 1, 1}, zero[3] = {0, 0, 0}, mean[3] = {1, 2, 3}, var[3] = {1, 1, 1};
    float want2[6] = {0, 0, 0, 3, 3, 3};
    float x4[8] = {7, 7, 7, 7, 9, 9, 9, 9}, y4[8];
    null_batchnorm_f32(x2, one, zero, mean, var, y2, 2, 3, 1, 0.0f);
    CHECK(all_equal(y2, want2, 6));
    null_batchnorm_f32(x4, one, zero, mean, var, y4, 1, 2, 4, 0.0f);
    CHECK(y4[0] == 6 && y4[3] == 6 && y4[4] == 7 && y4[7] == 7);
}

static void test_batchnorm_special_values(void) {
    /* NaN and inf stay in their own element; a negative variance makes the
       channel NaN, as the formula does; a zero variance is kept finite by epsilon */
    float x[4] = {NAN, INFINITY, 1.0f, 2.0f}, y[4];
    float one[2] = {1, 1}, zero[2] = {0, 0}, mean[2] = {0, 0}, var[2] = {1, -1};
    float zvar[2] = {0, 0};
    null_batchnorm_f32(x, one, zero, mean, var, y, 1, 2, 2, 0.0f);
    CHECK(y[0] != y[0]);
    CHECK(y[1] == INFINITY);
    CHECK(y[2] != y[2] && y[3] != y[3]);
    null_batchnorm_f32(x + 2, one, zero, mean, zvar, y, 1, 2, 1, 1e-5f);
    CHECK(y[0] == (float)(1.0 / sqrt((double)1e-5f)));
    null_batchnorm_f32(x, one, zero, mean, var, y, 0, 2, 2, 0.0f);   /* empty batch: no access */
}

static void run_norm_tests(void) {
    test_batchnorm_by_hand();
    test_batchnorm_layouts();
    test_batchnorm_special_values();
    test_layernorm_by_hand();
    test_layernorm_rows_and_epsilon();
    test_layernorm_large_offset();
    test_layernorm_special_values();
}
