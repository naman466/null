/* tests for runtime/null/softmax.h; included by test_kernels.c */

/* softmax along the middle of [outer, n, inner] in double, for comparison */
static void ref_softmax(const float* x, double* y, int64_t outer, int64_t n, int64_t inner) {
    int64_t o, i, k;
    for (o = 0; o < outer; ++o)
        for (i = 0; i < inner; ++i) {
            double m = x[o * n * inner + i], s = 0;
            for (k = 1; k < n; ++k) if (x[(o * n + k) * inner + i] > m) m = x[(o * n + k) * inner + i];
            for (k = 0; k < n; ++k) s += exp((double)x[(o * n + k) * inner + i] - m);
            for (k = 0; k < n; ++k) y[(o * n + k) * inner + i] = exp((double)x[(o * n + k) * inner + i] - m) / s;
        }
}

/* every element within rel relative error of the double reference */
static int close_to(const float* got, const double* want, int64_t count, double rel) {
    int64_t i;
    for (i = 0; i < count; ++i)
        if (fabs((double)got[i] - want[i]) > rel * fabs(want[i]) + 1e-45) return 0;
    return 1;
}

static void test_softmax_row(void) {
    float x[3] = {1.0f, 2.0f, 3.0f}, y[3];
    double want[3];
    null_softmax_f32(x, y, 1, 3, 1);
    ref_softmax(x, want, 1, 3, 1);
    CHECK(close_to(y, want, 3, 1e-6));
    CHECK(fabs((double)y[0] + y[1] + y[2] - 1.0) < 1e-6);
    CHECK(y[0] < y[1] && y[1] < y[2]);
}

static void test_softmax_axes(void) {
    /* [2, 3, 4] along the middle axis (outer 2, n 3, inner 4), against the reference,
       and along the last axis as [6, 4, 1] */
    float x[24], y[24];
    double want[24];
    int i;
    for (i = 0; i < 24; ++i) x[i] = (float)((i * 7) % 11) * 0.5f - 2.0f;
    null_softmax_f32(x, y, 2, 3, 4);
    ref_softmax(x, want, 2, 3, 4);
    CHECK(close_to(y, want, 24, 1e-6));
    null_softmax_f32(x, y, 6, 4, 1);
    ref_softmax(x, want, 6, 4, 1);
    CHECK(close_to(y, want, 24, 1e-6));
}

static void test_softmax_large_values(void) {
    /* without subtracting the max, exp(1000) would overflow to inf and give NaN */
    float x[3] = {1000.0f, 1001.0f, 1002.0f}, small[3] = {0.0f, 1.0f, 2.0f}, y[3], z[3];
    null_softmax_f32(x, y, 1, 3, 1);
    null_softmax_f32(small, z, 1, 3, 1);
    CHECK(!isnan(y[0]) && !isnan(y[2]));
    CHECK(fabs((double)y[2] - z[2]) < 1e-6);   /* shifting every input changes nothing */
}

static void test_softmax_special_values(void) {
    float x[3], y[3];
    /* -inf contributes exactly 0 */
    x[0] = -(float)INFINITY; x[1] = 0.0f; x[2] = 0.0f;
    null_softmax_f32(x, y, 1, 3, 1);
    CHECK(y[0] == 0.0f && y[1] == 0.5f && y[2] == 0.5f);
    /* NaN anywhere poisons the slice, wherever it appears */
    x[0] = 1.0f; x[1] = (float)NAN; x[2] = 2.0f;
    null_softmax_f32(x, y, 1, 3, 1);
    CHECK(isnan(y[0]) && isnan(y[1]) && isnan(y[2]));
    x[0] = (float)NAN; x[1] = 5.0f; x[2] = 2.0f;
    null_softmax_f32(x, y, 1, 3, 1);
    CHECK(isnan(y[0]) && isnan(y[1]) && isnan(y[2]));
    /* +inf gives inf - inf = NaN, like the reference */
    x[0] = (float)INFINITY; x[1] = 0.0f; x[2] = 0.0f;
    null_softmax_f32(x, y, 1, 3, 1);
    CHECK(isnan(y[0]) && isnan(y[1]));
    /* all -inf: no probability anywhere, NaN like the reference */
    x[0] = x[1] = x[2] = -(float)INFINITY;
    null_softmax_f32(x, y, 1, 3, 1);
    CHECK(isnan(y[0]));
}

static void test_softmax_sizes(void) {
    float x[4] = {3.0f, -1.0f, 0.5f, 8.0f}, y[4] = {-1, -1, -1, -1};
    /* n = 1: every output is exactly 1 */
    null_softmax_f32(x, y, 4, 1, 1);
    CHECK(y[0] == 1.0f && y[3] == 1.0f);
    /* n = 0 or outer = 0: nothing written */
    y[0] = -1.0f;
    null_softmax_f32(x, y, 4, 0, 1);
    null_softmax_f32(x, y, 0, 4, 1);
    CHECK(y[0] == -1.0f);
    /* equal inputs: exactly uniform */
    x[0] = x[1] = x[2] = x[3] = 7.0f;
    null_softmax_f32(x, y, 1, 4, 1);
    CHECK(y[0] == 0.25f && y[3] == 0.25f);
}

static void test_softmax_in_place(void) {
    float x[4] = {0.0f, 1.0f, 2.0f, 3.0f}, y[4];
    null_softmax_f32(x, y, 1, 4, 1);
    null_softmax_f32(x, x, 1, 4, 1);
    CHECK(all_equal(x, y, 4));
}

static void run_softmax_tests(void) {
    test_softmax_row();
    test_softmax_axes();
    test_softmax_large_values();
    test_softmax_special_values();
    test_softmax_sizes();
    test_softmax_in_place();
}
