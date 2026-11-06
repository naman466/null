/* tests for runtime/null/elementwise.h: relu, clip, sigmoid, tanh and the binary operators; included by test_kernels.c */

static void test_relu(void) {
    float x[6] = {-2.0f, -0.5f, 0.0f, 0.5f, 3.0f, -1e-30f};
    float want[6] = {0.0f, 0.0f, 0.0f, 0.5f, 3.0f, 0.0f};
    float y[6];
    null_relu_f32(x, y, 6);
    CHECK(all_equal(y, want, 6));

    /* in place */
    null_relu_f32(x, x, 6);
    CHECK(all_equal(x, want, 6));
}

static void test_relu_special_values(void) {
    float x[4], y[4];
    x[0] = (float)NAN;
    x[1] = (float)INFINITY;
    x[2] = -(float)INFINITY;
    x[3] = -0.0f;
    null_relu_f32(x, y, 4);
    CHECK(isnan(y[0]));                       /* NaN passes through */
    CHECK(y[1] == (float)INFINITY);
    CHECK(same_bits(y[2], 0.0f));
    CHECK(same_bits(y[3], -0.0f));            /* -0 is not < 0 */
    null_relu_f32(x, y, 0);                   /* empty: no access */
}

/* ---- binary, same shape ---- */

static void test_binary_flat(void) {
    float a[4] = {1.0f, 2.0f, -3.0f, 8.0f};
    float b[4] = {0.5f, -2.0f, 4.0f, 2.0f};
    float y[4];
    float add[4] = {1.5f, 0.0f, 1.0f, 10.0f};
    float sub[4] = {0.5f, 4.0f, -7.0f, 6.0f};
    float mul[4] = {0.5f, -4.0f, -12.0f, 16.0f};
    float div[4] = {2.0f, -1.0f, -0.75f, 4.0f};

    null_add_f32(a, b, y, 4); CHECK(all_equal(y, add, 4));
    null_sub_f32(a, b, y, 4); CHECK(all_equal(y, sub, 4));
    null_mul_f32(a, b, y, 4); CHECK(all_equal(y, mul, 4));
    null_div_f32(a, b, y, 4); CHECK(all_equal(y, div, 4));

    /* in place on the first input */
    null_add_f32(a, b, a, 4);
    CHECK(all_equal(a, add, 4));
}

static void test_div_ieee(void) {
    float a[3] = {1.0f, -1.0f, 0.0f};
    float b[3] = {0.0f, 0.0f, 0.0f};
    float y[3];
    null_div_f32(a, b, y, 3);
    CHECK(y[0] == (float)INFINITY);
    CHECK(y[1] == -(float)INFINITY);
    CHECK(isnan(y[2]));
}

/* ---- binary, broadcasting ---- */

static void test_bcast_bias_row(void) {
    /* [2, 3] + [3] */
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[3] = {10, 20, 30};
    float want[6] = {11, 22, 33, 14, 25, 36};
    float y[6];
    int64_t shape[2] = {2, 3}, a_shape[2] = {2, 3}, b_shape[1] = {3};
    int64_t sa[2], sb[2];
    null_broadcast_strides(2, shape, 2, a_shape, sa);
    null_broadcast_strides(2, shape, 1, b_shape, sb);
    null_add_bcast_f32(a, b, y, 2, shape, sa, sb);
    CHECK(all_equal(y, want, 6));
}

static void test_bcast_outer(void) {
    /* [2, 1] * [1, 3] -> [2, 3], an outer product */
    float a[2] = {2, 3};
    float b[3] = {1, 10, 100};
    float want[6] = {2, 20, 200, 3, 30, 300};
    float y[6];
    int64_t shape[2] = {2, 3}, a_shape[2] = {2, 1}, b_shape[2] = {1, 3};
    int64_t sa[2], sb[2];
    null_broadcast_strides(2, shape, 2, a_shape, sa);
    null_broadcast_strides(2, shape, 2, b_shape, sb);
    null_mul_bcast_f32(a, b, y, 2, shape, sa, sb);
    CHECK(all_equal(y, want, 6));
}

static void test_bcast_scalar(void) {
    /* [2, 2] - scalar, and scalar - [2, 2] */
    float a[4] = {1, 2, 3, 4};
    float s[1] = {1};
    float y[4];
    float want1[4] = {0, 1, 2, 3};
    float want2[4] = {0, -1, -2, -3};
    int64_t shape[2] = {2, 2}, a_shape[2] = {2, 2};
    int64_t sa[2], ss[2];
    null_broadcast_strides(2, shape, 2, a_shape, sa);
    null_broadcast_strides(2, shape, 0, a_shape, ss);
    null_sub_bcast_f32(a, s, y, 2, shape, sa, ss);
    CHECK(all_equal(y, want1, 4));
    null_sub_bcast_f32(s, a, y, 2, shape, ss, sa);
    CHECK(all_equal(y, want2, 4));
}

static void test_bcast_rank0_and_empty(void) {
    float a[1] = {6}, b[1] = {3}, y[1] = {0};
    int64_t none[1] = {0};
    int64_t empty[2] = {4, 0}, sa[2] = {0, 0}, sb[2] = {0, 0};
    null_div_bcast_f32(a, b, y, 0, none, none, none);   /* scalar / scalar */
    CHECK(y[0] == 2.0f);
    null_add_bcast_f32(a, b, y, 2, empty, sa, sb);      /* zero elements, no access */
    CHECK(y[0] == 2.0f);
}

static void test_bcast_rank4(void) {
    /* [8, 1, 6, 1] + [7, 1, 5] -> [8, 7, 6, 5], checked against a direct index computation */
    enum { N = 8 * 7 * 6 * 5 };
    static float a[8 * 6], b[7 * 5], y[N];
    int64_t shape[4] = {8, 7, 6, 5}, a_shape[4] = {8, 1, 6, 1}, b_shape[3] = {7, 1, 5};
    int64_t sa[4], sb[4];
    int i0, i1, i2, i3, ok = 1;
    for (i0 = 0; i0 < 8 * 6; ++i0) a[i0] = (float)i0;
    for (i0 = 0; i0 < 7 * 5; ++i0) b[i0] = (float)(1000 * i0);
    null_broadcast_strides(4, shape, 4, a_shape, sa);
    null_broadcast_strides(4, shape, 3, b_shape, sb);
    null_add_bcast_f32(a, b, y, 4, shape, sa, sb);
    for (i0 = 0; i0 < 8; ++i0)
        for (i1 = 0; i1 < 7; ++i1)
            for (i2 = 0; i2 < 6; ++i2)
                for (i3 = 0; i3 < 5; ++i3) {
                    float want = a[i0 * 6 + i2] + b[i1 * 5 + i3];
                    float got = y[((i0 * 7 + i1) * 6 + i2) * 5 + i3];
                    if (!same_bits(got, want)) ok = 0;
                }
    CHECK(ok);
}

static void test_bcast_in_place(void) {
    /* y aliases the full-shape input: x += bias */
    float x[6] = {1, 2, 3, 4, 5, 6};
    float bias[3] = {1, 1, 1};
    float want[6] = {2, 3, 4, 5, 6, 7};
    int64_t shape[2] = {2, 3}, b_shape[1] = {3};
    int64_t sx[2], sb[2];
    null_broadcast_strides(2, shape, 2, shape, sx);
    null_broadcast_strides(2, shape, 1, b_shape, sb);
    null_add_bcast_f32(x, bias, x, 2, shape, sx, sb);
    CHECK(all_equal(x, want, 6));
}

static void test_clip(void) {
    /* relu6, and a one-sided bound */
    float x[6] = {-2, 0, 3, 6, 7.5f, 100}, y[6];
    float want6[6] = {0, 0, 3, 6, 6, 6}, want_lo[6] = {-1, 0, 3, 6, 7.5f, 100};
    null_clip_f32(x, y, 6, 0.0f, 6.0f);
    CHECK(all_equal(y, want6, 6));
    null_clip_f32(x, y, 6, -1.0f, INFINITY);
    CHECK(all_equal(y, want_lo, 6));
    /* in place */
    null_clip_f32(x, x, 6, 0.0f, 6.0f);
    CHECK(all_equal(x, want6, 6));
}

static void test_clip_special_values(void) {
    /* NaN passes through; -0.0 is kept, as relu keeps it; infinities are clipped
       like any number; with min > max every output is max */
    float nan = NAN;
    float x[5] = {nan, -0.0f, INFINITY, -INFINITY, 1}, y[5];
    float want[5] = {nan, -0.0f, 6, 0, 1}, crossed[5] = {2, 2, 2, 2, 2};
    null_clip_f32(x, y, 5, 0.0f, 6.0f);
    CHECK(y[0] != y[0]);
    CHECK(all_equal(y + 1, want + 1, 4));
    null_clip_f32(x + 1, y, 4, 5.0f, 2.0f);
    CHECK(all_equal(y, crossed, 4));
}

static void test_sigmoid(void) {
    /* exact where the answer is: sigmoid(0) = 0.5; the rest within an ulp or two
       of the true value, computed in double; sigmoid(-x) = 1 - sigmoid(x) */
    float x[7] = {0.0f, 1.0f, -1.0f, 4.0f, -4.0f, 20.0f, -20.0f}, y[7];
    int i;
    null_sigmoid_f32(x, y, 7);
    CHECK(y[0] == 0.5f);
    for (i = 1; i < 7; ++i) {
        double want = 1.0 / (1.0 + exp(-(double)x[i]));
        CHECK(fabs(y[i] - want) <= 2.5e-7 * want);
    }
    CHECK(fabs((double)y[1] + y[2] - 1.0) < 1e-7);
    /* very negative x keeps its tiny result instead of rounding to 0: exp(-80) */
    x[0] = -80.0f;
    null_sigmoid_f32(x, y, 1);
    CHECK(y[0] > 0.0f && fabs(y[0] - exp(-80.0)) <= 1e-6 * exp(-80.0));
    /* in place */
    x[0] = 0.0f;
    null_sigmoid_f32(x, x, 7);
    CHECK(x[0] == 0.5f && x[5] == y[5] && x[6] == y[6]);
}

static void test_sigmoid_special_values(void) {
    /* the ends are exact, NaN passes through, -0 is 0.5 like +0, and nothing overflows */
    float x[6] = {INFINITY, -INFINITY, NAN, -0.0f, 1000.0f, -1000.0f}, y[6];
    null_sigmoid_f32(x, y, 6);
    CHECK(y[0] == 1.0f);
    CHECK(same_bits(y[1], 0.0f));
    CHECK(y[2] != y[2]);
    CHECK(y[3] == 0.5f);
    CHECK(y[4] == 1.0f);
    CHECK(same_bits(y[5], 0.0f));
    null_sigmoid_f32(x, y, 0);
}

static void test_tanh(void) {
    float x[6] = {0.5f, -0.5f, 2.0f, -3.0f, 10.0f, 1e-8f}, y[6];
    int i;
    null_tanh_f32(x, y, 6);
    for (i = 0; i < 6; ++i) CHECK(fabs(y[i] - tanh((double)x[i])) <= 2.5e-7 * fabs(tanh((double)x[i])));
    CHECK(y[0] == -y[1]);                     /* odd */
    null_tanh_f32(x, x, 6);                   /* in place */
    CHECK(all_equal(x, y, 6));
}

static void test_tanh_special_values(void) {
    float x[5] = {INFINITY, -INFINITY, NAN, -0.0f, 0.0f}, y[5];
    null_tanh_f32(x, y, 5);
    CHECK(y[0] == 1.0f && y[1] == -1.0f);
    CHECK(y[2] != y[2]);
    CHECK(same_bits(y[3], -0.0f));            /* tanh keeps the sign of zero */
    CHECK(same_bits(y[4], 0.0f));
}

static void run_elementwise_tests(void) {
    test_sigmoid();
    test_sigmoid_special_values();
    test_tanh();
    test_tanh_special_values();
    test_clip();
    test_clip_special_values();
    test_relu();
    test_relu_special_values();
    test_binary_flat();
    test_div_ieee();
    test_bcast_bias_row();
    test_bcast_outer();
    test_bcast_scalar();
    test_bcast_rank0_and_empty();
    test_bcast_rank4();
    test_bcast_in_place();
}
