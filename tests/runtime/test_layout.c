/* tests for runtime/null/layout.h: transpose and gather; included by test_kernels.c */

static void test_transpose_matrix(void) {
    /* [2, 3] -> [3, 2]: output dim 0 is input dim 1 (stride 1), output dim 1 is input dim 0 (stride 3) */
    float x[6] = {1, 2, 3,
                  4, 5, 6};
    float y[6], want[6] = {1, 4, 2, 5, 3, 6};
    int64_t shape[2] = {3, 2}, strides[2] = {1, 3};
    null_transpose_f32(x, y, 2, shape, strides);
    CHECK(all_equal(y, want, 6));
}

static void test_transpose_three_dims(void) {
    /* x [2, 3, 2] with perm [2, 0, 1]: y [2, 2, 3], y[a][b][c] = x[b][c][a] */
    float x[12], y[12], want[12];
    int64_t shape[3] = {2, 2, 3}, strides[3] = {1, 6, 2};
    int a, b, c;
    for (a = 0; a < 12; ++a) x[a] = (float)a;
    for (a = 0; a < 2; ++a)
        for (b = 0; b < 2; ++b)
            for (c = 0; c < 3; ++c) want[(a * 2 + b) * 3 + c] = x[(b * 3 + c) * 2 + a];
    null_transpose_f32(x, y, 3, shape, strides);
    CHECK(all_equal(y, want, 12));

    /* rank 0 copies the single element */
    null_transpose_f32(x + 7, y, 0, shape, strides);
    CHECK(same_bits(y[0], 7.0f));
}

static void test_gather(void) {
    /* x [2, 3, 2] viewed as [outer 2, n 3, inner 2]: indices pick blocks of 2,
       in any order, repeated, in every outer slice */
    float x[12], y[8];
    float want[8] = {4, 5, 0, 1, 4, 5, 4, 5};
    int64_t idx[2] = {2, 0}, again[2] = {2, 2};
    int i;
    for (i = 0; i < 12; ++i) x[i] = (float)i;
    null_gather_f32(x, y, 1, 3, 2, idx, 2);                 /* the first slice only */
    CHECK(all_equal(y, want, 4));
    null_gather_f32(x, y, 1, 3, 2, again, 2);
    CHECK(all_equal(y, want + 4, 4));
    {
        float want2[8] = {4, 5, 0, 1, 10, 11, 6, 7};
        null_gather_f32(x, y, 2, 3, 2, idx, 2);
        CHECK(all_equal(y, want2, 8));
    }
}

static void run_layout_tests(void) {
    test_transpose_matrix();
    test_transpose_three_dims();
    test_gather();
}
