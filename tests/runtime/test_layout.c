/* tests for runtime/null/layout.h: transpose, gather and concat; included by test_kernels.c */

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

static void test_concat(void) {
    /* [2, 2, 2] and [2, 1, 2] along axis 1 into [2, 3, 2]: outer 2, inner 2 */
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8}, b[4] = {-1, -2, -3, -4}, y[12];
    float want[12] = {1, 2, 3, 4, -1, -2, 5, 6, 7, 8, -3, -4};
    float want_first[12] = {-1, -2, 1, 2, 3, 4, -3, -4, 5, 6, 7, 8};
    null_concat_f32(a, y, 2, 2, 3, 0, 2);
    null_concat_f32(b, y, 2, 1, 3, 2, 2);
    CHECK(all_equal(y, want, 12));
    /* the other order: the offsets, not the calls, decide where each goes */
    null_concat_f32(a, y, 2, 2, 3, 1, 2);
    null_concat_f32(b, y, 2, 1, 3, 0, 2);
    CHECK(all_equal(y, want_first, 12));
    /* axis 0 (outer 1) is two plain copies, end to end */
    null_concat_f32(a, y, 1, 8, 12, 0, 1);
    null_concat_f32(b, y, 1, 4, 12, 8, 1);
    CHECK(all_equal(y, a, 8) && all_equal(y + 8, b, 4));
    /* NaN and -0.0 are copied as they are */
    a[0] = NAN;
    a[1] = -0.0f;
    null_concat_f32(a, y, 1, 2, 2, 0, 1);
    CHECK(y[0] != y[0] && same_bits(y[1], -0.0f));
}

static void test_slice(void) {
    /* x[3, 4] = 0..11; x[1:3, ::-2] starts at element 1*4 + 3, walks rows by 4 and columns by -2 */
    float x[12], y[4];
    float want[4] = {7, 5, 11, 9};
    int64_t shape[2] = {2, 2}, strides[2] = {4, -2};
    int64_t one[1] = {3}, unit[1] = {4};
    int i;
    for (i = 0; i < 12; ++i) x[i] = (float)i;
    null_slice_f32(x, y, 2, shape, strides, 7);
    CHECK(all_equal(y, want, 4));
    /* one dim, every fourth element from 2: a column */
    null_slice_f32(x, y, 1, one, unit, 2);
    CHECK(y[0] == 2 && y[1] == 6 && y[2] == 10);
    /* NaN and -0.0 are copied as they are */
    x[7] = NAN;
    x[5] = -0.0f;
    null_slice_f32(x, y, 2, shape, strides, 7);
    CHECK(y[0] != y[0] && same_bits(y[1], -0.0f));
}

static void run_layout_tests(void) {
    test_slice();
    test_concat();
    test_transpose_matrix();
    test_transpose_three_dims();
    test_gather();
}
