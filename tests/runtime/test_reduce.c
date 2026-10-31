/* tests for runtime/null/reduce.h: the mean over a set of axes; included by test_kernels.c */

static null_reduce_params reduce_params(int rank, const int64_t* shape, const int* reduce) {
    null_reduce_params p;
    int d;
    memset(&p, 0, sizeof p);
    p.rank = rank;
    for (d = 0; d < rank; ++d) {
        p.shape[d] = shape[d];
        p.reduce[d] = reduce[d];
    }
    return p;
}

static void test_reduce_mean_last_axis(void) {
    /* [2, 3] -> [2]: the mean of each row, as GlobalAveragePool sees [n*c, h*w] */
    float x[6] = {1, 2, 3, 4, 5, 9}, y[2], want[2] = {2, 6};
    int64_t shape[2] = {2, 3};
    int reduce[2] = {0, 1};
    null_reduce_params p = reduce_params(2, shape, reduce);
    null_reduce_mean_f32(x, y, &p);
    CHECK(all_equal(y, want, 2));
}

static void test_reduce_mean_first_axis(void) {
    /* [2, 3] -> [3]: the mean of each column, strided reads */
    float x[6] = {1, 2, 3, 5, 8, -3}, y[3], want[3] = {3, 5, 0};
    int64_t shape[2] = {2, 3};
    int reduce[2] = {1, 0};
    null_reduce_params p = reduce_params(2, shape, reduce);
    null_reduce_mean_f32(x, y, &p);
    CHECK(all_equal(y, want, 3));
}

static void test_reduce_mean_alternating(void) {
    /* [2, 2, 2] reducing dims 0 and 2: y[j] averages x[i][j][k] for all i, k */
    float x[8] = {1, 2, 3, 4,
                  5, 6, 7, 8};
    float y[2], want[2] = {(1 + 2 + 5 + 6) / 4.0f, (3 + 4 + 7 + 8) / 4.0f};
    int64_t shape[3] = {2, 2, 2};
    int reduce[3] = {1, 0, 1};
    null_reduce_params p = reduce_params(3, shape, reduce);
    null_reduce_mean_f32(x, y, &p);
    CHECK(all_equal(y, want, 2));
}

static void test_reduce_mean_everything_and_nothing(void) {
    /* every dim reduced: one mean. rank 0: the single element as it is */
    float x[4] = {1, 2, 3, 6}, y[1], want[1] = {3};
    int64_t shape[1] = {4};
    int reduce[1] = {1};
    null_reduce_params p = reduce_params(1, shape, reduce);
    null_reduce_mean_f32(x, y, &p);
    CHECK(all_equal(y, want, 1));

    p = reduce_params(0, shape, reduce);
    null_reduce_mean_f32(x, y, &p);
    CHECK(same_bits(y[0], 1.0f));
}

static void test_reduce_mean_rounds_once(void) {
    /* 1 + 2^-24 + 2^-24: summed in float, each 2^-24 is lost (1 + 2^-24 rounds
       back to 1) and the mean is 1/3; summed in double the total is 1 + 2^-23,
       which rounds to the float just above 1/3 after dividing by 3 */
    float tiny = 1.0f / 16777216.0f;
    float x[3] = {1.0f, tiny, tiny}, y[1];
    int64_t shape[1] = {3};
    int reduce[1] = {1};
    null_reduce_params p = reduce_params(1, shape, reduce);
    null_reduce_mean_f32(x, y, &p);
    CHECK(same_bits(y[0], (float)((1.0 + 2.0 * tiny) / 3.0)));
    CHECK(!same_bits(y[0], 1.0f / 3.0f));
}

static void test_reduce_mean_special_values(void) {
    /* NaN and +-inf propagate like a sum; a large sum does not overflow
       before the division; an empty set gives NaN */
    float big = 3e38f;
    float x[8] = {1, NAN, INFINITY, 1, INFINITY, -INFINITY, big, big};
    float y[4];
    int64_t shape[2] = {4, 2};
    int reduce[2] = {0, 1};
    null_reduce_params p = reduce_params(2, shape, reduce);
    null_reduce_mean_f32(x, y, &p);
    CHECK(y[0] != y[0]);
    CHECK(same_bits(y[1], INFINITY));
    CHECK(y[2] != y[2]);
    CHECK(same_bits(y[3], big));

    shape[1] = 0;
    p = reduce_params(2, shape, reduce);
    null_reduce_mean_f32(x, y, &p);
    CHECK(y[0] != y[0] && y[3] != y[3]);
}

static void run_reduce_tests(void) {
    test_reduce_mean_last_axis();
    test_reduce_mean_first_axis();
    test_reduce_mean_alternating();
    test_reduce_mean_everything_and_nothing();
    test_reduce_mean_rounds_once();
    test_reduce_mean_special_values();
}
