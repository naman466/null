/* tests for runtime/null/pool.h: 2-D max pooling; included by test_kernels.c */

static null_maxpool2d_params pool_params(int64_t n, int64_t c, int64_t h, int64_t w, int64_t kh, int64_t kw,
                                         int64_t stride, int64_t dil, int64_t pad_top, int64_t pad_left,
                                         int64_t out_h, int64_t out_w) {
    null_maxpool2d_params p;
    p.n = n; p.c = c; p.h = h; p.w = w; p.kh = kh; p.kw = kw;
    p.stride_h = p.stride_w = stride;
    p.dil_h = p.dil_w = dil;
    p.pad_top = pad_top; p.pad_left = pad_left;
    p.out_h = out_h; p.out_w = out_w;
    return p;
}

static void test_maxpool_by_hand(void) {
    /* 4x4, 2x2 windows, stride 2: the max of each quadrant */
    float x[16] = {1, 5, 2, 0,
                   3, 4, 8, 1,
                   0, -1, -3, -2,
                   9, 2, -4, -5};
    float y[4], want[4] = {5, 8, 9, -2};
    null_maxpool2d_params p = pool_params(1, 1, 4, 4, 2, 2, 2, 1, 0, 0, 2, 2);
    null_maxpool2d_f32(x, y, &p);
    CHECK(all_equal(y, want, 4));
}

static void test_maxpool_channels_and_padding(void) {
    /* two channels, 3x3 windows, stride 1, pad 1: the padding never wins,
       even when every pixel is negative */
    float x[8] = {-1, -2, -3, -4,     /* channel 0, 2x2 */
                  10, 20, 30, 40};    /* channel 1 */
    float y[8], want[8] = {-1, -1, -1, -1, 40, 40, 40, 40};
    null_maxpool2d_params p = pool_params(1, 2, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2);
    null_maxpool2d_f32(x, y, &p);
    CHECK(all_equal(y, want, 8));
}

static void test_maxpool_ceil_partial_window(void) {
    /* 3 wide, 2 wide windows with stride 2 and ceil_mode: the last window
       hangs off the edge and sees only one pixel */
    float x[3] = {1, 7, 4}, y[2], want[2] = {7, 4};
    null_maxpool2d_params p = pool_params(1, 1, 1, 3, 1, 2, 2, 1, 0, 0, 1, 2);
    null_maxpool2d_f32(x, y, &p);
    CHECK(all_equal(y, want, 2));
}

static void test_maxpool_dilation(void) {
    /* 1x5 row, a 1x2 kernel with dilation 3 compares x[i] and x[i + 3] */
    float x[5] = {1, 9, 2, 5, 3}, y[2], want[2] = {5, 9};
    null_maxpool2d_params p = pool_params(1, 1, 1, 5, 1, 2, 1, 3, 0, 0, 1, 2);
    null_maxpool2d_f32(x, y, &p);
    CHECK(all_equal(y, want, 2));
}

static void test_maxpool_special_values(void) {
    /* NaN is ignored while the window has a number, in any position;
       a window of only NaN gives that NaN; infinities and -0.0 are values */
    float nan = NAN;
    float x[8] = {nan, 1, 2, nan,       /* windows: {nan, 1}, {2, nan} */
                  nan, nan, -INFINITY, -INFINITY};
    float y[4];
    null_maxpool2d_params p = pool_params(1, 1, 1, 8, 1, 2, 2, 1, 0, 0, 1, 4);
    null_maxpool2d_f32(x, y, &p);
    CHECK(y[0] == 1.0f);
    CHECK(y[1] == 2.0f);
    CHECK(isnan(y[2]) && same_bits(y[2], nan));
    CHECK(isinf(y[3]) && y[3] < 0);

    {
        /* ties between -0.0 and +0.0 keep the first, like a strict > comparison */
        float z[2] = {-0.0f, 0.0f}, out[1];
        null_maxpool2d_params q = pool_params(1, 1, 1, 2, 1, 2, 1, 1, 0, 0, 1, 1);
        null_maxpool2d_f32(z, out, &q);
        CHECK(same_bits(out[0], -0.0f));
    }
}

static void test_maxpool_no_pixel(void) {
    /* with dilation a window can fall entirely in the padding: -inf, the max of nothing */
    float x[1] = {5}, y[3];
    null_maxpool2d_params p = pool_params(1, 1, 1, 1, 1, 2, 1, 2, 0, 1, 1, 1);
    null_maxpool2d_f32(x, y, &p);    /* taps at -1 and 1: both outside */
    CHECK(isinf(y[0]) && y[0] < 0);
}

static void run_pool_tests(void) {
    test_maxpool_by_hand();
    test_maxpool_channels_and_padding();
    test_maxpool_ceil_partial_window();
    test_maxpool_dilation();
    test_maxpool_special_values();
    test_maxpool_no_pixel();
}
