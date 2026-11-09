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

/* an average pool over one [h, w] channel: kernel, stride, padding all
   round, dilation 1 */
static null_avgpool2d_params avg_params(int64_t h, int64_t w, int64_t k, int64_t stride, int64_t pad_begin,
                                        int64_t pad_end, int64_t out, int include_pad) {
    null_avgpool2d_params p;
    p.n = 1; p.c = 1; p.h = h; p.w = w;
    p.kh = k; p.kw = k;
    p.stride_h = stride; p.stride_w = stride; p.dil_h = 1; p.dil_w = 1;
    p.pad_top = pad_begin; p.pad_left = pad_begin; p.pad_bottom = pad_end; p.pad_right = pad_end;
    p.out_h = out; p.out_w = out;
    p.count_include_pad = include_pad;
    return p;
}

static void test_avgpool_by_hand(void) {
    /* 2x2 windows, stride 2, over a 4x4 image */
    float x[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, y[4];
    float want[4] = {3.5f, 5.5f, 11.5f, 13.5f};
    null_avgpool2d_params p = avg_params(4, 4, 2, 2, 0, 0, 2, 0);
    null_avgpool2d_f32(x, y, &p);
    CHECK(all_equal(y, want, 4));
}

static void test_avgpool_padding_counts_or_not(void) {
    /* 3x3 windows, stride 1, padding 1 round a 2x2 image of ones and a 9:
       every window holds all four pixels. without count_include_pad the
       divisor is 4, with it 9 */
    float x[4] = {1, 1, 1, 9}, y[4];
    null_avgpool2d_params p = avg_params(2, 2, 3, 1, 1, 1, 2, 0);
    null_avgpool2d_f32(x, y, &p);
    CHECK(same_bits(y[0], 3.0f) && same_bits(y[3], 3.0f));
    p.count_include_pad = 1;
    null_avgpool2d_f32(x, y, &p);
    CHECK(same_bits(y[0], (float)(12.0 / 9.0)));
}

static void test_avgpool_ceil_overhang(void) {
    /* a 3x3 image, kernel 2, stride 2, ceil_mode: 2x2 windows, the last ones
       hang past the end. even with count_include_pad, what hangs past the
       padded image does not count, so they average only the pixels they hold */
    float x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, y[4];
    null_avgpool2d_params p = avg_params(3, 3, 2, 2, 0, 0, 2, 1);
    null_avgpool2d_f32(x, y, &p);
    CHECK(same_bits(y[0], 3.0f));            /* 1 2 4 5 */
    CHECK(same_bits(y[1], 4.5f));            /* 3 6 */
    CHECK(same_bits(y[3], 9.0f));            /* 9 */
}

static void test_avgpool_rounds_once(void) {
    /* 1 + 2^-24 + 2^-24 + 0: added in float the tiny ones are lost */
    float tiny = 1.0f / 16777216.0f;
    float x[4] = {1.0f, tiny, tiny, 0.0f}, y[1];
    null_avgpool2d_params p = avg_params(2, 2, 2, 1, 0, 0, 1, 0);
    null_avgpool2d_f32(x, y, &p);
    CHECK(same_bits(y[0], (float)((1.0 + 2.0 * tiny) / 4.0)));
    CHECK(!same_bits(y[0], 0.25f));
}

static void test_avgpool_nothing_to_count(void) {
    /* dilation 2 over a 1-pixel image with padding 1: the taps at -1 and 1
       both miss, so without count_include_pad 0 / 0 = NaN; with it 0 / 4 */
    float x[1] = {5}, y[1];
    null_avgpool2d_params p = avg_params(1, 1, 2, 1, 1, 1, 1, 0);
    p.dil_h = 2; p.dil_w = 2;
    null_avgpool2d_f32(x, y, &p);
    CHECK(y[0] != y[0]);
    p.count_include_pad = 1;
    null_avgpool2d_f32(x, y, &p);
    CHECK(same_bits(y[0], 0.0f));
}

static void run_pool_tests(void) {
    test_avgpool_by_hand();
    test_avgpool_padding_counts_or_not();
    test_avgpool_ceil_overhang();
    test_avgpool_rounds_once();
    test_avgpool_nothing_to_count();
    test_maxpool_by_hand();
    test_maxpool_channels_and_padding();
    test_maxpool_ceil_partial_window();
    test_maxpool_dilation();
    test_maxpool_special_values();
    test_maxpool_no_pixel();
}
