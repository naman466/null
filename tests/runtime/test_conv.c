/* tests for runtime/null/conv.h: 2-D convolution; included by test_kernels.c */

static null_conv2d_params conv_params(int64_t n, int64_t c, int64_t h, int64_t w, int64_t m,
                                      int64_t kh, int64_t kw, int64_t group, int64_t stride,
                                      int64_t dil, int64_t pad_top, int64_t pad_left,
                                      int64_t pad_bottom, int64_t pad_right) {
    null_conv2d_params p;
    p.n = n; p.c = c; p.h = h; p.w = w; p.m = m; p.kh = kh; p.kw = kw; p.group = group;
    p.stride_h = p.stride_w = stride;
    p.dil_h = p.dil_w = dil;
    p.pad_top = pad_top; p.pad_left = pad_left;
    p.out_h = (h + pad_top + pad_bottom - ((kh - 1) * dil + 1)) / stride + 1;
    p.out_w = (w + pad_left + pad_right - ((kw - 1) * dil + 1)) / stride + 1;
    return p;
}

/* reference built differently: copy x into a zero-padded buffer, then sum
   in the same ci, r, s order with no bounds checks. for finite inputs the
   extra 0 * w terms leave the float sum unchanged, so this matches bit for bit */
static void ref_conv2d(const float* x, const float* w, const float* b, float* y,
                       const null_conv2d_params* p, int64_t pad_bottom, int64_t pad_right) {
    static float padded[4096];
    const int64_t ph = p->h + p->pad_top + pad_bottom, pw = p->w + p->pad_left + pad_right;
    const int64_t cg = p->c / p->group, mg = p->m / p->group;
    int64_t n, c, i, j, m, oh, ow, ci, r, s;
    for (n = 0; n < p->n; ++n) {
        memset(padded, 0, sizeof padded);
        for (c = 0; c < p->c; ++c)
            for (i = 0; i < p->h; ++i)
                for (j = 0; j < p->w; ++j)
                    padded[(c * ph + i + p->pad_top) * pw + j + p->pad_left] = x[((n * p->c + c) * p->h + i) * p->w + j];
        for (m = 0; m < p->m; ++m)
            for (oh = 0; oh < p->out_h; ++oh)
                for (ow = 0; ow < p->out_w; ++ow) {
                    float acc = 0.0f;
                    for (ci = 0; ci < cg; ++ci)
                        for (r = 0; r < p->kh; ++r)
                            for (s = 0; s < p->kw; ++s) {
                                const int64_t c_in = (m / mg) * cg + ci;
                                const float xv = padded[(c_in * ph + oh * p->stride_h + r * p->dil_h) * pw +
                                                        ow * p->stride_w + s * p->dil_w];
                                const float prod = xv * w[((m * cg + ci) * p->kh + r) * p->kw + s];
                                acc = acc + prod;
                            }
                    if (b) acc = acc + b[m];
                    y[((n * p->m + m) * p->out_h + oh) * p->out_w + ow] = acc;
                }
    }
}

/* deterministic values in [-1, 1) with plenty of mantissa bits */
static void fill(float* v, int64_t count, unsigned seed) {
    int64_t i;
    for (i = 0; i < count; ++i) {
        seed = seed * 1664525u + 1013904223u;
        v[i] = (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
}

static void test_conv_by_hand(void) {
    /* 3x3 input 1..9, 2x2 kernel of ones: each output sums a 2x2 window */
    float x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, w[4] = {1, 1, 1, 1}, y[4];
    float want[4] = {12, 16, 24, 28}, b[1] = {0.5f}, want_b[4] = {12.5f, 16.5f, 24.5f, 28.5f};
    null_conv2d_params p = conv_params(1, 1, 3, 3, 1, 2, 2, 1, 1, 1, 0, 0, 0, 0);
    null_conv2d_f32(x, w, NULL, y, &p);
    CHECK(p.out_h == 2 && p.out_w == 2);
    CHECK(all_equal(y, want, 4));
    null_conv2d_f32(x, w, b, y, &p);
    CHECK(all_equal(y, want_b, 4));
}

static void test_conv_padding_by_hand(void) {
    /* ones everywhere, 3x3 kernel, pad 1: each output counts its in-bounds neighbours */
    float x[9], w[9], y[9];
    float want[9] = {4, 6, 4, 6, 9, 6, 4, 6, 4};
    int i;
    null_conv2d_params p = conv_params(1, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1);
    for (i = 0; i < 9; ++i) x[i] = w[i] = 1.0f;
    null_conv2d_f32(x, w, NULL, y, &p);
    CHECK(all_equal(y, want, 9));
}

static void test_conv_padding_is_skipped(void) {
    /* padded positions are skipped, not multiplied by zero: an infinite
       weight that only ever lands on padding must not turn into NaN */
    float x[4] = {1, 2, 3, 4}, w[9] = {INFINITY, 0, 0, 0, 1, 0, 0, 0, 0}, y[4];
    null_conv2d_params p = conv_params(1, 1, 2, 2, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1);
    null_conv2d_f32(x, w, NULL, y, &p);
    CHECK(y[0] == 1.0f);           /* top-left: the inf tap is in the padding */
    CHECK(isinf(y[3]));            /* bottom-right: the inf tap hits x[0] */
}

static void test_conv_against_reference(void) {
    /* n, c, h, w, m, kh, kw, group, stride, dilation, pads t l b r */
    static const int64_t cases[][14] = {
        {1, 1, 5, 5, 1, 3, 3, 1, 1, 1, 0, 0, 0, 0},
        {2, 3, 7, 6, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        {1, 3, 8, 8, 2, 3, 3, 1, 2, 1, 1, 1, 1, 1},     /* stride */
        {1, 2, 9, 9, 3, 3, 3, 1, 1, 2, 0, 0, 0, 0},     /* dilation */
        {1, 4, 6, 6, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1},     /* two groups */
        {2, 4, 5, 5, 4, 3, 3, 4, 1, 1, 1, 1, 1, 1},     /* depthwise */
        {1, 2, 6, 7, 3, 2, 4, 1, 2, 1, 1, 0, 0, 2},     /* asymmetric */
        {1, 1, 28, 28, 6, 5, 5, 1, 1, 1, 2, 2, 2, 2},   /* lenet conv1 */
        {1, 3, 4, 4, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0},     /* pointwise */
    };
    static float x[4096], w[2048], b[16], y[8192], want[8192];
    size_t k;
    for (k = 0; k < sizeof cases / sizeof cases[0]; ++k) {
        const int64_t* c = cases[k];
        null_conv2d_params p = conv_params(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
                                           c[10], c[11], c[12], c[13]);
        const int64_t out = p.n * p.m * p.out_h * p.out_w;
        fill(x, p.n * p.c * p.h * p.w, (unsigned)k + 1);
        fill(w, p.m * (p.c / p.group) * p.kh * p.kw, (unsigned)k + 100);
        fill(b, p.m, (unsigned)k + 200);
        null_conv2d_f32(x, w, b, y, &p);
        ref_conv2d(x, w, b, want, &p, c[12], c[13]);
        CHECK(all_equal(y, want, out));
        null_conv2d_f32(x, w, NULL, y, &p);
        ref_conv2d(x, w, NULL, want, &p, c[12], c[13]);
        CHECK(all_equal(y, want, out));
    }
}

static void test_conv_summation_order(void) {
    /* acc runs over channels first: (1e8 + 1) - 1e8 = 0 in float32 */
    float x[3] = {1e8f, 1.0f, -1e8f}, w[3] = {1, 1, 1}, y[1];
    null_conv2d_params p = conv_params(1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
    null_conv2d_f32(x, w, NULL, y, &p);
    CHECK(y[0] == 0.0f);
}

static void run_conv_tests(void) {
    test_conv_by_hand();
    test_conv_padding_by_hand();
    test_conv_padding_is_skipped();
    test_conv_against_reference();
    test_conv_summation_order();
}
