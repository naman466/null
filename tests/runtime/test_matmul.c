/* tests for runtime/null/matmul.h: gemm and matmul; included by test_kernels.c */

/* reference: y = alpha * op(a) op(b) + beta * c, computed with doubles on small integers (exact) */
static void ref_gemm(const float* a, const float* b, const float* c, float* y,
                     int M, int N, int K, int ta, int tb, float alpha, float beta,
                     int64_t c_rs, int64_t c_cs) {
    int i, j, k;
    for (i = 0; i < M; ++i)
        for (j = 0; j < N; ++j) {
            double acc = 0;
            for (k = 0; k < K; ++k) {
                double av = ta ? a[k * M + i] : a[i * K + k];
                double bv = tb ? b[j * K + k] : b[k * N + j];
                acc += av * bv;
            }
            acc = alpha * acc;
            if (c) acc += beta * c[i * c_rs + j * c_cs];
            y[i * N + j] = (float)acc;
        }
}

static void test_gemm_plain(void) {
    /* [2, 3] x [3, 2] */
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {7, 8, 9, 10, 11, 12};
    float want[4] = {58, 64, 139, 154};
    float y[4];
    null_gemm_f32(a, b, NULL, y, 2, 2, 3, 0, 0, 1.0f, 0.0f, 0, 0, 0);
    CHECK(all_equal(y, want, 4));
}

static void test_gemm_transposes(void) {
    /* every combination of trans_a / trans_b against the reference */
    float a[12], b[12], y[9], want[9];
    int i, ta, tb;
    for (i = 0; i < 12; ++i) { a[i] = (float)(i - 5); b[i] = (float)(2 * i - 7); }
    for (ta = 0; ta <= 1; ++ta)
        for (tb = 0; tb <= 1; ++tb) {
            /* M = 3, N = 3, K = 4: a is [3,4] or [4,3], b is [4,3] or [3,4] */
            null_gemm_f32(a, b, NULL, y, 3, 3, 4, ta, tb, 1.0f, 0.0f, 0, 0, 0);
            ref_gemm(a, b, NULL, want, 3, 3, 4, ta, tb, 1.0f, 0.0f, 0, 0);
            CHECK(all_equal(y, want, 9));
        }
}

static void test_gemm_linear_layer(void) {
    /* pytorch Linear: x[2, 3] times w[4, 3] transposed, plus bias[4] */
    float x[6] = {1, 0, -1, 2, 1, 0};
    float w[12] = {1, 1, 1, 0, 1, 0, 2, 0, 1, -1, -1, -1};
    float bias[4] = {10, 20, 30, 40};
    float y[8], want[8];
    int64_t out[2] = {2, 4}, bshape[1] = {4}, s[2];
    null_broadcast_strides(2, out, 1, bshape, s);
    null_gemm_f32(x, w, bias, y, 2, 4, 3, 0, 1, 1.0f, 1.0f, s[0], s[1], 0);
    ref_gemm(x, w, bias, want, 2, 4, 3, 0, 1, 1.0f, 1.0f, s[0], s[1]);
    CHECK(all_equal(y, want, 8));
    CHECK(y[0] == 10.0f && y[2] == 31.0f && y[7] == 37.0f);
}

static void test_gemm_bias_shapes(void) {
    /* c as a scalar, [N], [1, N], [M, 1] and [M, N], with alpha and beta */
    float a[6] = {1, 2, 3, 4, 5, 6};          /* [2, 3] */
    float b[9] = {1, 0, 2, 0, 1, 0, 3, 0, 1}; /* [3, 3] */
    float c[6] = {1, 2, 3, 4, 5, 6};
    float y[6], want[6];
    int64_t out[2] = {2, 3};
    int64_t shapes[5][2] = {{0, 0}, {3, 0}, {1, 3}, {2, 1}, {2, 3}};
    int ranks[5] = {0, 1, 2, 2, 2};
    int t;
    for (t = 0; t < 5; ++t) {
        int64_t s[2];
        null_broadcast_strides(2, out, ranks[t], shapes[t], s);
        null_gemm_f32(a, b, c, y, 2, 3, 3, 0, 0, 2.0f, 0.5f, s[0], s[1], 0);
        ref_gemm(a, b, c, want, 2, 3, 3, 0, 0, 2.0f, 0.5f, s[0], s[1]);
        CHECK(all_equal(y, want, 6));
    }
}

static void test_gemm_edge_sizes(void) {
    float a[4] = {1, 2, 3, 4}, b[4] = {1, 1, 1, 1}, c[2] = {5, 6};
    float y[4] = {-1, -1, -1, -1};
    /* K = 0: the products sum to 0, so y = beta * c */
    null_gemm_f32(a, b, c, y, 1, 2, 0, 0, 0, 1.0f, 1.0f, 0, 1, 0);
    CHECK(y[0] == 5.0f && y[1] == 6.0f);
    /* M = 0 or N = 0: nothing is written */
    y[0] = -1.0f;
    null_gemm_f32(a, b, c, y, 0, 2, 2, 0, 0, 1.0f, 1.0f, 0, 1, 0);
    null_gemm_f32(a, b, c, y, 2, 0, 2, 0, 0, 1.0f, 1.0f, 0, 1, 0);
    CHECK(y[0] == -1.0f);
    /* 1x1x1 */
    null_gemm_f32(a, b, NULL, y, 1, 1, 1, 0, 0, 3.0f, 0.0f, 0, 0, 0);
    CHECK(y[0] == 3.0f);
}

static void test_gemm_nan_propagates(void) {
    float a[2], b[2] = {1, 1}, y[1];
    a[0] = 1.0f;
    a[1] = (float)NAN;
    null_gemm_f32(a, b, NULL, y, 1, 1, 2, 0, 0, 1.0f, 0.0f, 0, 0, 0);
    CHECK(isnan(y[0]));
}

/* ---- matmul ---- */

static void test_matmul_2d_and_vectors(void) {
    float a[6] = {1, 2, 3, 4, 5, 6};   /* [2, 3] */
    float b[6] = {1, 0, 0, 1, 1, 1};   /* [3, 2] */
    float want[4] = {4, 5, 10, 11};
    float v[3] = {1, 1, 1};
    float y[4];
    int64_t none[1] = {0};
    null_matmul_f32(a, b, y, 2, 2, 3, 0, none, none, none);
    CHECK(all_equal(y, want, 4));

    /* [3] x [3, 2] -> [2]: a vector is [1, K] in memory */
    null_matmul_f32(v, b, y, 1, 2, 3, 0, none, none, none);
    CHECK(y[0] == 2.0f && y[1] == 2.0f);

    /* [2, 3] x [3] -> [2]: b vector is [K, 1] */
    null_matmul_f32(a, v, y, 2, 1, 3, 0, none, none, none);
    CHECK(y[0] == 6.0f && y[1] == 15.0f);

    /* [3] x [3] -> scalar dot product */
    null_matmul_f32(v, v, y, 1, 1, 3, 0, none, none, none);
    CHECK(y[0] == 3.0f);
}

static void test_matmul_batched(void) {
    /* [2, 2, 2] x [2, 2] (shared weights) and x [2, 2, 2] (per batch) */
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float w[4] = {1, 0, 0, 2};                 /* scales column 1 by 2 */
    float b2[8] = {1, 0, 0, 1, 0, 1, 1, 0};    /* identity, then swap */
    float want_shared[8] = {1, 4, 3, 8, 5, 12, 7, 16};
    float want_batch[8] = {1, 2, 3, 4, 6, 5, 8, 7};
    float y[8];
    int64_t bshape[1] = {2}, a_bs[1] = {4}, shared_bs[1] = {0}, b_bs[1] = {4};
    null_matmul_f32(a, w, y, 2, 2, 2, 1, bshape, a_bs, shared_bs);
    CHECK(all_equal(y, want_shared, 8));
    null_matmul_f32(a, b2, y, 2, 2, 2, 1, bshape, a_bs, b_bs);
    CHECK(all_equal(y, want_batch, 8));
}

static void test_matmul_batch_broadcast(void) {
    /* a [2, 1, 1, 2] x b [3, 2, 1] -> [2, 3, 1, 1]: batch dims [2, 1] and [3] broadcast to [2, 3] */
    float a[4] = {1, 2, 3, 4};        /* two 1x2 rows */
    float b[6] = {1, 1, 0, 1, 2, 0};  /* three 2x1 columns */
    float y[6];
    float want[6] = {3, 2, 2, 7, 4, 6};
    int64_t out_batch[2] = {2, 3}, a_batch[2] = {2, 1}, b_batch[1] = {3};
    int64_t sa[2], sb[2];
    int d;
    null_broadcast_strides(2, out_batch, 2, a_batch, sa);
    null_broadcast_strides(2, out_batch, 1, b_batch, sb);
    for (d = 0; d < 2; ++d) { sa[d] *= 1 * 2; sb[d] *= 2 * 1; }   /* times M*K and K*N */
    null_matmul_f32(a, b, y, 1, 1, 2, 2, out_batch, sa, sb);
    CHECK(all_equal(y, want, 6));
}

static void test_matmul_summation_order(void) {
    /* the dot product is summed left to right in float: (1e8 + 1) - 1e8 = 0 in float32,
       which pins the documented order (any reordering would give 1) */
    float a[3] = {1e8f, 1.0f, -1e8f}, b[3] = {1, 1, 1}, y[1];
    int64_t none[1] = {0};
    null_matmul_f32(a, b, y, 1, 1, 3, 0, none, none, none);
    CHECK(y[0] == 0.0f);
}

static void test_gemm_fused_relu(void) {
    /* relu applied while writing gives exactly gemm then relu: negative
       results become 0, -0.0 and NaN pass through as they are */
    float a[6] = {1, -2, 3, -4, 5, -6}, b[6] = {1, 0, -1, 2, 0, 1}, c[3] = {0.5f, -100, 0};
    float fused[6], plain[6];
    float nan_a[2] = {NAN, 1}, nan_b[2] = {1, 1}, nz_a[1] = {-0.0f}, nz_b[1] = {1};
    float f1[1], p1[1];
    null_gemm_f32(a, b, c, fused, 2, 3, 2, 0, 0, 1.0f, 1.0f, 0, 1, 1);
    null_gemm_f32(a, b, c, plain, 2, 3, 2, 0, 0, 1.0f, 1.0f, 0, 1, 0);
    null_relu_f32(plain, plain, 6);
    CHECK(all_equal(fused, plain, 6));
    null_gemm_f32(nan_a, nan_b, NULL, f1, 1, 1, 2, 0, 0, 1.0f, 0.0f, 0, 0, 1);
    CHECK(f1[0] != f1[0]);
    /* a gemm's sum starts at +0.0, so -0.0 * 1 still sums to +0.0: only
       alpha can make the output -0.0 (-1 times +0.0), and relu must keep it */
    null_gemm_f32(nz_a, nz_b, NULL, f1, 1, 1, 1, 0, 0, -1.0f, 0.0f, 0, 0, 1);
    null_gemm_f32(nz_a, nz_b, NULL, p1, 1, 1, 1, 0, 0, -1.0f, 0.0f, 0, 0, 0);
    CHECK(same_bits(p1[0], -0.0f));
    null_relu_f32(p1, p1, 1);
    CHECK(same_bits(f1[0], p1[0]));
    CHECK(same_bits(f1[0], -0.0f));
}

/* ---- blocked against reference ---- */

/* the blocked gemm and the reference loops on random shapes, every flag,
   with sizes on both sides of the tile edges (4 rows, 16 columns, 128 k)
   and NaN, infinities and -0.0 among the values: the same bits, always
   (but for which NaN a NaN is, see all_same_values) */
static void test_gemm_blocked_matches_reference(void) {
    static float a[40 * 300], b[300 * 40], c[40 * 40], y[40 * 40], want[40 * 40];
    int t, mismatches = 0;
    for (t = 0; t < 300; ++t) {
        const int64_t M = rand_int(0, 40), N = rand_int(0, 40), K = t % 10 == 0 ? rand_int(129, 300) : rand_int(0, 40);
        const int ta = (int)rand_int(0, 1), tb = (int)rand_int(0, 1), relu = (int)rand_int(0, 1);
        const int special = t % 3 == 0;
        const int cs = (int)rand_int(0, 3);   /* none, [N], [M, 1], [M, N] */
        const int64_t c_rs = cs == 2 ? 1 : cs == 3 ? N : 0, c_cs = cs == 1 || cs == 3 ? 1 : 0;
        const float alpha = t % 4 == 0 ? 0.5f : 1.0f, beta = t % 5 == 0 ? -2.0f : 1.0f;
        int64_t i;
        for (i = 0; i < M * K; ++i) a[i] = rand_value(special);
        for (i = 0; i < K * N; ++i) b[i] = rand_value(special);
        for (i = 0; i < M * N; ++i) c[i] = rand_value(special);
        null_gemm_f32(a, b, cs ? c : NULL, y, M, N, K, ta, tb, alpha, beta, c_rs, c_cs, relu);
        null_gemm_ref_f32(a, b, cs ? c : NULL, want, M, N, K, ta, tb, alpha, beta, c_rs, c_cs, relu);
        if (!all_same_values(y, want, M * N)) ++mismatches;
    }
    CHECK(mismatches == 0);
}

static void test_matmul_blocked_matches_reference(void) {
    static float a[3 * 20 * 30], b[3 * 30 * 20], y[3 * 20 * 20], want[3 * 20 * 20];
    int t, mismatches = 0;
    for (t = 0; t < 50; ++t) {
        const int64_t M = rand_int(1, 20), N = rand_int(1, 20), K = rand_int(1, 30);
        int64_t bshape[1] = {3}, a_bs[1], b_bs[1], i;
        a_bs[0] = M * K;
        b_bs[0] = t % 2 ? 0 : K * N;      /* per-batch b, or one b shared */
        for (i = 0; i < 3 * M * K; ++i) a[i] = rand_value(1);
        for (i = 0; i < 3 * K * N; ++i) b[i] = rand_value(1);
        null_matmul_f32(a, b, y, M, N, K, 1, bshape, a_bs, b_bs);
        null_matmul_ref_f32(a, b, want, M, N, K, 1, bshape, a_bs, b_bs);
        if (!all_same_values(y, want, 3 * M * N)) ++mismatches;
    }
    CHECK(mismatches == 0);
}

static void run_matmul_tests(void) {
    test_gemm_fused_relu();
    test_gemm_plain();
    test_gemm_transposes();
    test_gemm_linear_layer();
    test_gemm_bias_shapes();
    test_gemm_edge_sizes();
    test_gemm_nan_propagates();
    test_matmul_2d_and_vectors();
    test_matmul_batched();
    test_matmul_batch_broadcast();
    test_matmul_summation_order();
    test_gemm_blocked_matches_reference();
    test_matmul_blocked_matches_reference();
}
