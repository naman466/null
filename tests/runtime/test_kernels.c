/*
 * tests for runtime/null_runtime.h
 * compiled twice: as C99 (test_runtime_c) and as C++ (test_runtime_cpp),
 * so the header is proven usable from both generated C and the compiler
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "null_runtime.h"

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            printf("  [FAIL] %s:%d: %s\n", __func__, __LINE__, #cond);           \
        }                                                                        \
    } while (0)

/* bitwise float equality, so -0.0 and NaN payloads are compared exactly */
static int same_bits(float a, float b) {
    return memcmp(&a, &b, sizeof a) == 0;
}

static int all_equal(const float* got, const float* want, int64_t n) {
    int64_t i;
    for (i = 0; i < n; ++i)
        if (!same_bits(got[i], want[i])) return 0;
    return 1;
}

/* ---- shapes and strides ---- */

static void test_numel(void) {
    int64_t s[4] = {2, 3, 4, 5};
    int64_t z[2] = {3, 0};
    CHECK(null_numel(4, s) == 120);
    CHECK(null_numel(0, s) == 1);   /* scalar */
    CHECK(null_numel(2, z) == 0);   /* empty */
}

static void test_broadcast_strides(void) {
    int64_t out[2] = {2, 3};
    int64_t full[2] = {2, 3}, row[1] = {3}, col[2] = {2, 1}, one[1] = {1};
    int64_t st[2];

    null_broadcast_strides(2, out, 2, full, st);
    CHECK(st[0] == 3 && st[1] == 1);          /* contiguous */

    null_broadcast_strides(2, out, 1, row, st);
    CHECK(st[0] == 0 && st[1] == 1);          /* missing leading dim */

    null_broadcast_strides(2, out, 2, col, st);
    CHECK(st[0] == 1 && st[1] == 0);          /* size-1 trailing dim */

    null_broadcast_strides(2, out, 1, one, st);
    CHECK(st[0] == 0 && st[1] == 0);          /* broadcast scalar */

    null_broadcast_strides(2, out, 0, one, st);
    CHECK(st[0] == 0 && st[1] == 0);          /* rank 0 input */

    {
        /* [8, 1, 6, 1] read into [8, 7, 6, 5] */
        int64_t o[4] = {8, 7, 6, 5}, a[4] = {8, 1, 6, 1}, s4[4];
        null_broadcast_strides(4, o, 4, a, s4);
        CHECK(s4[0] == 6 && s4[1] == 0 && s4[2] == 1 && s4[3] == 0);
    }
}

/* ---- relu ---- */

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

/* ---- gemm ---- */

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
    null_gemm_f32(a, b, NULL, y, 2, 2, 3, 0, 0, 1.0f, 0.0f, 0, 0);
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
            null_gemm_f32(a, b, NULL, y, 3, 3, 4, ta, tb, 1.0f, 0.0f, 0, 0);
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
    null_gemm_f32(x, w, bias, y, 2, 4, 3, 0, 1, 1.0f, 1.0f, s[0], s[1]);
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
        null_gemm_f32(a, b, c, y, 2, 3, 3, 0, 0, 2.0f, 0.5f, s[0], s[1]);
        ref_gemm(a, b, c, want, 2, 3, 3, 0, 0, 2.0f, 0.5f, s[0], s[1]);
        CHECK(all_equal(y, want, 6));
    }
}

static void test_gemm_edge_sizes(void) {
    float a[4] = {1, 2, 3, 4}, b[4] = {1, 1, 1, 1}, c[2] = {5, 6};
    float y[4] = {-1, -1, -1, -1};
    /* K = 0: the products sum to 0, so y = beta * c */
    null_gemm_f32(a, b, c, y, 1, 2, 0, 0, 0, 1.0f, 1.0f, 0, 1);
    CHECK(y[0] == 5.0f && y[1] == 6.0f);
    /* M = 0 or N = 0: nothing is written */
    y[0] = -1.0f;
    null_gemm_f32(a, b, c, y, 0, 2, 2, 0, 0, 1.0f, 1.0f, 0, 1);
    null_gemm_f32(a, b, c, y, 2, 0, 2, 0, 0, 1.0f, 1.0f, 0, 1);
    CHECK(y[0] == -1.0f);
    /* 1x1x1 */
    null_gemm_f32(a, b, NULL, y, 1, 1, 1, 0, 0, 3.0f, 0.0f, 0, 0);
    CHECK(y[0] == 3.0f);
}

static void test_gemm_nan_propagates(void) {
    float a[2], b[2] = {1, 1}, y[1];
    a[0] = 1.0f;
    a[1] = (float)NAN;
    null_gemm_f32(a, b, NULL, y, 1, 1, 2, 0, 0, 1.0f, 0.0f, 0, 0);
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

int main(void) {
    test_numel();
    test_broadcast_strides();
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

#ifdef __cplusplus
    printf("=== runtime (C++) ===\n");
#else
    printf("=== runtime (C) ===\n");
#endif
    printf("Results: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
