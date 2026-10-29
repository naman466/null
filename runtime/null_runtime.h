/*
 * null runtime: the kernels that generated model.c files call
 *
 * plain C99, header only, no allocation. it needs only stdint, stddef and
 * math.h (expf, for softmax), so the generated program builds with any C
 * compiler for any target. on unix-like systems link with -lm.
 * also valid C++, so the compiler can reuse the same kernels at compile time
 * (constant folding).
 *
 * tensors are dense, row-major, contiguous float arrays. shapes and strides
 * are int64_t, matching the compiler's IR.
 */
#ifndef NULL_RUNTIME_H
#define NULL_RUNTIME_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* largest rank a broadcasting kernel handles */
#define NULL_MAX_RANK 8

/* ---------------------------------------------------------------------------
 * shapes and strides
 * ------------------------------------------------------------------------ */

/* number of elements in a shape, 1 for rank 0 */
static inline int64_t null_numel(int rank, const int64_t* shape) {
    int64_t n = 1;
    int d;
    for (d = 0; d < rank; ++d) n *= shape[d];
    return n;
}

/*
 * strides for reading an input of shape in_shape (rank in_rank) while
 * walking an output of shape out_shape (rank out_rank), numpy broadcasting:
 * dims are aligned from the right, a missing or size-1 input dim gets
 * stride 0 so the same element is reused along that output dim.
 * out_strides has out_rank entries.
 */
static inline void null_broadcast_strides(int out_rank, const int64_t* out_shape,
                                          int in_rank, const int64_t* in_shape,
                                          int64_t* out_strides) {
    int64_t stride = 1;
    int d;
    for (d = out_rank - 1; d >= 0; --d) {
        int in_d = d - (out_rank - in_rank);
        if (in_d < 0 || in_shape[in_d] == 1 || out_shape[d] == 1) {
            out_strides[d] = 0;
        } else {
            out_strides[d] = stride;
        }
        if (in_d >= 0) stride *= in_shape[in_d];
    }
}

/* ---------------------------------------------------------------------------
 * unary
 * ------------------------------------------------------------------------ */

/*
 * y = max(x, 0), elementwise. written as x < 0 ? 0 : x so that NaN passes
 * through unchanged, like onnx (numpy maximum) does. y may alias x.
 */
static inline void null_relu_f32(const float* x, float* y, int64_t n) {
    int64_t i;
    for (i = 0; i < n; ++i) y[i] = x[i] < 0.0f ? 0.0f : x[i];
}

/* ---------------------------------------------------------------------------
 * binary elementwise, with and without broadcasting
 *
 *   null_<op>_f32(a, b, y, n)
 *       a, b and y all have n elements, the same shape
 *
 *   null_<op>_bcast_f32(a, b, y, rank, shape, a_strides, b_strides)
 *       y has the given shape. a and b are read with the given strides,
 *       from null_broadcast_strides (0 along broadcast dims). the output is
 *       walked like an odometer: the last dim counts fastest, and when a dim
 *       wraps the read offsets step back by (size - 1) * stride, so no
 *       division or modulo is needed per element.
 *
 * y may alias an input only if that input has the full output shape (it is
 * then read at each position before that position is written).
 * ------------------------------------------------------------------------ */

#define NULL_DEFINE_BINARY_F32(name, expr)                                         \
    static inline void null_##name##_f32(const float* a, const float* b,            \
                                         float* y, int64_t n) {                     \
        int64_t i;                                                                  \
        for (i = 0; i < n; ++i) {                                                   \
            float x0 = a[i], x1 = b[i];                                             \
            y[i] = (expr);                                                          \
        }                                                                           \
    }                                                                               \
    static inline void null_##name##_bcast_f32(const float* a, const float* b,      \
                                               float* y, int rank,                  \
                                               const int64_t* shape,                \
                                               const int64_t* a_strides,            \
                                               const int64_t* b_strides) {          \
        int64_t idx[NULL_MAX_RANK] = {0};                                           \
        int64_t n = null_numel(rank, shape);                                        \
        int64_t ia = 0, ib = 0, i;                                                  \
        int d;                                                                      \
        for (i = 0; i < n; ++i) {                                                   \
            float x0 = a[ia], x1 = b[ib];                                           \
            y[i] = (expr);                                                          \
            for (d = rank - 1; d >= 0; --d) {                                       \
                if (++idx[d] < shape[d]) {                                          \
                    ia += a_strides[d];                                             \
                    ib += b_strides[d];                                             \
                    break;                                                          \
                }                                                                   \
                ia -= a_strides[d] * (shape[d] - 1);                                \
                ib -= b_strides[d] * (shape[d] - 1);                                \
                idx[d] = 0;                                                         \
            }                                                                       \
        }                                                                           \
    }

NULL_DEFINE_BINARY_F32(add, x0 + x1)
NULL_DEFINE_BINARY_F32(sub, x0 - x1)
NULL_DEFINE_BINARY_F32(mul, x0 * x1)
NULL_DEFINE_BINARY_F32(div, x0 / x1)

#undef NULL_DEFINE_BINARY_F32

/* ---------------------------------------------------------------------------
 * matrix multiplication
 *
 * every dot product is summed in float, in order of increasing k, starting
 * from 0:  acc = ((0 + a0*b0) + a1*b1) + ...  then scaled and biased.
 * that fixed order makes results reproducible, but only if the C compiler
 * does not fuse a multiply and an add into one fma instruction, which rounds
 * once instead of twice and changes the last bit. compile with
 * -ffp-contract=off for bit-reproducible results; with fma allowed, results
 * stay within the same error bound but may differ in the last bit.
 * (onnx runtime sums in a different, blocked order, so its results differ in
 * the last bits either way, by about 1e-7 relative.)
 *
 * y must not alias a, b or c.
 * ------------------------------------------------------------------------ */

/*
 * y[M, N] = alpha * a' * b' + beta * c
 *   a' = a is [M, K], or a is [K, M] with trans_a (a' = transpose(a))
 *   b' = b is [K, N], or b is [N, K] with trans_b (b' = transpose(b))
 *   c is optional (NULL for none) and is read at c[i * c_rs + j * c_cs],
 *   so its broadcast into [M, N] is given by its strides: [N] has
 *   c_rs = 0, c_cs = 1; [M, 1] has c_rs = 1, c_cs = 0; a scalar has both 0.
 *   null_broadcast_strides(2, {M, N}, rank(c), shape(c), s) gives s[0], s[1].
 * pytorch's Linear exports as gemm with trans_b = 1 (weights stored [N, K]),
 * which reads both a and b contiguously in the inner loop.
 */
static inline void null_gemm_f32(const float* a, const float* b, const float* c, float* y,
                                 int64_t M, int64_t N, int64_t K,
                                 int trans_a, int trans_b, float alpha, float beta,
                                 int64_t c_rs, int64_t c_cs) {
    /* distance between consecutive k for a fixed row of a' / column of b' */
    const int64_t a_k = trans_a ? M : 1, a_i = trans_a ? 1 : K;
    const int64_t b_k = trans_b ? 1 : N, b_j = trans_b ? K : 1;
    int64_t i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            const float* pa = a + i * a_i;
            const float* pb = b + j * b_j;
            float acc = 0.0f;
            for (k = 0; k < K; ++k) {
                float prod = pa[k * a_k] * pb[k * b_k];
                acc = acc + prod;
            }
            {
                float out = alpha * acc;
                if (c) {
                    float bias = beta * c[i * c_rs + j * c_cs];
                    out = out + bias;
                }
                y[i * N + j] = out;
            }
        }
    }
}

/*
 * y = a @ b, numpy matmul with batch broadcasting
 *   each matrix product is [M, K] x [K, N] -> [M, N]
 *   the output has batch dims batch_shape (rank batch_rank, may be 0) in
 *   front of [M, N]; a and b are found at element offsets advanced by
 *   a_batch_strides / b_batch_strides per batch dim (0 along broadcast
 *   dims), i.e. null_broadcast_strides over the batch dims, times M*K for a
 *   and K*N for b.
 * 1-D operands are the caller's job: a [K] is the same memory as [1, K]
 * (M = 1) and b [K] the same as [K, 1] (N = 1); only the reported output
 * shape differs, which the compiler already knows.
 */
static inline void null_matmul_f32(const float* a, const float* b, float* y,
                                   int64_t M, int64_t N, int64_t K,
                                   int batch_rank, const int64_t* batch_shape,
                                   const int64_t* a_batch_strides,
                                   const int64_t* b_batch_strides) {
    int64_t idx[NULL_MAX_RANK] = {0};
    int64_t batches = null_numel(batch_rank, batch_shape);
    int64_t ia = 0, ib = 0, t;
    int d;
    for (t = 0; t < batches; ++t) {
        null_gemm_f32(a + ia, b + ib, NULL, y + t * M * N, M, N, K, 0, 0, 1.0f, 0.0f, 0, 0);
        /* odometer over the batch dims, as in the broadcasting kernels */
        for (d = batch_rank - 1; d >= 0; --d) {
            if (++idx[d] < batch_shape[d]) {
                ia += a_batch_strides[d];
                ib += b_batch_strides[d];
                break;
            }
            ia -= a_batch_strides[d] * (batch_shape[d] - 1);
            ib -= b_batch_strides[d] * (batch_shape[d] - 1);
            idx[d] = 0;
        }
    }
}

/* ---------------------------------------------------------------------------
 * softmax
 *
 * y = exp(x - max) / sum(exp(x - max)) along one axis, as onnx (opset 13+)
 * defines it. the tensor is viewed as [outer, n, inner]: n is the size of the
 * softmax axis, outer the product of the dims before it and inner the product
 * of the dims after it, so element (o, k, i) is at (o * n + k) * inner + i.
 *
 * subtracting the max makes exp() safe: every exponent is <= 0, so nothing
 * overflows, and the result is mathematically unchanged.
 *
 * a NaN anywhere along the axis makes that whole slice NaN, and so does +inf
 * (inf - inf is NaN), matching onnx's reference implementation. the sum is
 * taken in float, in increasing k. expf is not correctly rounded, so results
 * can differ from other implementations in the last bit or two.
 *
 * y may alias x: each element is read before the same position is written.
 * ------------------------------------------------------------------------ */
static inline void null_softmax_f32(const float* x, float* y,
                                    int64_t outer, int64_t n, int64_t inner) {
    int64_t o, i, k;
    if (n == 0) return;
    for (o = 0; o < outer; ++o) {
        for (i = 0; i < inner; ++i) {
            const int64_t base = o * n * inner + i;
            float m = x[base];
            float sum = 0.0f;
            /* max along the axis. NaN needs no special case: a NaN element
               gives exp(NaN - m) = NaN, which makes the sum, and so every
               output of the slice, NaN */
            for (k = 1; k < n; ++k) {
                float v = x[base + k * inner];
                if (v > m) m = v;
            }
            for (k = 0; k < n; ++k) {
                float e = expf(x[base + k * inner] - m);
                y[base + k * inner] = e;
                sum = sum + e;
            }
            for (k = 0; k < n; ++k) y[base + k * inner] = y[base + k * inner] / sum;
        }
    }
}

/* ---------------------------------------------------------------------------
 * 2-D convolution
 *
 * x [N, C, H, W], w [M, C / group, kh, kw], b [M] or NULL, y [N, M, out_h, out_w]
 * (row-major NCHW, as onnx stores it). every output is
 *
 *   y[n, m, oh, ow] = b[m] + sum over input channel ci of the filter's group,
 *                     kernel row r and kernel column s, of
 *                     x[n, ci, oh * stride_h - pad_top + r * dil_h,
 *                             ow * stride_w - pad_left + s * dil_w] * w[m, ci', r, s]
 *
 * positions that fall in the padding are skipped (they would contribute zero).
 * the sum is taken in float, starting from 0, over ci, then r, then s in
 * increasing order, and the bias is added last; like the matrix kernels, that
 * fixed order makes results reproducible with -ffp-contract=off.
 *
 * y must not alias x, w or b.
 * ------------------------------------------------------------------------ */
typedef struct {
    int64_t n, c, h, w;            /* input */
    int64_t m, kh, kw;             /* weight [m, c / group, kh, kw] */
    int64_t group;
    int64_t stride_h, stride_w, dil_h, dil_w;
    int64_t pad_top, pad_left;     /* the end padding only shows in out_h / out_w */
    int64_t out_h, out_w;
} null_conv2d_params;

static inline void null_conv2d_f32(const float* x, const float* w, const float* b, float* y,
                                   const null_conv2d_params* p) {
    const int64_t cg = p->c / p->group;      /* input channels per group */
    const int64_t mg = p->m / p->group;      /* output channels per group */
    int64_t n, m, oh, ow, ci, r, s;
    for (n = 0; n < p->n; ++n) {
        for (m = 0; m < p->m; ++m) {
            const int64_t c0 = (m / mg) * cg;                 /* first input channel of m's group */
            const float* wm = w + m * cg * p->kh * p->kw;     /* filter m */
            for (oh = 0; oh < p->out_h; ++oh) {
                for (ow = 0; ow < p->out_w; ++ow) {
                    float acc = 0.0f;
                    for (ci = 0; ci < cg; ++ci) {
                        const float* xc = x + ((n * p->c) + c0 + ci) * p->h * p->w;
                        const float* wc = wm + ci * p->kh * p->kw;
                        for (r = 0; r < p->kh; ++r) {
                            const int64_t ih = oh * p->stride_h - p->pad_top + r * p->dil_h;
                            if (ih < 0 || ih >= p->h) continue;
                            for (s = 0; s < p->kw; ++s) {
                                const int64_t iw = ow * p->stride_w - p->pad_left + s * p->dil_w;
                                float prod;
                                if (iw < 0 || iw >= p->w) continue;
                                prod = xc[ih * p->w + iw] * wc[r * p->kw + s];
                                acc = acc + prod;
                            }
                        }
                    }
                    if (b) acc = acc + b[m];
                    y[((n * p->m + m) * p->out_h + oh) * p->out_w + ow] = acc;
                }
            }
        }
    }
}

#endif /* NULL_RUNTIME_H */
