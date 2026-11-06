/*
 * null runtime, elementwise: relu, clip, sigmoid, tanh and the binary operators, with and without broadcasting
 * needs null/core.h first; include runtime/null_runtime.h to get every part
 */
#ifndef NULL_RUNTIME_ELEMENTWISE_H
#define NULL_RUNTIME_ELEMENTWISE_H

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

/*
 * y = min(max(x, lo), hi), elementwise; ReLU6 is lo = 0, hi = 6. the compiler
 * passes the largest finite floats for a missing bound, as onnx defines it,
 * so infinities are clipped even then. as with relu, the comparisons are written so that
 * NaN passes through unchanged, and -0.0 is kept. when lo > hi every output
 * is hi, as the onnx specification says: the max with lo is taken first.
 * y may alias x.
 */
static inline void null_clip_f32(const float* x, float* y, int64_t n, float lo, float hi) {
    int64_t i;
    for (i = 0; i < n; ++i) {
        float v = x[i] < lo ? lo : x[i];
        y[i] = v > hi ? hi : v;
    }
}

/*
 * y = 1 / (1 + exp(-x)), elementwise. expf is only ever given a number <= 0,
 * so nothing overflows: for x >= 0 it is 1 / (1 + exp(-x)), and for x < 0 the
 * same value written as exp(x) / (1 + exp(x)), which keeps the small results
 * of very negative x instead of rounding 1 + huge. sigmoid(+inf) = 1,
 * sigmoid(-inf) = 0, and NaN fails x >= 0, so it reaches expf and comes out
 * NaN. expf is not correctly rounded, so results can differ from other
 * implementations in the last bit or two. y may alias x.
 */
static inline void null_sigmoid_f32(const float* x, float* y, int64_t n) {
    int64_t i;
    for (i = 0; i < n; ++i) {
        float v = x[i];
        if (v >= 0.0f) {
            y[i] = 1.0f / (1.0f + expf(-v));
        } else {
            float e = expf(v);
            y[i] = e / (1.0f + e);
        }
    }
}

/*
 * y = tanh(x), elementwise, by the C library's tanhf: tanh(+-inf) = +-1,
 * tanh(-0) = -0, NaN passes through. like expf, tanhf is not correctly
 * rounded. y may alias x.
 */
static inline void null_tanh_f32(const float* x, float* y, int64_t n) {
    int64_t i;
    for (i = 0; i < n; ++i) y[i] = tanhf(x[i]);
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

#endif /* NULL_RUNTIME_ELEMENTWISE_H */
