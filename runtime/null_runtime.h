/*
 * null runtime: the kernels that generated model.c files call
 *
 * plain C99, header only, no allocation and no libc beyond stdint/stddef,
 * so the generated program builds with any C compiler for any target.
 * also valid C++, so the compiler can reuse the same kernels at compile time
 * (constant folding).
 *
 * tensors are dense, row-major, contiguous float arrays. shapes and strides
 * are int64_t, matching the compiler's IR.
 */
#ifndef NULL_RUNTIME_H
#define NULL_RUNTIME_H

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

#endif /* NULL_RUNTIME_H */
