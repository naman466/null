/*
 * null runtime, layout: transpose, gather and concat, which move data without
 * arithmetic
 * needs null/core.h first; include runtime/null_runtime.h to get every part
 */
#ifndef NULL_RUNTIME_LAYOUT_H
#define NULL_RUNTIME_LAYOUT_H

/* ---------------------------------------------------------------------------
 * transpose
 *
 * y, of shape out_shape, takes its elements from x in row-major order of y:
 * moving one step along output dim d moves x_strides[d] elements in x. the
 * compiler computes x_strides from the permutation (the stride in x of the
 * input dim that output dim d came from) and drops dims of size 1, so rank
 * is small. the output is walked like an odometer, as in the broadcasting
 * kernels, with no division. rank 0 copies the one element.
 *
 * y must not alias x.
 * ------------------------------------------------------------------------ */
static inline void null_transpose_f32(const float* x, float* y, int rank,
                                      const int64_t* out_shape, const int64_t* x_strides) {
    int64_t idx[NULL_MAX_RANK];
    int64_t total = null_numel(rank, out_shape), i, at = 0;
    int d;
    for (d = 0; d < rank; ++d) idx[d] = 0;
    for (i = 0; i < total; ++i) {
        y[i] = x[at];
        for (d = rank - 1; d >= 0; --d) {
            if (++idx[d] < out_shape[d]) {
                at += x_strides[d];
                break;
            }
            at -= (out_shape[d] - 1) * x_strides[d];
            idx[d] = 0;
        }
    }
}

/* ---------------------------------------------------------------------------
 * gather
 *
 * x viewed as [outer, n, inner] along the gather axis; y is
 * [outer, count, inner], block j of each outer slice being block indices[j]
 * of x. the indices are compile-time constants, already made non-negative
 * and checked to be in [0, n) by the compiler.
 *
 * y must not alias x.
 * ------------------------------------------------------------------------ */
static inline void null_gather_f32(const float* x, float* y, int64_t outer, int64_t n, int64_t inner,
                                   const int64_t* indices, int64_t count) {
    int64_t o, j, i;
    for (o = 0; o < outer; ++o) {
        for (j = 0; j < count; ++j) {
            const float* src = x + (o * n + indices[j]) * inner;
            for (i = 0; i < inner; ++i) *y++ = src[i];
        }
    }
}

/* ---------------------------------------------------------------------------
 * concat
 *
 * one input's share of a concatenation: y viewed as [outer, n_out, inner]
 * along the concat axis and x as [outer, n_in, inner]; x's block of each outer
 * slice goes to positions offset .. offset + n_in - 1 of y's. the compiler
 * calls this once per input, with the offsets adding up, so together the
 * calls write every element of y exactly once.
 *
 * y must not alias x.
 * ------------------------------------------------------------------------ */
static inline void null_concat_f32(const float* x, float* y, int64_t outer, int64_t n_in, int64_t n_out,
                                   int64_t offset, int64_t inner) {
    int64_t o, i;
    const int64_t block = n_in * inner;
    for (o = 0; o < outer; ++o) {
        const float* src = x + o * block;
        float* dst = y + (o * n_out + offset) * inner;
        for (i = 0; i < block; ++i) dst[i] = src[i];
    }
}

#endif /* NULL_RUNTIME_LAYOUT_H */
