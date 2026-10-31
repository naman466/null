/*
 * null runtime, softmax: softmax along any axis
 * needs null/core.h first; include runtime/null_runtime.h to get every part
 */
#ifndef NULL_RUNTIME_SOFTMAX_H
#define NULL_RUNTIME_SOFTMAX_H

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

#endif /* NULL_RUNTIME_SOFTMAX_H */
