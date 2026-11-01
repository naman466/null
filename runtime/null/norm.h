/*
 * null runtime, norm: layer normalization
 * needs null/core.h first; include runtime/null_runtime.h to get every part
 */
#ifndef NULL_RUNTIME_NORM_H
#define NULL_RUNTIME_NORM_H

/* ---------------------------------------------------------------------------
 * layer normalization
 *
 * the tensor is viewed as [outer, n]: each of the outer rows of n values is
 * normalised on its own, y = (x - mean) / sqrt(var + epsilon) * scale + bias,
 * with var the mean of (x - mean)^2. scale and bias have the n values of the
 * normalized dims; bias may be NULL.
 *
 * like the mean in reduce.h, everything is computed in double and rounded to
 * float once per output: the mean, the variance (in a second pass over the
 * row, which is exact enough that no cancellation can make it negative), the
 * division and the scale and shift. NaN and inf propagate: one in a row makes
 * that whole row NaN, since the mean is then NaN or inf - inf.
 *
 * y may alias x: each element is read before the same position is written.
 * ------------------------------------------------------------------------ */
static inline void null_layernorm_f32(const float* x, const float* scale, const float* bias, float* y,
                                      int64_t outer, int64_t n, float epsilon) {
    int64_t o, i;
    if (n == 0) return;
    for (o = 0; o < outer; ++o) {
        const float* xr = x + o * n;
        float* yr = y + o * n;
        double sum = 0.0, sq = 0.0, mean, sd;
        for (i = 0; i < n; ++i) sum += xr[i];
        mean = sum / (double)n;
        for (i = 0; i < n; ++i) {
            const double d = xr[i] - mean;
            sq += d * d;
        }
        sd = sqrt(sq / (double)n + (double)epsilon);
        for (i = 0; i < n; ++i) {
            double v = (xr[i] - mean) / sd * scale[i];
            if (bias) v += bias[i];
            yr[i] = (float)v;
        }
    }
}

#endif /* NULL_RUNTIME_NORM_H */
