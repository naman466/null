/*
 * null runtime, norm: layer and batch normalization
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

/* ---------------------------------------------------------------------------
 * batch normalization, inference
 *
 * the tensor is viewed as [n, c, inner]: channel k of every image is
 * y = (x - mean[k]) / sqrt(var[k] + epsilon) * scale[k] + bias[k], with the
 * mean and variance given (recorded during training), not computed.
 *
 * the formula is the specification's, taken in double and rounded to float
 * once per output, like layer normalization's. sqrt(var + epsilon) is computed
 * once per channel. a variance below -epsilon gives NaN for that channel, as
 * the formula does; NaN and inf in x propagate to their own output only.
 *
 * y may alias x: each element is read before the same position is written.
 * ------------------------------------------------------------------------ */
static inline void null_batchnorm_f32(const float* x, const float* scale, const float* bias,
                                      const float* mean, const float* var, float* y,
                                      int64_t n, int64_t c, int64_t inner, float epsilon) {
    int64_t b, k, i;
    for (k = 0; k < c; ++k) {
        const double sd = sqrt((double)var[k] + (double)epsilon);
        const double m = mean[k], s = scale[k], t = bias[k];
        for (b = 0; b < n; ++b) {
            const int64_t base = (b * c + k) * inner;
            for (i = 0; i < inner; ++i) y[base + i] = (float)((x[base + i] - m) / sd * s + t);
        }
    }
}

#endif /* NULL_RUNTIME_NORM_H */
