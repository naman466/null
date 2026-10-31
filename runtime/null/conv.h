/*
 * null runtime, conv: 2-D convolution
 * needs null/core.h first; include runtime/null_runtime.h to get every part
 */
#ifndef NULL_RUNTIME_CONV_H
#define NULL_RUNTIME_CONV_H

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

#endif /* NULL_RUNTIME_CONV_H */
