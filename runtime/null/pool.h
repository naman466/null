/*
 * null runtime, pool: 2-D max pooling
 * needs null/core.h first; include runtime/null_runtime.h to get every part
 */
#ifndef NULL_RUNTIME_POOL_H
#define NULL_RUNTIME_POOL_H

/* ---------------------------------------------------------------------------
 * pooling
 * ------------------------------------------------------------------------ */

/* geometry of a 2-D max pool, precomputed by the compiler */
typedef struct {
    int64_t n, c, h, w;              /* input [n, c, h, w], output [n, c, out_h, out_w] */
    int64_t kh, kw;
    int64_t stride_h, stride_w, dil_h, dil_w;
    int64_t pad_top, pad_left;       /* padding before the image; the padding after is implied */
    int64_t out_h, out_w;
} null_maxpool2d_params;

/*
 * y = MaxPool(x), each channel separately. positions in the padding are
 * skipped. NaN is ignored while the window holds any number, as in onnx's
 * reference and onnx runtime; a window of nothing but NaN gives its first NaN.
 * a window with no pixel at all (only possible with dilation) gives -inf
 */
static inline void null_maxpool2d_f32(const float* x, float* y, const null_maxpool2d_params* p) {
    int64_t nc, oh, ow, r, s;
    for (nc = 0; nc < p->n * p->c; ++nc) {
        const float* xc = x + nc * p->h * p->w;
        for (oh = 0; oh < p->out_h; ++oh) {
            for (ow = 0; ow < p->out_w; ++ow) {
                float acc = -INFINITY;
                int seen = 0;       /* 0: no pixel yet, 1: only NaN so far, 2: a number */
                for (r = 0; r < p->kh; ++r) {
                    const int64_t ih = oh * p->stride_h - p->pad_top + r * p->dil_h;
                    if (ih < 0 || ih >= p->h) continue;
                    for (s = 0; s < p->kw; ++s) {
                        const int64_t iw = ow * p->stride_w - p->pad_left + s * p->dil_w;
                        float v;
                        if (iw < 0 || iw >= p->w) continue;
                        v = xc[ih * p->w + iw];
                        if (v != v) {
                            if (seen == 0) { acc = v; seen = 1; }   /* kept only if no number follows */
                        } else {
                            if (seen < 2 || v > acc) acc = v;
                            seen = 2;
                        }
                    }
                }
                y[(nc * p->out_h + oh) * p->out_w + ow] = acc;
            }
        }
    }
}

#endif /* NULL_RUNTIME_POOL_H */
