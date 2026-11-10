/*
 * null runtime, conv: 2-D convolution, blocked and reference
 * needs null/core.h and null/matmul.h first; include runtime/null_runtime.h
 * to get every part
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
 * null_conv2d_ref_f32 is that formula as loops. null_conv2d_f32, which
 * generated code calls by default, gives the same bits faster: for each image
 * and group it is a matrix product, filters [mg, cg*kh*kw] times patches
 * [cg*kh*kw, out_h*out_w], on matmul.h's blocked product, with the patches
 * packed a panel at a time (a 1x1 convolution with stride 1 and no padding
 * reads x in place; a group of fewer than NULL_TILE_M filters, like a
 * depthwise convolution's, slides each filter over the image instead, which
 * skips the padding like the reference). k runs over (ci, r, s) in the formula's order. a packed
 * position in the padding holds 0 rather than being skipped, which adds a
 * product of zero: that leaves the sum's bits alone, because a sum that
 * starts from +0 can never be -0, and adding a zero to anything but -0
 * changes nothing. it only fails when a weight is infinite or NaN (0 * inf
 * is NaN), so a convolution with padding and such a weight takes the
 * reference loops.
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
    int relu;                      /* 1: a Relu fused into this conv, applied as each output is written */
} null_conv2d_params;

static inline void null_conv2d_ref_f32(const float* x, const float* w, const float* b, float* y,
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
                    /* the fused relu is null_relu_f32's very expression */
                    y[((n * p->m + m) * p->out_h + oh) * p->out_w + ow] = p->relu && acc < 0.0f ? 0.0f : acc;
                }
            }
        }
    }
}

/* the patches of one image's group of channels, for null_gemm_blocked_f32 */
typedef struct {
    const float* x;                /* the group's first input channel */
    const null_conv2d_params* p;
} null_pack_patches_ctx;

/* panel rows k = (ci, r, s) in [k0, k0 + kc), columns = output positions
   oh * out_w + ow in [j0, j0 + nr): the input each filter tap meets there */
static inline void null_pack_patches_f32(const void* ctx, int64_t k0, int64_t kc, int64_t j0, int64_t nr,
                                         float* out) {
    const null_pack_patches_ctx* c = (const null_pack_patches_ctx*)ctx;
    const null_conv2d_params* p = c->p;
    /* where each position's window starts (row, column, offset), and the
       extremes over the panel; no divisions inside the loops, they would
       cost more than the copying */
    int64_t ih0[NULL_TILE_N], iw0[NULL_TILE_N], at0[NULL_TILE_N];
    int64_t ih_lo = INT64_MAX, ih_hi = INT64_MIN, iw_lo = INT64_MAX, iw_hi = INT64_MIN;
    int64_t oh = j0 / p->out_w, ow = j0 % p->out_w;
    int64_t ci = k0 / (p->kh * p->kw), r = k0 / p->kw % p->kh, s = k0 % p->kw;
    int64_t jj, kk;
    for (jj = 0; jj < nr; ++jj) {
        ih0[jj] = oh * p->stride_h - p->pad_top;
        iw0[jj] = ow * p->stride_w - p->pad_left;
        at0[jj] = ih0[jj] * p->w + iw0[jj];
        if (++ow == p->out_w) {
            ow = 0;
            ++oh;
        }
    }
    for (jj = 0; jj < nr; ++jj) {
        if (ih0[jj] < ih_lo) ih_lo = ih0[jj];
        if (ih0[jj] > ih_hi) ih_hi = ih0[jj];
        if (iw0[jj] < iw_lo) iw_lo = iw0[jj];
        if (iw0[jj] > iw_hi) iw_hi = iw0[jj];
    }
    for (kk = 0; kk < kc; ++kk) {
        const float* xc = c->x + ci * p->h * p->w;
        const int64_t dr = r * p->dil_h, ds = s * p->dil_w;
        float* o = out + kk * NULL_TILE_N;
        if (ih_lo + dr >= 0 && ih_hi + dr < p->h && iw_lo + ds >= 0 && iw_hi + ds < p->w) {
            /* this tap is inside the image for every position */
            const float* xt = xc + dr * p->w + ds;
            for (jj = 0; jj < nr; ++jj) o[jj] = xt[at0[jj]];
        } else {
            for (jj = 0; jj < nr; ++jj) {
                const int64_t ih = ih0[jj] + dr, iw = iw0[jj] + ds;
                o[jj] = ih >= 0 && ih < p->h && iw >= 0 && iw < p->w ? xc[ih * p->w + iw] : 0.0f;
            }
        }
        for (; jj < NULL_TILE_N; ++jj) o[jj] = 0.0f;
        if (++s == p->kw) {
            s = 0;
            if (++r == p->kh) {
                r = 0;
                ++ci;
            }
        }
    }
}

/* a group with too few filters to fill a tile (depthwise convolution has
   one): packing patches would cost as much as the product, so each filter
   slides over the image directly, NULL_CONV_ROW outputs of a row at a time.
   positions in the padding are skipped, as in the reference loops */
#define NULL_CONV_ROW 8

static inline void null_conv2d_direct_f32(const float* xg, const float* wm, float* ym,
                                          const null_conv2d_params* p, int64_t cg) {
    int64_t oh, ow0, ci, r, s;
    int jj;
    for (oh = 0; oh < p->out_h; ++oh)
        for (ow0 = 0; ow0 < p->out_w; ow0 += NULL_CONV_ROW) {
            const int owb = (int)(p->out_w - ow0 < NULL_CONV_ROW ? p->out_w - ow0 : NULL_CONV_ROW);
            float acc[NULL_CONV_ROW];
            for (jj = 0; jj < NULL_CONV_ROW; ++jj) acc[jj] = 0.0f;
            for (ci = 0; ci < cg; ++ci) {
                const float* xc = xg + ci * p->h * p->w;
                for (r = 0; r < p->kh; ++r) {
                    const int64_t ih = oh * p->stride_h - p->pad_top + r * p->dil_h;
                    const float* xr = xc + ih * p->w;
                    const float* wr = wm + (ci * p->kh + r) * p->kw;
                    if (ih < 0 || ih >= p->h) continue;
                    for (s = 0; s < p->kw; ++s) {
                        const int64_t iw0 = ow0 * p->stride_w - p->pad_left + s * p->dil_w;
                        const float wv = wr[s];
                        if (owb == NULL_CONV_ROW && iw0 >= 0 && iw0 + (NULL_CONV_ROW - 1) * p->stride_w < p->w) {
                            /* the whole row inside the image */
                            if (p->stride_w == 1) {
                                const float* xs = xr + iw0;
                                for (jj = 0; jj < NULL_CONV_ROW; ++jj) {
                                    float prod = xs[jj] * wv;
                                    acc[jj] = acc[jj] + prod;
                                }
                            } else {
                                for (jj = 0; jj < NULL_CONV_ROW; ++jj) {
                                    float prod = xr[iw0 + jj * p->stride_w] * wv;
                                    acc[jj] = acc[jj] + prod;
                                }
                            }
                        } else {
                            for (jj = 0; jj < owb; ++jj) {
                                const int64_t iw = iw0 + jj * p->stride_w;
                                float prod;
                                if (iw < 0 || iw >= p->w) continue;
                                prod = xr[iw] * wv;
                                acc[jj] = acc[jj] + prod;
                            }
                        }
                    }
                }
            }
            for (jj = 0; jj < owb; ++jj) ym[oh * p->out_w + ow0 + jj] = acc[jj];
        }
}

/* null_conv2d_ref_f32's contract and bits, as matrix products */
static inline void null_conv2d_f32(const float* x, const float* w, const float* b, float* y,
                                   const null_conv2d_params* p) {
    const int64_t cg = p->c / p->group, mg = p->m / p->group;
    const int64_t fs = cg * p->kh * p->kw;                     /* one filter's size, the product's K */
    const int64_t plane = p->out_h * p->out_w;                 /* the product's N */
    /* does any window reach into the padding? */
    const int padded = p->pad_top > 0 || p->pad_left > 0 ||
                       (p->out_h - 1) * p->stride_h - p->pad_top + (p->kh - 1) * p->dil_h >= p->h ||
                       (p->out_w - 1) * p->stride_w - p->pad_left + (p->kw - 1) * p->dil_w >= p->w;
    const int in_place = p->kh == 1 && p->kw == 1 && p->stride_h == 1 && p->stride_w == 1 && !padded;
    int64_t n, g, i, j;
    if (padded && mg >= NULL_TILE_M) {
        for (i = 0; i < p->m * fs; ++i)
            if (!(w[i] - w[i] == 0.0f)) {      /* infinite or NaN: v - v is NaN */
                null_conv2d_ref_f32(x, w, b, y, p);
                return;
            }
    }
    for (n = 0; n < p->n; ++n)
        for (g = 0; g < p->group; ++g) {
            null_pack_patches_ctx ctx;
            float* yg = y + (n * p->m + g * mg) * plane;
            ctx.x = x + (n * p->c + g * cg) * p->h * p->w;
            ctx.p = p;
            if (mg < NULL_TILE_M) {
                for (i = 0; i < mg; ++i)
                    null_conv2d_direct_f32(ctx.x, w + (g * mg + i) * fs, yg + i * plane, p, cg);
            } else {
                null_gemm_blocked_f32(w + g * mg * fs, fs, 1, in_place ? ctx.x : NULL, plane,
                                      null_pack_patches_f32, &ctx, yg, plane, mg, plane, fs);
            }
            for (i = 0; i < mg; ++i)
                for (j = 0; j < plane; ++j) {
                    float acc = yg[i * plane + j];
                    if (b) acc = acc + b[g * mg + i];
                    yg[i * plane + j] = p->relu && acc < 0.0f ? 0.0f : acc;
                }
        }
}

#endif /* NULL_RUNTIME_CONV_H */
