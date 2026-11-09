/*
 * null runtime, reduce: the mean, sum or max over any set of axes
 * needs null/core.h first; include runtime/null_runtime.h to get every part
 */
#ifndef NULL_RUNTIME_REDUCE_H
#define NULL_RUNTIME_REDUCE_H

/* ---------------------------------------------------------------------------
 * reductions
 * ------------------------------------------------------------------------ */

/* which dims of the input a reduction takes away, precomputed by the
   compiler. the compiler drops size-1 dims and merges neighbours that are
   both kept or both reduced, so rank is usually 1 or 2 */
typedef struct {
    int rank;                        /* 0 to NULL_MAX_RANK */
    int64_t shape[NULL_MAX_RANK];    /* input dims */
    int reduce[NULL_MAX_RANK];       /* 1 for a dim that is reduced away */
} null_reduce_params;

enum { NULL_REDUCE_MEAN, NULL_REDUCE_SUM, NULL_REDUCE_MAX };

/*
 * y = the mean, sum or max of x over the reduced dims. y holds the kept dims
 * in order, so keepdims (size-1 dims in the output shape) changes nothing
 * here. each output's set is visited in row-major order.
 *
 * a mean or sum is added up in double and rounded to float once: for any
 * realistic count that is the exact result, correctly rounded, so it can
 * differ in the last bit from implementations that add in float. NaN and inf
 * propagate as in any sum (inf and -inf give NaN). the sum of an empty set is
 * 0 and its mean 0 / 0 = NaN.
 *
 * a max is exact. it is NaN if any element is NaN (as numpy's max, which the
 * specification follows), and the max of an empty set is -inf, as the
 * specification says. -0.0 and +0.0 compare equal, so whichever comes first
 * is kept.
 */
static inline void null_reduce_f32(const float* x, float* y, const null_reduce_params* p, int op) {
    int64_t stride[NULL_MAX_RANK];
    int64_t kept = 1, count = 1, step = 1, o, r;
    int d;
    for (d = p->rank - 1; d >= 0; --d) {
        stride[d] = step;
        step *= p->shape[d];
        if (p->reduce[d]) count *= p->shape[d];
        else kept *= p->shape[d];
    }
    for (o = 0; o < kept; ++o) {
        /* the first element of y[o]'s set: o's digits along the kept dims */
        int64_t base = 0, rest = o;
        double sum = 0.0;
        float max = -INFINITY;
        for (d = p->rank - 1; d >= 0; --d) {
            if (p->reduce[d]) continue;
            base += rest % p->shape[d] * stride[d];
            rest /= p->shape[d];
        }
        for (r = 0; r < count; ++r) {
            int64_t at = base;
            float v;
            rest = r;
            for (d = p->rank - 1; d >= 0; --d) {
                if (!p->reduce[d]) continue;
                at += rest % p->shape[d] * stride[d];
                rest /= p->shape[d];
            }
            v = x[at];
            if (op != NULL_REDUCE_MAX) sum += v;
            else if (v > max || v != v) {
                max = v;
                if (v != v) break;               /* NaN: nothing can change it */
            }
        }
        if (op == NULL_REDUCE_MAX) y[o] = max;
        else if (op == NULL_REDUCE_SUM) y[o] = (float)sum;
        else y[o] = (float)(sum / (double)count);
    }
}

static inline void null_reduce_mean_f32(const float* x, float* y, const null_reduce_params* p) {
    null_reduce_f32(x, y, p, NULL_REDUCE_MEAN);
}

static inline void null_reduce_sum_f32(const float* x, float* y, const null_reduce_params* p) {
    null_reduce_f32(x, y, p, NULL_REDUCE_SUM);
}

static inline void null_reduce_max_f32(const float* x, float* y, const null_reduce_params* p) {
    null_reduce_f32(x, y, p, NULL_REDUCE_MAX);
}

#endif /* NULL_RUNTIME_REDUCE_H */
