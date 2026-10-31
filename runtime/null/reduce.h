/*
 * null runtime, reduce: the mean over any set of axes
 * needs null/core.h first; include runtime/null_runtime.h to get every part
 */
#ifndef NULL_RUNTIME_REDUCE_H
#define NULL_RUNTIME_REDUCE_H

/* ---------------------------------------------------------------------------
 * reductions
 * ------------------------------------------------------------------------ */

/* which dims of the input a reduction averages away, precomputed by the
   compiler. the compiler drops size-1 dims and merges neighbours that are
   both kept or both reduced, so rank is usually 1 or 2 */
typedef struct {
    int rank;                        /* 0 to NULL_MAX_RANK */
    int64_t shape[NULL_MAX_RANK];    /* input dims */
    int reduce[NULL_MAX_RANK];       /* 1 for a dim that is averaged away */
} null_reduce_params;

/*
 * y = the mean of x over the reduced dims. y holds the kept dims in order,
 * so keepdims (size-1 dims in the output shape) changes nothing here.
 *
 * each mean is summed in double, in row-major order, then divided and
 * rounded to float once: for any realistic count that is the exact mean,
 * correctly rounded, so it can differ from implementations that sum in
 * float in the last bit. NaN and inf propagate as in any sum (inf and -inf
 * give NaN); averaging an empty set gives 0 / 0 = NaN.
 */
static inline void null_reduce_mean_f32(const float* x, float* y, const null_reduce_params* p) {
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
        for (d = p->rank - 1; d >= 0; --d) {
            if (p->reduce[d]) continue;
            base += rest % p->shape[d] * stride[d];
            rest /= p->shape[d];
        }
        for (r = 0; r < count; ++r) {
            int64_t at = base;
            rest = r;
            for (d = p->rank - 1; d >= 0; --d) {
                if (!p->reduce[d]) continue;
                at += rest % p->shape[d] * stride[d];
                rest /= p->shape[d];
            }
            sum += x[at];
        }
        y[o] = (float)(sum / (double)count);
    }
}

#endif /* NULL_RUNTIME_REDUCE_H */
