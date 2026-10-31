/*
 * null runtime, core: the kernels that generated model.c files call
 *
 * plain C99, header only, no allocation. it needs only stdint, stddef and
 * math.h (expf, INFINITY), so the generated program builds with any C
 * compiler for any target. on unix-like systems link with -lm.
 * also valid C++, so the compiler can reuse the same kernels at compile time
 * (constant folding).
 *
 * tensors are dense, row-major, contiguous float arrays. shapes and strides
 * are int64_t, matching the compiler's IR.
 *
 * the runtime is split into parts (runtime/null/<part>.h): this core, then one
 * part per family of kernels. generated files embed only the core and the
 * parts they call; runtime/null_runtime.h includes them all.
 */
#ifndef NULL_RUNTIME_CORE_H
#define NULL_RUNTIME_CORE_H

#include <math.h>
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

#endif /* NULL_RUNTIME_CORE_H */
