/*
 * null runtime, matmul: gemm and matmul, blocked and reference
 * needs null/core.h first; include runtime/null_runtime.h to get every part
 */
#ifndef NULL_RUNTIME_MATMUL_H
#define NULL_RUNTIME_MATMUL_H

/* ---------------------------------------------------------------------------
 * matrix multiplication
 *
 * every dot product is summed in float, in order of increasing k, starting
 * from 0:  acc = ((0 + a0*b0) + a1*b1) + ...  then scaled and biased.
 * that fixed order makes results reproducible, but only if the C compiler
 * does not fuse a multiply and an add into one fma instruction, which rounds
 * once instead of twice and changes the last bit. compile with
 * -ffp-contract=off for bit-reproducible results; with fma allowed, results
 * stay within the same error bound but may differ in the last bit.
 * (onnx runtime sums in a different, blocked order, so its results differ in
 * the last bits either way, by about 1e-7 relative.)
 *
 * there are two kernels for each operation. the reference ones (_ref) are
 * the plain loops, one output at a time. the blocked ones, which generated
 * code calls unless nullc is given --reference-kernels, work on a tile of
 * outputs at once so that the compiler can keep the tile in registers and
 * vectorise across it, and on panels of b small enough to stay in cache.
 * tiling reorders which outputs are computed when, never the operations
 * inside one output: every output is still the same sum in the same order,
 * so both kernels give the same bits. the tests compare them on random
 * shapes, NaN and infinities included. one thing is left to the compiler,
 * in both: which NaN an operation on two NaNs returns. ieee 754 leaves it
 * open, x86 returns the first operand's, and a compiler may swap the
 * operands of a + b or a * b, so a NaN result can differ in sign or payload
 * between builds; it is NaN in every one.
 *
 * y must not alias a, b or c.
 * ------------------------------------------------------------------------ */

/*
 * y[M, N] = alpha * a' * b' + beta * c
 *   a' = a is [M, K], or a is [K, M] with trans_a (a' = transpose(a))
 *   b' = b is [K, N], or b is [N, K] with trans_b (b' = transpose(b))
 *   c is optional (NULL for none) and is read at c[i * c_rs + j * c_cs],
 *   so its broadcast into [M, N] is given by its strides: [N] has
 *   c_rs = 0, c_cs = 1; [M, 1] has c_rs = 1, c_cs = 0; a scalar has both 0.
 *   null_broadcast_strides(2, {M, N}, rank(c), shape(c), s) gives s[0], s[1].
 * pytorch's Linear exports as gemm with trans_b = 1 (weights stored [N, K]);
 * the optimiser transposes constant weights at compile time, because the
 * blocked kernel reads b fastest with trans_b = 0.
 * with relu set, each result goes through relu as it is written (a Relu the
 * optimiser fused into this gemm), with null_relu_f32's very expression.
 */
static inline void null_gemm_ref_f32(const float* a, const float* b, const float* c, float* y,
                                     int64_t M, int64_t N, int64_t K,
                                     int trans_a, int trans_b, float alpha, float beta,
                                     int64_t c_rs, int64_t c_cs, int relu) {
    /* distance between consecutive k for a fixed row of a' / column of b' */
    const int64_t a_k = trans_a ? M : 1, a_i = trans_a ? 1 : K;
    const int64_t b_k = trans_b ? 1 : N, b_j = trans_b ? K : 1;
    int64_t i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            const float* pa = a + i * a_i;
            const float* pb = b + j * b_j;
            float acc = 0.0f;
            for (k = 0; k < K; ++k) {
                float prod = pa[k * a_k] * pb[k * b_k];
                acc = acc + prod;
            }
            {
                float out = alpha * acc;
                if (c) {
                    float bias = beta * c[i * c_rs + j * c_cs];
                    out = out + bias;
                }
                y[i * N + j] = relu && out < 0.0f ? 0.0f : out;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * the blocked product, shared with conv.h
 *
 * y[i * ldy + j] = sum over k of a(i, k) * b(k, j), the raw sums only (the
 * callers scale, bias and relu them afterwards). a(i, k) is a[i * a_i + k * a_k].
 * b is seen through panels: NULL_TILE_K rows of k by NULL_TILE_N columns,
 * row-major, which `pack` writes into a buffer on the stack (columns past N
 * as 0, never stored). a gemm with trans_b = 0 needs no packing for whole
 * panels and points into b directly; conv packs image patches (conv.h).
 *
 * the loops: for each panel of NULL_TILE_N columns, for each chunk of
 * NULL_TILE_K values of k, for each tile of NULL_TILE_M rows, the tile's sums
 * carry on from where the previous chunk left them in y. so one output's
 * sum is built in order k = 0, 1, ... across the chunks, through a float
 * stored to y and read back, which is the same float.
 * ------------------------------------------------------------------------ */
#define NULL_TILE_M 4      /* rows of a tile */
#define NULL_TILE_N 16     /* columns of a tile and of a panel */
#define NULL_TILE_K 128    /* rows of a panel: 8 KB of floats on the stack */

/* writes panel rows [k0, k0 + kc) of columns [j0, j0 + nr) of b into out,
   out[kk * NULL_TILE_N + jj], with 0 for jj >= nr */
typedef void (*null_pack_fn)(const void* ctx, int64_t k0, int64_t kc, int64_t j0, int64_t nr, float* out);

/* NULL_TILE_M rows of a (ar[ii] points at a(row, k0)), one panel */
static inline void null_tile_m_f32(const float* const* ar, int64_t a_k, const float* p, int64_t ldp,
                                   int64_t kc, float* y, int64_t ldy, int64_t nr, int first) {
    float acc[NULL_TILE_M][NULL_TILE_N];
    int64_t kk;
    int ii, jj;
    for (ii = 0; ii < NULL_TILE_M; ++ii)
        for (jj = 0; jj < NULL_TILE_N; ++jj)
            acc[ii][jj] = !first && jj < nr ? y[ii * ldy + jj] : 0.0f;
    for (kk = 0; kk < kc; ++kk) {
        const float* pk = p + kk * ldp;
        for (ii = 0; ii < NULL_TILE_M; ++ii) {
            const float v = ar[ii][kk * a_k];
            for (jj = 0; jj < NULL_TILE_N; ++jj) {
                float prod = v * pk[jj];
                acc[ii][jj] = acc[ii][jj] + prod;
            }
        }
    }
    for (ii = 0; ii < NULL_TILE_M; ++ii)
        for (jj = 0; jj < nr; ++jj) y[ii * ldy + jj] = acc[ii][jj];
}

/* one row of a, one panel: the rows left over after the whole tiles */
static inline void null_tile_1_f32(const float* ar, int64_t a_k, const float* p, int64_t ldp,
                                   int64_t kc, float* y, int64_t nr, int first) {
    float acc[NULL_TILE_N];
    int64_t kk;
    int jj;
    for (jj = 0; jj < NULL_TILE_N; ++jj) acc[jj] = !first && jj < nr ? y[jj] : 0.0f;
    for (kk = 0; kk < kc; ++kk) {
        const float* pk = p + kk * ldp;
        const float v = ar[kk * a_k];
        for (jj = 0; jj < NULL_TILE_N; ++jj) {
            float prod = v * pk[jj];
            acc[jj] = acc[jj] + prod;
        }
    }
    for (jj = 0; jj < nr; ++jj) y[jj] = acc[jj];
}

/* b given directly as row-major [K, N] with row stride ldb */
typedef struct {
    const float* b;
    int64_t ldb;
} null_pack_rows_ctx;

static inline void null_pack_rows_f32(const void* ctx, int64_t k0, int64_t kc, int64_t j0, int64_t nr, float* out) {
    const null_pack_rows_ctx* c = (const null_pack_rows_ctx*)ctx;
    int64_t kk, jj;
    for (kk = 0; kk < kc; ++kk)
        for (jj = 0; jj < NULL_TILE_N; ++jj)
            out[kk * NULL_TILE_N + jj] = jj < nr ? c->b[(k0 + kk) * c->ldb + j0 + jj] : 0.0f;
}

/* b given as its transpose, row-major [N, K] with row stride ldb */
static inline void null_pack_cols_f32(const void* ctx, int64_t k0, int64_t kc, int64_t j0, int64_t nr, float* out) {
    const null_pack_rows_ctx* c = (const null_pack_rows_ctx*)ctx;
    int64_t kk, jj;
    for (kk = 0; kk < kc; ++kk)
        for (jj = 0; jj < NULL_TILE_N; ++jj)
            out[kk * NULL_TILE_N + jj] = jj < nr ? c->b[(j0 + jj) * c->ldb + k0 + kk] : 0.0f;
}

/* the raw sums; with direct non-NULL, whole panels are read from it in
   place (row-major [K, N], row stride ldd) and only the last, narrower
   panel goes through pack. K = 0 gives sums of 0 */
static inline void null_gemm_blocked_f32(const float* a, int64_t a_i, int64_t a_k,
                                         const float* direct, int64_t ldd,
                                         null_pack_fn pack, const void* ctx,
                                         float* y, int64_t ldy, int64_t M, int64_t N, int64_t K) {
    float panel[NULL_TILE_K * NULL_TILE_N];
    int64_t j0, k0, i0, i;
    if (K == 0) {
        for (i = 0; i < M; ++i)
            for (j0 = 0; j0 < N; ++j0) y[i * ldy + j0] = 0.0f;
        return;
    }
    for (j0 = 0; j0 < N; j0 += NULL_TILE_N) {
        const int64_t nr = N - j0 < NULL_TILE_N ? N - j0 : NULL_TILE_N;
        for (k0 = 0; k0 < K; k0 += NULL_TILE_K) {
            const int64_t kc = K - k0 < NULL_TILE_K ? K - k0 : NULL_TILE_K;
            const float* p = panel;
            int64_t ldp = NULL_TILE_N;
            if (direct && nr == NULL_TILE_N) {
                p = direct + k0 * ldd + j0;
                ldp = ldd;
            } else {
                pack(ctx, k0, kc, j0, nr, panel);
            }
            for (i0 = 0; i0 + NULL_TILE_M <= M; i0 += NULL_TILE_M) {
                const float* ar[NULL_TILE_M];
                int ii;
                for (ii = 0; ii < NULL_TILE_M; ++ii) ar[ii] = a + (i0 + ii) * a_i + k0 * a_k;
                null_tile_m_f32(ar, a_k, p, ldp, kc, y + i0 * ldy + j0, ldy, nr, k0 == 0);
            }
            for (; i0 < M; ++i0)
                null_tile_1_f32(a + i0 * a_i + k0 * a_k, a_k, p, ldp, kc, y + i0 * ldy + j0, nr, k0 == 0);
        }
    }
}

/* null_gemm_ref_f32's contract and bits, blocked */
static inline void null_gemm_f32(const float* a, const float* b, const float* c, float* y,
                                 int64_t M, int64_t N, int64_t K,
                                 int trans_a, int trans_b, float alpha, float beta,
                                 int64_t c_rs, int64_t c_cs, int relu) {
    const int64_t a_k = trans_a ? M : 1, a_i = trans_a ? 1 : K;
    null_pack_rows_ctx ctx;
    int64_t i, j;
    /* a transposed b with fewer rows than a tile: packing b would cost as
       much as the product, and the plain loop reads it in order */
    if (trans_b && M < NULL_TILE_M) {
        null_gemm_ref_f32(a, b, c, y, M, N, K, trans_a, trans_b, alpha, beta, c_rs, c_cs, relu);
        return;
    }
    ctx.b = b;
    ctx.ldb = trans_b ? K : N;
    if (trans_b)
        null_gemm_blocked_f32(a, a_i, a_k, NULL, 0, null_pack_cols_f32, &ctx, y, N, M, N, K);
    else
        null_gemm_blocked_f32(a, a_i, a_k, b, N, null_pack_rows_f32, &ctx, y, N, M, N, K);
    for (i = 0; i < M; ++i)
        for (j = 0; j < N; ++j) {
            float out = alpha * y[i * N + j];
            if (c) {
                float bias = beta * c[i * c_rs + j * c_cs];
                out = out + bias;
            }
            y[i * N + j] = relu && out < 0.0f ? 0.0f : out;
        }
}

/*
 * y = a @ b, numpy matmul with batch broadcasting
 *   each matrix product is [M, K] x [K, N] -> [M, N]
 *   the output has batch dims batch_shape (rank batch_rank, may be 0) in
 *   front of [M, N]; a and b are found at element offsets advanced by
 *   a_batch_strides / b_batch_strides per batch dim (0 along broadcast
 *   dims), i.e. null_broadcast_strides over the batch dims, times M*K for a
 *   and K*N for b.
 * 1-D operands are the caller's job: a [K] is the same memory as [1, K]
 * (M = 1) and b [K] the same as [K, 1] (N = 1); only the reported output
 * shape differs, which the compiler already knows.
 * `ref` picks the gemm: 0 for null_gemm_f32, 1 for null_gemm_ref_f32
 */
static inline void null_matmul_any_f32(const float* a, const float* b, float* y,
                                       int64_t M, int64_t N, int64_t K,
                                       int batch_rank, const int64_t* batch_shape,
                                       const int64_t* a_batch_strides,
                                       const int64_t* b_batch_strides, int ref) {
    int64_t idx[NULL_MAX_RANK] = {0};
    int64_t batches = null_numel(batch_rank, batch_shape);
    int64_t ia = 0, ib = 0, t;
    int d;
    for (t = 0; t < batches; ++t) {
        if (ref)
            null_gemm_ref_f32(a + ia, b + ib, NULL, y + t * M * N, M, N, K, 0, 0, 1.0f, 0.0f, 0, 0, 0);
        else
            null_gemm_f32(a + ia, b + ib, NULL, y + t * M * N, M, N, K, 0, 0, 1.0f, 0.0f, 0, 0, 0);
        /* odometer over the batch dims, as in the broadcasting kernels */
        for (d = batch_rank - 1; d >= 0; --d) {
            if (++idx[d] < batch_shape[d]) {
                ia += a_batch_strides[d];
                ib += b_batch_strides[d];
                break;
            }
            ia -= a_batch_strides[d] * (batch_shape[d] - 1);
            ib -= b_batch_strides[d] * (batch_shape[d] - 1);
            idx[d] = 0;
        }
    }
}

static inline void null_matmul_f32(const float* a, const float* b, float* y,
                                   int64_t M, int64_t N, int64_t K,
                                   int batch_rank, const int64_t* batch_shape,
                                   const int64_t* a_batch_strides,
                                   const int64_t* b_batch_strides) {
    null_matmul_any_f32(a, b, y, M, N, K, batch_rank, batch_shape, a_batch_strides, b_batch_strides, 0);
}

static inline void null_matmul_ref_f32(const float* a, const float* b, float* y,
                                       int64_t M, int64_t N, int64_t K,
                                       int batch_rank, const int64_t* batch_shape,
                                       const int64_t* a_batch_strides,
                                       const int64_t* b_batch_strides) {
    null_matmul_any_f32(a, b, y, M, N, K, batch_rank, batch_shape, a_batch_strides, b_batch_strides, 1);
}

#endif /* NULL_RUNTIME_MATMUL_H */
