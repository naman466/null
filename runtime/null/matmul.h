/*
 * null runtime, matmul: gemm and matmul
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
 * pytorch's Linear exports as gemm with trans_b = 1 (weights stored [N, K]),
 * which reads both a and b contiguously in the inner loop.
 */
static inline void null_gemm_f32(const float* a, const float* b, const float* c, float* y,
                                 int64_t M, int64_t N, int64_t K,
                                 int trans_a, int trans_b, float alpha, float beta,
                                 int64_t c_rs, int64_t c_cs) {
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
                y[i * N + j] = out;
            }
        }
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
 */
static inline void null_matmul_f32(const float* a, const float* b, float* y,
                                   int64_t M, int64_t N, int64_t K,
                                   int batch_rank, const int64_t* batch_shape,
                                   const int64_t* a_batch_strides,
                                   const int64_t* b_batch_strides) {
    int64_t idx[NULL_MAX_RANK] = {0};
    int64_t batches = null_numel(batch_rank, batch_shape);
    int64_t ia = 0, ib = 0, t;
    int d;
    for (t = 0; t < batches; ++t) {
        null_gemm_f32(a + ia, b + ib, NULL, y + t * M * N, M, N, K, 0, 0, 1.0f, 0.0f, 0, 0);
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

#endif /* NULL_RUNTIME_MATMUL_H */
