/*
 * end-to-end driver: runs a generated model on recorded inputs and compares
 * its outputs with onnx runtime's (the .vectors files in tests/fixtures/e2e)
 *
 *   run_vectors <file.vectors>
 *
 * linked with a generated model.c, through its array-form entry point.
 * an output element passes when |got - ref| <= ATOL + RTOL * |ref| (numpy's
 * allclose), NaN must match NaN, and for [1, C] outputs the index of the
 * largest value (the predicted class) must agree.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ATOL 1e-6
#define RTOL 1e-5

extern const int model_run_num_inputs;
extern const int model_run_num_outputs;
extern const int64_t model_run_input_sizes[];
extern const int64_t model_run_output_sizes[];
void model_run_v(const float* const* inputs, float* const* outputs);

static int64_t argmax(const float* v, int64_t n) {
    int64_t i, best = 0;
    for (i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
    return best;
}

int main(int argc, char** argv) {
    FILE* f;
    char magic[8];
    int32_t trials, n_in, n_out, t, k;
    int64_t sizes[16];
    float *in[8], *out[8], *ref[8];
    double worst_abs = 0, worst_rel = 0;
    long bad = 0, checked = 0, class_bad = 0;

    if (argc != 2 || !(f = fopen(argv[1], "rb"))) { fprintf(stderr, "usage: run_vectors <file.vectors>\n"); return 2; }
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "NULLVEC1", 8) != 0 ||
        fread(&trials, 4, 1, f) != 1 || fread(&n_in, 4, 1, f) != 1 || fread(&n_out, 4, 1, f) != 1 ||
        n_in > 8 || n_out > 8 || fread(sizes, 8, (size_t)(n_in + n_out), f) != (size_t)(n_in + n_out)) {
        fprintf(stderr, "bad vectors file\n");
        return 2;
    }
    /* the model must have the interface the vectors were recorded for */
    if (n_in != model_run_num_inputs || n_out != model_run_num_outputs) {
        fprintf(stderr, "model has %d inputs / %d outputs, vectors have %d / %d\n",
                model_run_num_inputs, model_run_num_outputs, n_in, n_out);
        return 1;
    }
    for (k = 0; k < n_in; ++k) {
        if (sizes[k] != model_run_input_sizes[k]) { fprintf(stderr, "input %d size mismatch\n", k); return 1; }
        in[k] = (float*)malloc((size_t)sizes[k] * 4);
    }
    for (k = 0; k < n_out; ++k) {
        if (sizes[n_in + k] != model_run_output_sizes[k]) { fprintf(stderr, "output %d size mismatch\n", k); return 1; }
        out[k] = (float*)malloc((size_t)sizes[n_in + k] * 4);
        ref[k] = (float*)malloc((size_t)sizes[n_in + k] * 4);
    }

    for (t = 0; t < trials; ++t) {
        for (k = 0; k < n_in; ++k)
            if (fread(in[k], 4, (size_t)sizes[k], f) != (size_t)sizes[k]) { fprintf(stderr, "truncated\n"); return 2; }
        for (k = 0; k < n_out; ++k)
            if (fread(ref[k], 4, (size_t)sizes[n_in + k], f) != (size_t)sizes[n_in + k]) { fprintf(stderr, "truncated\n"); return 2; }
        model_run_v((const float* const*)in, out);
        for (k = 0; k < n_out; ++k) {
            int64_t i, n = sizes[n_in + k];
            for (i = 0; i < n; ++i) {
                double g = out[k][i], r = ref[k][i], err = fabs(g - r);
                ++checked;
                if (isnan(r) || isnan(g)) {
                    if (isnan(r) != isnan(g)) ++bad;
                    continue;
                }
                if (err > worst_abs) worst_abs = err;
                if (fabs(r) > 1e-6 && err / fabs(r) > worst_rel) worst_rel = err / fabs(r);
                if (err > ATOL + RTOL * fabs(r)) {
                    if (bad < 3) printf("  trial %d output %d element %lld: got %.9g, onnxruntime %.9g\n",
                                        t, k, (long long)i, g, r);
                    ++bad;
                }
            }
            if (n > 1 && argmax(out[k], n) != argmax(ref[k], n)) {
                printf("  trial %d output %d: predicted class %lld, onnxruntime %lld\n",
                       t, k, (long long)argmax(out[k], n), (long long)argmax(ref[k], n));
                ++class_bad;
            }
        }
    }
    printf("%d trials, %ld values: %ld outside tolerance, %ld class mismatches "
           "(largest difference %.2g absolute, %.2g relative)\n",
           trials, checked, bad, class_bad, worst_abs, worst_rel);
    return bad || class_bad ? 1 : 0;
}
