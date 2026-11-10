/*
 * benchmark driver: times a generated model's model_run_v on one fixed input
 *
 *   bench_model [seconds [outputs.bin]]
 *
 * linked with a generated model.c. fills every input with the same
 * deterministic values, runs the model a few times to warm the caches, then
 * runs it until `seconds` (default 1) have passed and at least 10 runs are
 * done, and prints one line:
 *
 *   runs <n> median_ms <t> min_ms <t>
 *
 * the median is the number to quote: the minimum shows how much the machine's
 * noise moved it. with a second argument, the outputs are written there, raw,
 * so builds can be compared with each other and with onnx runtime.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 199309L   /* clock_gettime under -std=c99 */
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern const int model_run_num_inputs;
extern const int model_run_num_outputs;
extern const int64_t model_run_input_sizes[];
extern const int64_t model_run_output_sizes[];
void model_run_v(const float* const* inputs, float* const* outputs);

#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1000.0 * (double)t.tv_sec + (double)t.tv_nsec * 1e-6;
}
#endif

static int by_value(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

#define MAX_RUNS 100000

int main(int argc, char** argv) {
    static double times[MAX_RUNS];
    const double budget = argc > 1 ? 1000.0 * atof(argv[1]) : 1000.0;
    float* in[16];
    float* out[16];
    double start;
    int i, runs = 0;
    int64_t j;
    if (model_run_num_inputs > 16 || model_run_num_outputs > 16) {
        fprintf(stderr, "too many inputs or outputs\n");
        return 2;
    }
    for (i = 0; i < model_run_num_inputs; ++i) {
        in[i] = (float*)malloc((size_t)(model_run_input_sizes[i] ? model_run_input_sizes[i] : 1) * sizeof(float));
        if (!in[i]) return 2;
        for (j = 0; j < model_run_input_sizes[i]; ++j) in[i][j] = (float)sin(0.1 * (double)j + i);
    }
    for (i = 0; i < model_run_num_outputs; ++i) {
        out[i] = (float*)malloc((size_t)(model_run_output_sizes[i] ? model_run_output_sizes[i] : 1) * sizeof(float));
        if (!out[i]) return 2;
    }
    for (i = 0; i < 3; ++i) model_run_v((const float* const*)in, out);
    start = now_ms();
    while (runs < MAX_RUNS && (runs < 10 || now_ms() - start < budget)) {
        const double t0 = now_ms();
        model_run_v((const float* const*)in, out);
        times[runs++] = now_ms() - t0;
    }
    qsort(times, (size_t)runs, sizeof times[0], by_value);
    printf("runs %d median_ms %.4f min_ms %.4f\n", runs, times[runs / 2], times[0]);
    if (argc > 2) {
        FILE* f = fopen(argv[2], "wb");
        if (!f) return 2;
        for (i = 0; i < model_run_num_outputs; ++i)
            fwrite(out[i], sizeof(float), (size_t)model_run_output_sizes[i], f);
        fclose(f);
    }
    return 0;
}
