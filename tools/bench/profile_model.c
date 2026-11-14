/*
 * profiling driver: times every step of a generated model
 *
 *   profile_model [runs]
 *
 * linked with a model.c that tools/bench/profile_layers.py has instrumented:
 * a call null_prof_mark(i) before each step of model_run, one after the last,
 * and null_prof_n, the number of steps. fills every input like bench_model.c does,
 * runs the model once to warm the caches, then `runs` times (default 10)
 * with the clock on, and prints one line per step:
 *
 *   step <i> ms <mean milliseconds per run>
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
extern const int null_prof_n;
void model_run_v(const float* const* inputs, float* const* outputs);
void null_prof_mark(int i);

#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    static double scale = 0;
    LARGE_INTEGER c;
    if (scale == 0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        scale = 1000.0 / (double)f.QuadPart;
    }
    QueryPerformanceCounter(&c);
    return scale * (double)c.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1000.0 * (double)t.tv_sec + (double)t.tv_nsec * 1e-6;
}
#endif

static double* spent;   /* per step, summed over the timed runs */
static double last;
static int timing;

/* the time since the previous mark belongs to the step before this one */
void null_prof_mark(int i) {
    const double t = now_ms();
    if (timing && i > 0) spent[i - 1] += t - last;
    last = now_ms();     /* the mark's own cost is left out */
}

int main(int argc, char** argv) {
    const int runs = argc > 1 && atoi(argv[1]) > 0 ? atoi(argv[1]) : 10;
    float* in[16];
    float* out[16];
    int i;
    int64_t j;
    if (model_run_num_inputs > 16 || model_run_num_outputs > 16) {
        fprintf(stderr, "too many inputs or outputs\n");
        return 2;
    }
    spent = (double*)calloc((size_t)(null_prof_n ? null_prof_n : 1), sizeof(double));
    if (!spent) return 2;
    for (i = 0; i < model_run_num_inputs; ++i) {
        in[i] = (float*)malloc((size_t)(model_run_input_sizes[i] ? model_run_input_sizes[i] : 1) * sizeof(float));
        if (!in[i]) return 2;
        for (j = 0; j < model_run_input_sizes[i]; ++j) in[i][j] = (float)sin(0.1 * (double)j + i);
    }
    for (i = 0; i < model_run_num_outputs; ++i) {
        out[i] = (float*)malloc((size_t)(model_run_output_sizes[i] ? model_run_output_sizes[i] : 1) * sizeof(float));
        if (!out[i]) return 2;
    }
    model_run_v((const float* const*)in, out);
    timing = 1;
    for (i = 0; i < runs; ++i) model_run_v((const float* const*)in, out);
    for (i = 0; i < null_prof_n; ++i) printf("step %d ms %.6f\n", i, spent[i] / runs);
    return 0;
}
