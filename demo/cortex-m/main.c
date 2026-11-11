/*
 * LeNet, compiled by nullc, on a Cortex-M4 with no operating system
 *
 * runs the generated model on every input of an embedded .vectors file
 * (inputs and onnx runtime's outputs, recorded by
 * tests/fixtures/e2e/make_e2e_vectors.py), checks each output against onnx
 * runtime's like tests/e2e/run_vectors.c does (|got - ref| <= 1e-6 + 1e-5 |ref|,
 * and the same predicted class), and writes every output, raw, to a file on
 * the host through semihosting, so it can be compared bit for bit with the
 * same model built for the host.
 *
 * the generated model.c needs nothing from a C library here: no malloc, no
 * printf, no memcpy. the 64-bit integer arithmetic in its shapes comes from
 * libgcc, the compiler's own support library.
 */
#include <stdint.h>

#include "semihosting.h"

extern const int model_run_num_inputs;
extern const int model_run_num_outputs;
extern const int64_t model_run_input_sizes[];
extern const int64_t model_run_output_sizes[];
void model_run_v(const float* const* inputs, float* const* outputs);

extern const unsigned char null_vectors[], null_vectors_end[];

#define MAX_VALUES 4096

static float in_buf[MAX_VALUES], out_buf[MAX_VALUES], ref_buf[MAX_VALUES];

static void copy(void* to, const unsigned char* from, uint32_t n) {
    unsigned char* t = (unsigned char*)to;
    while (n--) *t++ = *from++;
}

static void put_int(int64_t v) {
    char s[24];
    int i = 23;
    int neg = v < 0;
    s[i] = 0;
    if (neg) v = -v;
    do {
        s[--i] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    if (neg) s[--i] = '-';
    sh_write0(s + i);
}

/* v >= 0 in the form 1.2e-07 */
static void put_sci(double v) {
    int e = 0;
    int64_t m;
    char s[16];
    if (v == 0) {
        sh_write0("0");
        return;
    }
    while (v >= 10) { v /= 10; ++e; }
    while (v < 1) { v *= 10; --e; }
    m = (int64_t)(v * 10 + 0.5);
    if (m >= 100) { m /= 10; ++e; }
    s[0] = (char)('0' + m / 10);
    s[1] = '.';
    s[2] = (char)('0' + m % 10);
    s[3] = 'e';
    s[4] = e < 0 ? '-' : '+';
    if (e < 0) e = -e;
    s[5] = (char)('0' + e / 10);
    s[6] = (char)('0' + e % 10);
    s[7] = 0;
    sh_write0(s);
}

static int64_t argmax(const float* v, int64_t n) {
    int64_t i, best = 0;
    for (i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

int main(void) {
    const unsigned char* p = null_vectors;
    int32_t trials, n_in, n_out, t;
    int64_t in_size, out_size, i;
    int32_t fd;
    long bad = 0, class_bad = 0;
    double worst = 0;
    const float* ins[1];
    float* outs[1];

    sh_write0("null demo: LeNet-5 on a Cortex-M4 (qemu mps2-an386), no OS, no malloc\n");
    if (null_vectors_end - null_vectors < 20 || p[0] != 'N' || p[7] != '1') {
        sh_write0("bad vectors file\n");
        return 1;
    }
    copy(&trials, p + 8, 4);
    copy(&n_in, p + 12, 4);
    copy(&n_out, p + 16, 4);
    if (n_in != 1 || n_out != 1 || model_run_num_inputs != 1 || model_run_num_outputs != 1) {
        sh_write0("the demo expects one input and one output\n");
        return 1;
    }
    copy(&in_size, p + 20, 8);
    copy(&out_size, p + 28, 8);
    if (in_size != model_run_input_sizes[0] || out_size != model_run_output_sizes[0] ||
        in_size > MAX_VALUES || out_size > MAX_VALUES) {
        sh_write0("the vectors do not fit the model\n");
        return 1;
    }
    p += 36;

    fd = sh_open("lenet_cortex_m.out", 4);
    if (fd < 0) {
        sh_write0("cannot open lenet_cortex_m.out on the host\n");
        return 1;
    }
    ins[0] = in_buf;
    outs[0] = out_buf;
    for (t = 0; t < trials; ++t) {
        copy(in_buf, p, (uint32_t)(in_size * 4));
        p += in_size * 4;
        copy(ref_buf, p, (uint32_t)(out_size * 4));
        p += out_size * 4;
        model_run_v(ins, outs);
        for (i = 0; i < out_size; ++i) {
            const double got = out_buf[i], ref = ref_buf[i];
            const double diff = got > ref ? got - ref : ref - got;
            if (!(diff <= 1e-6 + 1e-5 * (ref < 0 ? -ref : ref))) ++bad;
            if (diff > worst) worst = diff;
        }
        if (argmax(out_buf, out_size) != argmax(ref_buf, out_size)) ++class_bad;
        sh_write0("  input ");
        put_int(t);
        sh_write0(": class ");
        put_int(argmax(out_buf, out_size));
        sh_write0(", onnx runtime says ");
        put_int(argmax(ref_buf, out_size));
        sh_write0("\n");
        if (sh_write(fd, out_buf, (int32_t)(out_size * 4)) != 0) {
            sh_write0("cannot write the outputs\n");
            return 1;
        }
    }
    sh_close(fd);

    put_int(trials);
    sh_write0(" inputs: ");
    put_int(bad);
    sh_write0(" values outside tolerance, ");
    put_int(class_bad);
    sh_write0(" class mismatches, largest difference from onnx runtime ");
    put_sci(worst);
    sh_write0("\n");
    return bad == 0 && class_bad == 0 ? 0 : 1;
}
