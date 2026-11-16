# Benchmark

`bench.py` times models compiled by Null against ONNX Runtime on one CPU thread, at batch 1. Each model is timed three ways:

- **ONNX Runtime**, with one intra-op and one inter-op thread, sequential execution and every graph optimisation on;
- **Null, blocked**: the generated C as `nullc` writes it by default;
- **Null, reference**: the same model compiled with `--reference-kernels`, which calls the plain one-output-at-a-time GEMM, MatMul and
  convolution loops instead of the blocked ones.

Each Null build is linked with [`bench_model.c`](bench_model.c), which fills the inputs with fixed values, warms up, then runs
`model_run_v` repeatedly for `--seconds` (at least ten runs) and reports the median. The script also checks that the blocked and
reference builds give **bit-identical** outputs, and fails if they do not.

```bash
.venv/Scripts/python -I tools/bench/bench.py                    # the PyTorch fixtures
.venv/Scripts/python -I tools/bench/bench.py --torchvision      # plus full-size ResNet-18 and MobileNetV2 (random weights)
.venv/Scripts/python -I tools/bench/bench.py --cflags "-O3 -march=native -ffp-contract=off"
```

The generated C is built with `-O2 -ffp-contract=off` by default. Leaving out `-ffp-contract=off` lets gcc fuse multiplies and adds
on machines with FMA, which is faster but no longer gives the reference kernels' bits.

## Where the time goes

`profile_layers.py` shows a single model's time step by step. It compiles the model, puts a timer call before every step of the
generated `model_run`, builds it with [`profile_model.c`](profile_model.c), and prints each step's mean time per run, its share of
the total and, for convolutions and matrix products, the GFLOP/s reached, then a summary by operator. Steps that move no data,
like a Reshape that only renames its input, are left out.

```bash
.venv/Scripts/python -I tools/bench/profile_layers.py tests/fixtures/pytorch/resnet18_small__legacy.onnx
.venv/Scripts/python -I tools/bench/profile_layers.py --torchvision --top 6       # the six slowest steps of each full-size net
.venv/Scripts/python -I tools/bench/profile_layers.py model.onnx --reference-kernels --cflags "-O2 -march=native -ffp-contract=off"
```

```text
resnet18.onnx: 251.036 ms per run, mean of 10 (cflags -O2 -ffp-contract=off)

step                         op                 work                                ms  share  GFLOP/s
/layer4/layer4.1/conv1/Conv  Conv               512->512 3x3 7x7                  18.7   7.4%     12.4
/layer4/layer4.1/conv2/Conv  Conv               512->512 3x3 7x7                  18.1   7.2%     12.8
...
by operator          steps        ms  share  GFLOP/s
Conv                    20     246.4  98.1%     14.7
MaxPool                  1     3.839   1.5%
```

On both full-size networks, convolutions take 97 to 99% of the time. Before the GEMM tile kept its sums in vector registers, every
convolution ran at 3 to 5 GFLOP/s whatever its shape, which pointed at the inner kernel. Now they run at 12 to 17 GFLOP/s with SSE2
and 25 to 33 with AVX. A taller tile no longer makes them faster, and patch packing is about a sixth of ResNet-18's time.

## Results

Measured on 2026-11-16 on a laptop x86-64 CPU (Windows, gcc 15.2, onnxruntime 1.31.1).

Built with `-O2 -ffp-contract=off`, the default, which on x86-64 gives the kernels SSE2's vector registers:

| Model | ONNX Runtime | Null, blocked | Null, reference | blocked vs reference | Null vs ONNX Runtime |
|---|---:|---:|---:|---:|---:|
| MLP (784-32-10) | 0.033 ms | 0.006 ms | 0.038 ms | 5.9× faster | 5.1× faster |
| LeNet-5 | 0.080 ms | 0.133 ms | 0.415 ms | 3.1× faster | 1.7× slower |
| Transformer layer (d=32) | 0.091 ms | 0.043 ms | 0.092 ms | 2.1× faster | 2.1× faster |
| MobileNetV2 (¼ width, 64×64) | 0.375 ms | 1.2 ms | 8.6 ms | 7.2× faster | 3.2× slower |
| ResNet-18 (⅛ width, 64×64) | 0.281 ms | 1.2 ms | 5.3 ms | 4.2× faster | 4.4× slower |
| ResNet-18 (224×224) | 66.3 ms | 252 ms | 2960 ms | 11.7× faster | 3.8× slower |
| MobileNetV2 (224×224) | 15.1 ms | 53.5 ms | 892 ms | 16.7× faster | 3.5× slower |

With `-O3 -march=native -ffp-contract=off`, which gives them AVX2 (but still not FMA):

| Model | ONNX Runtime | Null, blocked | Null, reference | blocked vs reference | Null vs ONNX Runtime |
|---|---:|---:|---:|---:|---:|
| MLP (784-32-10) | 0.033 ms | 0.004 ms | 0.038 ms | 9.1× faster | 7.8× faster |
| LeNet-5 | 0.079 ms | 0.083 ms | 0.403 ms | 4.9× faster | 1.0× slower |
| Transformer layer (d=32) | 0.092 ms | 0.032 ms | 0.082 ms | 2.6× faster | 2.9× faster |
| MobileNetV2 (¼ width, 64×64) | 0.370 ms | 1.0 ms | 8.7 ms | 8.5× faster | 2.8× slower |
| ResNet-18 (⅛ width, 64×64) | 0.288 ms | 0.802 ms | 3.2 ms | 4.0× faster | 2.8× slower |
| ResNet-18 (224×224) | 65.5 ms | 122 ms | 2497 ms | 20.4× faster | 1.9× slower |
| MobileNetV2 (224×224) | 14.6 ms | 31.0 ms | 876 ms | 28.2× faster | 2.1× slower |

Every model's blocked and reference outputs were identical, and their largest difference from ONNX Runtime is the same as the
end-to-end tests report.

