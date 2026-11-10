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

## Results

Measured on 2026-11-10 on a laptop x86-64 CPU (Windows, gcc 15.2, onnxruntime 1.31.1).

Built with `-O2 -ffp-contract=off`, the default:

| Model | ONNX Runtime | Null, blocked | Null, reference | blocked vs reference | Null vs ONNX Runtime |
|---|---:|---:|---:|---:|---:|
| MLP (784-32-10) | 0.035 ms | 0.012 ms | 0.040 ms | 3.4× faster | 2.9× faster |
| LeNet-5 | 0.087 ms | 0.226 ms | 0.452 ms | 2.0× faster | 2.6× slower |
| Transformer layer (d=32) | 0.099 ms | 0.064 ms | 0.097 ms | 1.5× faster | 1.6× faster |
| MobileNetV2 (¼ width, 64×64) | 0.490 ms | 2.5 ms | 9.4 ms | 3.8× faster | 5.1× slower |
| ResNet-18 (⅛ width, 64×64) | 0.333 ms | 2.1 ms | 5.8 ms | 2.7× faster | 6.4× slower |
| ResNet-18 (224×224) | 74.8 ms | 662 ms | 3152 ms | 4.8× faster | 8.9× slower |
| MobileNetV2 (224×224) | 15.6 ms | 115 ms | 966 ms | 8.4× faster | 7.4× slower |

With `-O3 -march=native -ffp-contract=off`, which lets gcc use AVX2 (but still not FMA):

| Model | ONNX Runtime | Null, blocked | Null, reference | blocked vs reference | Null vs ONNX Runtime |
|---|---:|---:|---:|---:|---:|
| MLP (784-32-10) | 0.036 ms | 0.057 ms | 0.041 ms | 0.7× | 1.6× slower |
| LeNet-5 | 0.087 ms | 0.204 ms | 0.435 ms | 2.1× faster | 2.4× slower |
| Transformer layer (d=32) | 0.099 ms | 0.077 ms | 0.086 ms | 1.1× faster | 1.3× faster |
| MobileNetV2 (¼ width, 64×64) | 0.399 ms | 1.5 ms | 9.4 ms | 6.2× faster | 3.8× slower |
| ResNet-18 (⅛ width, 64×64) | 0.315 ms | 1.5 ms | 3.4 ms | 2.4× faster | 4.6× slower |
| ResNet-18 (224×224) | 72.9 ms | 305 ms | 2787 ms | 9.1× faster | 4.2× slower |
| MobileNetV2 (224×224) | 16.5 ms | 83.9 ms | 996 ms | 11.9× faster | 5.1× slower |

Every model's blocked and reference outputs were identical, and their largest difference from ONNX Runtime is the same as the
end-to-end tests report.

The MLP's `-O3 -march=native` time is a gcc 15 quirk: interprocedural constant propagation (`-fipa-cp`) specialises the one-row tile
for that layer badly; `-O2 -march=native` runs it in 0.005 ms, and so does `-O3 -march=native -fno-ipa-cp`.

