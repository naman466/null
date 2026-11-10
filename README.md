# Null: An Educational Deep Learning Compiler

[![ci](https://github.com/naman466/Null/actions/workflows/ci.yml/badge.svg)](https://github.com/naman466/Null/actions/workflows/ci.yml)

**Null** compiles a trained neural network, given as an `.onnx` file, ahead of time into **one standalone C99 file**. The
generated `model.c` holds the weights, the kernels it calls and a static memory arena: it needs no runtime library, no
allocator and no C++, and it builds with any C compiler.

It is written in **C++20** as a learning project: every stage, from reading protobuf by hand to planning memory, is small enough
to read, and every result is checked against the ONNX specification and ONNX Runtime rather than taken on trust.

```bash
nullc model.onnx -o model.c        # then: cc -O2 -c model.c, and call model_run(input, output)
```

---

## Results

Every PyTorch model below is exported by both of PyTorch's ONNX exporters (the TorchScript-based one and the newer dynamo one),
compiled by Null, built with gcc, run on recorded inputs and compared with ONNX Runtime. These are `ctest` tests, run in CI.

| Model | ONNX nodes | Weights | Arena (planned) | Buffers unplanned | Largest difference from ONNX Runtime | Class mismatches |
|---|---:|---:|---:|---:|---|---:|
| MLP (784-32-10, softmax), dynamic batch | 4 | 99.4 KB | 192 B | 256 B | 6.0e-8 | 0 |
| LeNet-5 | 8 | 241.0 KB | 23.0 KB | 31.8 KB | 6.4e-6 | 0 |
| Conv + BatchNorm + ReLU | 4 | 2.4 KB | 64.1 KB | 64.1 KB | 6.7e-6 | 0 |
| Transformer encoder layer (d=32, 4 heads) | 39 | 33.0 KB | 6.0 KB | 32.0 KB | 4.8e-7 | 0 |
| MobileNetV2 (¼ width, 64×64) | 101 | 959.4 KB | 384.0 KB | 1.4 MB | 3.5e-5 | 0 |
| ResNet-18 (⅛ width, 64×64) | 40 | 690.5 KB | 40.0 KB | 167.3 KB | 7.3e-4 (2.2e-5 relative) | 0 |
| DenseNet-style block (`torch.cat`) | 7 | 1.9 KB | 22.0 KB | 26.1 KB | 2.9e-6 | 0 |
| ShuffleNet-style block (`chunk`, `split`, slicing, `flip`) | 14 | 16.5 KB | 20.0 KB | 36.2 KB | 1.1e-5 | 0 |
| CBAM-style attention (mean, max, sum) | 19 | 1.8 KB | 4.2 KB | 8.2 KB | 1.9e-6 | 0 |
| Average pooling, three ways | 8 | 8.9 KB | 7.5 KB | 14.9 KB | 1.2e-6 | 0 |

- **Differences** are absolute, the worst over both exports and 8 random inputs each. Null works in float32 like ONNX Runtime,
  but sums reductions and normalisations in double, so most of the difference is ONNX Runtime's rounding. ResNet's outputs are
  around 30, which is why its absolute difference is larger.
- **Arena (planned)** is the memory for intermediate tensors after Null's planner packs buffers whose lifetimes do not overlap
  into one static array; **unplanned** is what one buffer per tensor would need. Reshapes, flattens and contiguous slices take
  no memory at all: they are pointers into an existing buffer.
- **Optimised and unoptimised** builds of every model give bit-identical outputs, which every end-to-end test checks: the default
  build (optimised, memory planned, blocked kernels) against `--no-opt --no-plan --reference-kernels`.

### Speed

Batch 1, one thread, on a laptop x86-64 CPU; ONNX Runtime 1.31 with every graph optimisation, Null's C built by gcc 15
([`tools/bench`](tools/bench/README.md)). Full-size networks have random weights.

| Model | ONNX Runtime | Null `-O2` | Null `-O2`, reference kernels | Null `-O3 -march=native` |
|---|---:|---:|---:|---:|
| ResNet-18, 224×224 | 75 ms | 662 ms | 3152 ms | 305 ms |
| MobileNetV2, 224×224 | 16 ms | 115 ms | 966 ms | 84 ms |
| LeNet-5 | 0.087 ms | 0.226 ms | 0.452 ms | 0.204 ms |
| Transformer encoder layer | 0.099 ms | 0.064 ms | 0.097 ms | 0.077 ms |
| MLP (784-32-10) | 0.035 ms | 0.012 ms | 0.040 ms | 0.057 ms |

The GEMM and convolution kernels compute tiles of outputs at once over cache-sized panels, and the C compiler vectorises them, which
makes full-size networks 5 to 12 times faster than the plain loops. Each output is still the same sum in the same order, so the fast
and plain kernels give **the same bits** (any NaN result is a NaN in both), which every end-to-end test checks. Small fully connected
models beat ONNX Runtime, whose per-operator overhead dominates there; on convolutional networks ONNX Runtime's hand-written AVX2 and
FMA kernels are still 4 to 9 times faster than portable C. (The MLP's `-O3` time is a gcc 15 quirk: its interprocedural constant
propagation specialises the tile for that layer badly; `-O2 -march=native` runs it in 0.005 ms.)

### Cross-checks

`ctest` proves Null still does what its tests say. The scripts in [`tools/crosscheck`](tools/crosscheck/README.md) ask a harder
question on hundreds of random one-operator models each: does Null agree with the ONNX specification, and where ONNX Runtime or
ONNX's own reference implementation disagree with it, who is right? The arbiter for each is an emulation of the specification's
formulas, and each reference's known deviations are modelled exactly so that a new one fails the run.

| Check | Random models | Result |
|---|---:|---|
| Conv | 400 | 0 failures; bit-identical to the documented summation order |
| MaxPool, AveragePool | 500 + 500 | 0 failures; bit-identical to the specification |
| ReduceMean, ReduceSum, ReduceMax | 400 + 400 | 0 failures; bit-identical to the float64 result rounded once |
| Concat, Slice, Split | 300 + 400 | 0 failures; bit-identical, NaN and -0.0 included |
| Shape operators, Transpose, Gather, LayerNorm, BatchNorm, elementwise | 600 + 600 + 300 + 450 | 0 failures |
| Optimiser (folding, fusion, BatchNorm folding) | 300 + 200 | 0 failures; optimised and unoptimised builds bit-identical |

Along the way they found real bugs in Null, each fixed in its own commit, and deviations from the specification in ONNX Runtime
(for example its ReduceMax keeps a NaN only if it comes first, and it does not clamp a negative `SAME` padding).

### Tests

109 `ctest` tests: 349 C++ test cases, 81 runtime kernel tests compiled as both C99 and C++, a protobuf fuzzer, command-line
tests, a compile of every generated model as strict C99 (`-Wall -Wextra -Wpedantic -Werror`), and 33 end-to-end runs against
ONNX Runtime. CI runs them on Linux (gcc, and gcc with the undefined-behaviour sanitizer) and on Windows (MinGW g++).

---

## What it supports

**Operators:** Conv, MaxPool, AveragePool, GlobalAveragePool, Gemm, MatMul, BatchNormalization, LayerNormalization, Softmax,
Relu, Sigmoid, Tanh, Exp, Log, Sqrt, Abs, Neg, Pow, Clip, Add, Sub, Mul, Div (with broadcasting), ReduceMean, ReduceSum,
ReduceMax, Concat, Slice, Split, Transpose, Gather, Reshape, Flatten, Squeeze, Unsqueeze, Constant, Dropout, Identity.
Opsets 13 to 28, float32 data. Integer tensors are computed at compile time when they are shapes, which is also where Shape and
Cast are supported.

**Optimisations:** constant folding (including the shape arithmetic exporters emit), identity and dead-code removal, ReLU
fusion into Gemm and Conv, BatchNorm folding into Conv, Linear weights transposed at compile time, static memory planning, and
blocked GEMM and convolution kernels.

**Not yet:** integer or boolean data at run time, control flow, multithreading, and kernels for a particular instruction set: the
runtime is portable C99 that the C compiler vectorises.

---

## Compilation pipeline

1. **Frontend:** a hand-written protobuf reader turns the `.onnx` file into Null's IR, refusing anything malformed with a
   message that names the problem. It survives a mutation fuzzer.
2. **Shape inference:** every tensor's type and shape, and the value of every integer tensor that can be known at compile
   time, so the shape computations exporters write disappear before code generation.
3. **Optimiser:** small passes run to a fixed point, each one removing or rewriting nodes without changing any output bit.
4. **Code generation:** liveness analysis, a greedy memory plan into one static arena, and C that calls the runtime's kernels
   with every shape and stride as a constant. Only the parts of the runtime a model uses are embedded.
5. **Runtime:** header-only C99 kernels. GEMM and convolution work on 4 × 16 tiles of outputs over cache-sized panels, with each
   output's sum in the same order as the plain reference loops, so they are several times faster and give the same bits.

## Project structure

```text
Null/
├── src/
│   ├── core/           # IR: tensors, nodes, attributes, the graph
│   ├── frontend/       # protobuf reader and ONNX to IR
│   ├── analysis/       # shape inference
│   ├── opt/            # optimisation passes
│   ├── backend/        # memory planning and C code generation
│   └── main.cpp        # the nullc command-line tool
├── runtime/            # the C99 kernels generated code embeds, one header per family
├── tests/              # unit, fixture, CLI, generated-code and end-to-end tests
├── tools/crosscheck/   # random cross-checks against the specification and ONNX Runtime
├── tools/bench/        # benchmark against ONNX Runtime
└── .github/workflows/  # CI
```

---

## Getting started

**Prerequisites:** a C++20 compiler (GCC 10+ or Clang 12+), CMake 3.15+, and a C compiler for the generated code. Python is
needed only to regenerate fixtures or run the cross-checks.

```bash
git clone https://github.com/naman466/Null.git
cd Null
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/nullc model.onnx                  # load and check a model
./build/nullc model.onnx --dump           # every input, output, weight and node, with shapes
./build/nullc model.onnx --batch 1 -o model.c
```

`--batch N` or `--input-shape name=d0,d1,...` fix dynamic dimensions, `--no-opt` and `--no-plan` turn off the optimiser and the
memory planner, and `--reference-kernels` swaps the blocked kernels for the plain loops. The generated file documents its own interface in its first lines; build it with `-ffp-contract=off` for
results that are bit-reproducible across compilers.

For a build that traps on undefined behaviour and bounds-checks standard containers:

```bash
cmake -S . -B build-san -DNULL_SANITIZE=ON && cmake --build build-san -j && ctest --test-dir build-san
```

To regenerate the test fixtures, or run the cross-checks, see [`tools/crosscheck`](tools/crosscheck/README.md) for the
pinned Python environment; the benchmark is in [`tools/bench`](tools/bench/README.md).

---

## License

This project is licensed under the MIT License. See [`LICENSE.txt`](LICENSE.txt) for details.
