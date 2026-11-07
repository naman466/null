# Cross-checks

`ctest` proves that Null still does what its tests say. These scripts ask a harder question: on hundreds of random models, does
Null agree with the ONNX specification, with ONNX Runtime and with ONNX's own reference implementation, and where they disagree,
who is right? They are slower than `ctest` and need Python, so they are run by hand when an operator is added or changed.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r tools/crosscheck/requirements.txt    # .venv/bin/python elsewhere
```

The versions are pinned: they are the ones the recorded fixtures and the results below were produced with. A C compiler (`gcc` by
default, `--cc` to change it) builds each generated model.

## The checks

Each builds random one-node models, compiles them with `nullc`, runs the generated C and compares. Each has a fixed seed, so a
run is reproducible; `--count` sets the number of models, `--nullc` the compiler to test (default `build-win/nullc`), and
`--work` where the models and binaries go (default `build-win/crosscheck/<name>`). A script exits non-zero if anything failed.

| Script | Arbiter | Result when last run |
|---|---|---|
| `check_conv.py` | ONNX shape inference for shapes; a numpy emulation of the documented summation order, bit for bit; ONNX Runtime within 10⁻⁵ | 400 models: 292 match, 78 rejected by both, 30 too small for the kernel (sized by ONNX, refused by ONNX Runtime and Null), 0 failures |
| `check_maxpool.py` | an emulation of the specification's formulas, bit for bit (ONNX Runtime, the reference evaluator and ONNX shape inference each deviate in corners, and the deviations are listed) | 500 models: 431 bit-identical, 69 invalid ones rejected, 0 failures |
| `check_reduce.py` | the mean in float64, rounded once, bit for bit; ONNX Runtime and the reference reported in ulps and against the float32 summation bound | 400 models: 390 bit-identical, 10 invalid ones rejected, 0 failures |
| `check_shape_ops.py` | onnxruntime, which executes each model: Null's output type and shape must match it, every value Null computes at compile time must equal onnxruntime's output exactly, and what onnxruntime refuses Null must refuse | 600 models: 581 agree (393 with a value computed at compile time, all exact), 19 refused by both, 0 failures |
| `check_transformer_ops.py` | Transpose and Gather bit for bit against numpy and onnxruntime; LayerNormalization against the formula in float64 (at most 1 ulp, exact matches counted) and onnxruntime within 1e-5 | 600 models: 582 match (all 182 layer norms bit-identical), 10 refused by both, 8 broadcast scales refused as not supported yet, 0 failures |
| `check_elementwise.py` | Clip, Abs, Neg, Sqrt: numpy evaluating the specification, and onnxruntime, identical outputs (-0.0 = +0.0, NaN = NaN). Sigmoid, Tanh, Exp, Log, Pow: the true value (float64, rounded once), nullc within 4 ulp; onnxruntime within 4 ulp for Exp, Log and Pow, and within 2.5e-7 absolute for Sigmoid and Tanh (its Sigmoid is a fast approximation, 0 below about -17). Pow's other operand is a scalar or broadcast initializer, exponent or base, with exponents such as 0, 1/3 and negatives. Inputs with NaN, infinities and -0.0 | 450 random models, 50 per operator: 0 failures; nullc at most 1 ulp from the true value (Sigmoid), 0 for the rest; onnxruntime 1 ulp (Exp), 2 ulp (Log), 0 ulp (Pow), 8.3e-8 and 1.8e-7 absolute (Sigmoid, Tanh) |
| `check_optimize.py` | the same model built optimised and planned, and with `--no-opt --no-plan`: outputs bit-identical, both matching onnxruntime, and no more nodes after optimising | 300 random graphs (with Constant nodes and arithmetic on constants for folding, and Gemm layers with Relu for fusion): all bit-identical and matching, 3,092 nodes optimised down to 505 (906 constants folded, 153 Relus fused), 0 failures |

```bash
.venv/Scripts/python -I tools/crosscheck/check_reduce.py --count 400
```

## Sabotage

A check that has never failed proves little. `sabotage.py` takes a spec of deliberate bugs, applies each one alone, rebuilds,
and runs `ctest` and a cross-check to see what notices. Each edit must match exactly once and build, and every file is restored
byte for byte, verified by hash.

```bash
.venv/Scripts/python -I tools/crosscheck/sabotage.py tools/crosscheck/sabotages/reduce.py
.venv/Scripts/python -I tools/crosscheck/sabotage.py tools/crosscheck/sabotages/shape_ops.py
.venv/Scripts/python -I tools/crosscheck/sabotage.py tools/crosscheck/sabotages/transformer_ops.py
.venv/Scripts/python -I tools/crosscheck/sabotage.py tools/crosscheck/sabotages/optimize.py
.venv/Scripts/python -I tools/crosscheck/sabotage.py tools/crosscheck/sabotages/fold.py
.venv/Scripts/python -I tools/crosscheck/sabotage.py tools/crosscheck/sabotages/fuse.py
```
