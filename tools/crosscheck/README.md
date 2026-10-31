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
run is reproducible; `--count` sets the number of models, `--nullc` the compiler to test (default `build-win/null_cli`), and
`--work` where the models and binaries go (default `build-win/crosscheck/<name>`). A script exits non-zero if anything failed.

| Script | Arbiter | Result when last run |
|---|---|---|
| `check_conv.py` | ONNX shape inference for shapes; a numpy emulation of the documented summation order, bit for bit; ONNX Runtime within 10⁻⁵ | 400 models: 292 match, 78 rejected by both, 30 too small for the kernel (sized by ONNX, refused by ONNX Runtime and Null), 0 failures |
| `check_maxpool.py` | an emulation of the specification's formulas, bit for bit (ONNX Runtime, the reference evaluator and ONNX shape inference each deviate in corners, and the deviations are listed) | 500 models: 431 bit-identical, 69 invalid ones rejected, 0 failures |
| `check_reduce.py` | the mean in float64, rounded once, bit for bit; ONNX Runtime and the reference reported in ulps and against the float32 summation bound | 400 models: 390 bit-identical, 10 invalid ones rejected, 0 failures |

```bash
.venv/Scripts/python -I tools/crosscheck/check_reduce.py --count 400
```

## Sabotage

A check that has never failed proves little. `sabotage.py` takes a spec of deliberate bugs, applies each one alone, rebuilds,
and runs `ctest` and a cross-check to see what notices. Each edit must match exactly once and build, and every file is restored
byte for byte, verified by hash.

```bash
.venv/Scripts/python -I tools/crosscheck/sabotage.py tools/crosscheck/sabotages/reduce.py
```
