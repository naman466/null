"""Time models compiled by Null against ONNX Runtime, single-threaded.

For each model: ONNX Runtime with one thread and every graph optimisation;
Null's generated C with the blocked kernels (the default); and the same C with
--reference-kernels, the plain loops. Each Null build is a separate program
(tools/bench/bench_model.c) timing model_run_v on one fixed input; the median
of at least ten runs is reported. Batch 1 throughout.

    .venv/Scripts/python -I tools/bench/bench.py                  # the PyTorch fixtures
    .venv/Scripts/python -I tools/bench/bench.py --torchvision    # plus full-size ResNet-18 and MobileNetV2
    .venv/Scripts/python -I tools/bench/bench.py --cflags "-O3 -march=native"

The C is built with -O2 -ffp-contract=off unless --cflags says otherwise:
without -ffp-contract=off, gcc may fuse multiplies and adds on targets that
have fma (-march=native on most machines today), which is faster but gives
different bits from the reference kernels. The script checks that the blocked
and reference builds' outputs are bit-identical, and how far both are from
ONNX Runtime's.
"""
import argparse, os, re, shlex, subprocess, sys, time

import numpy as np
import onnx
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "tools", "bench")

FIXTURES = [
    ("MLP (784-32-10)", "pytorch/mlp__legacy_dyn"),
    ("LeNet-5", "pytorch/lenet__legacy"),
    ("Transformer layer (d=32)", "pytorch/transformer__dynamo"),
    ("MobileNetV2 (1/4 width, 64x64)", "pytorch/mobilenet_v2_small__legacy"),
    ("ResNet-18 (1/8 width, 64x64)", "pytorch/resnet18_small__legacy"),
]


def shape_of(vi):
    return [d.dim_value if d.WhichOneof("value") == "dim_value" else 1 for d in vi.type.tensor_type.shape.dim]


def inputs_of(model):
    inits = {t.name for t in model.graph.initializer}
    return [vi for vi in model.graph.input if vi.name not in inits]


def bench_inputs(model):
    """the values bench_model.c fills each input with: sin(0.1 j + i)"""
    xs = {}
    for i, vi in enumerate(inputs_of(model)):
        s = shape_of(vi)
        xs[vi.name] = np.sin(0.1 * np.arange(int(np.prod(s))) + i).astype(np.float32).reshape(s)
    return xs


def time_ort(model, xs, seconds):
    if model.ir_version > 10:     # onnxruntime reads IR <= 13 while onnx writes 14; only the stamp changes
        model.ir_version = 10
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    for _ in range(3):
        ys = sess.run(None, xs)
    times = []
    start = time.perf_counter()
    while len(times) < 10 or time.perf_counter() - start < seconds:
        t0 = time.perf_counter()
        sess.run(None, xs)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2], [np.asarray(y, np.float32) for y in ys]


def build_null(args, path, work, tag, extra):
    c_file = os.path.join(work, tag + ".c")
    exe = os.path.join(work, tag + ".exe")
    out = os.path.join(work, tag + ".out")
    subprocess.run([args.nullc, path, "--batch", "1", *extra, "-o", c_file], check=True, capture_output=True)
    libm = [] if sys.platform == "win32" else ["-lm"]
    subprocess.run([args.cc, "-std=c99", *shlex.split(args.cflags), c_file,
                    os.path.join(HERE, "bench_model.c"), "-o", exe, *libm], check=True)
    r = subprocess.run([exe, str(args.seconds), out], check=True, capture_output=True, text=True)
    m = re.search(r"median_ms ([0-9.]+)", r.stdout)
    return float(m.group(1)), np.fromfile(out, np.float32)


def fmt(ms):
    return f"{ms:.3f} ms" if ms < 1 else f"{ms:.1f} ms" if ms < 100 else f"{ms:.0f} ms"


def export_torchvision(work):
    import torch, torchvision
    paths = []
    for label, name, factory in (("ResNet-18 (224x224)", "resnet18", torchvision.models.resnet18),
                                 ("MobileNetV2 (224x224)", "mobilenet_v2", torchvision.models.mobilenet_v2)):
        path = os.path.join(work, name + ".onnx")
        if not os.path.exists(path):
            torch.manual_seed(0)
            model = factory(weights=None).eval()     # random weights: nothing is downloaded
            torch.onnx.export(model, (torch.randn(1, 3, 224, 224),), path, input_names=["x"],
                              output_names=["y"], dynamo=False)
        paths.append((label, path))
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nullc", default=os.path.join(ROOT, "build-win", "nullc"))
    p.add_argument("--cc", default="gcc")
    p.add_argument("--cflags", default="-O2 -ffp-contract=off")
    p.add_argument("--seconds", type=float, default=2.0, help="time spent on each measurement")
    p.add_argument("--work", default=os.path.join(ROOT, "build-win", "bench"))
    p.add_argument("--torchvision", action="store_true", help="also full-size ResNet-18 and MobileNetV2")
    p.add_argument("models", nargs="*", help="more .onnx files")
    args = p.parse_args()
    os.makedirs(args.work, exist_ok=True)

    models = [(label, os.path.join(ROOT, "tests", "fixtures", name + ".onnx")) for label, name in FIXTURES]
    if args.torchvision:
        models += export_torchvision(args.work)
    models += [(os.path.basename(m), m) for m in args.models]

    print(f"cflags: {args.cflags}; onnxruntime {ort.__version__}, 1 thread\n")
    print("| Model | ONNX Runtime | Null, blocked | Null, reference | blocked vs reference | Null vs ONNX Runtime "
          "| blocked = reference | largest difference from ONNX Runtime |")
    print("|---|---:|---:|---:|---:|---:|---|---:|")
    failed = False
    for label, path in models:
        tag = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0])
        model = onnx.load(path)
        xs = bench_inputs(model)
        ort_ms, ys = time_ort(model, xs, args.seconds)
        fast_ms, fast = build_null(args, path, args.work, tag + "_blocked", [])
        ref_ms, ref = build_null(args, path, args.work, tag + "_ref", ["--reference-kernels"])
        want = np.concatenate([y.ravel() for y in ys])
        same = fast.tobytes() == ref.tobytes()
        diff = float(np.max(np.abs(fast - want))) if want.size else 0.0
        failed = failed or not same
        vs = f"{ort_ms / fast_ms:.1f}x faster" if fast_ms <= ort_ms else f"{fast_ms / ort_ms:.1f}x slower"
        print(f"| {label} | {fmt(ort_ms)} | {fmt(fast_ms)} | {fmt(ref_ms)} | {ref_ms / fast_ms:.1f}x faster | {vs} "
              f"| {'bit-identical' if same else 'DIFFERENT'} | {diff:.1e} |")
        sys.stdout.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
