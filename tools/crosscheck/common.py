"""Shared plumbing for the random cross-checks: compile a one-node model with
nullc, build the generated C with a tiny driver, run it, and read the output
back. Nothing here draws random numbers, so each check's results depend only
on its own seed."""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXE = ".exe" if sys.platform == "win32" else ""
# the same strictness the generated-code tests use, and no fused multiply-add,
# so results are bit-reproducible
CFLAGS = ["-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]


def parse_args(description, default_count, name):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--nullc", default=os.path.join(ROOT, "build-win", "null_cli" + EXE),
                   help="the null_cli executable (default: build-win/null_cli)")
    p.add_argument("--work", default=os.path.join(ROOT, "build-win", "crosscheck", name),
                   help="folder for the generated models, C files and binaries")
    p.add_argument("--count", type=int, default=default_count, help="number of random models")
    p.add_argument("--cc", default="gcc", help="C compiler for the generated code")
    args = p.parse_args()
    os.makedirs(args.work, exist_ok=True)
    return args


def compile_model(nullc, model_path, c_file):
    """nullc -o; returns (ok, stderr)"""
    r = subprocess.run([nullc, model_path, "-o", c_file], capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def output_shape(c_file, name="y"):
    """the output shape nullc wrote in the generated file's header comment"""
    m = re.search(r"\* output " + name + r": float32\[([0-9, ]*)\]", open(c_file).read())
    return [int(v) for v in m.group(1).split(",")] if m.group(1) else []


def run_generated(cc, c_file, x, out_shape, work):
    """build c_file with a driver for one input and one output, run it on x"""
    size_in, size_out = max(1, x.size), max(1, int(np.prod(out_shape)))
    driver = os.path.join(work, "driver.c")
    open(driver, "w").write(
        "#include <stdio.h>\nvoid model_run(const float*, float*);\n"
        f"static float x[{size_in}], y[{size_out}];\n"
        "int main(int argc, char** argv) { FILE* f = fopen(argv[1], \"rb\"); FILE* o = fopen(argv[2], \"wb\");\n"
        f"  (void)argc; if (fread(x, 4, {size_in}, f) != {size_in}) return 3; model_run(x, y); "
        f"fwrite(y, 4, {size_out}, o); fclose(o); return 0; }}\n")
    exe = os.path.join(work, "m" + EXE)
    subprocess.run([cc, *CFLAGS, c_file, driver, "-o", exe], check=True)
    np.ascontiguousarray(x, np.float32).tofile(os.path.join(work, "in.bin"))
    subprocess.run([exe, os.path.join(work, "in.bin"), os.path.join(work, "out.bin")], check=True)
    return np.fromfile(os.path.join(work, "out.bin"), np.float32).reshape(out_shape)


def run_ort(model, x):
    """(output, None) from onnxruntime, or (None, error message)"""
    import onnxruntime as ort
    ort.set_default_logger_severity(4)
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess.run(None, {"x": x})[0], None
    except Exception as e:
        return None, str(e)


def ulps(a, b):
    """largest distance in float32 units in the last place; NaN equals only NaN"""
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if (nan_a != nan_b).any():
        return float("inf")
    a, b = a[~nan_a], b[~nan_b]
    if (np.isinf(a) | np.isinf(b)).any() and (a[np.isinf(a) | np.isinf(b)] != b[np.isinf(a) | np.isinf(b)]).any():
        return float("inf")
    # map float bits onto a line where adjacent floats differ by 1 (and -0.0 == +0.0)
    ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return int(np.abs(ia - ib).max()) if a.size else 0
