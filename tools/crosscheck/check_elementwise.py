"""Random single-node elementwise models, compiled by nullc and run, on inputs
that include NaN, infinities, -0.0 and large values. The arbiter is numpy:
nullc's outputs must be identical to it (with -0.0 and +0.0 treated as
equal, and any NaN as any NaN), and to onnxruntime. For now the operator is
Clip, with bounds from initializers or Constant nodes, missing on either
side, infinite, or crossed (min > max); later commits add more. A missing
Clip bound is the largest finite float, as the onnx spec defines it (and
onnxruntime does; the onnx reference evaluator keeps infinities instead)."""
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort, ulps

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 300, "elementwise")
rng = np.random.default_rng(1105)


def random_input(shape):
    x = np.asarray(rng.standard_normal(shape) * rng.choice([1, 5, 100]), np.float32)
    flat = x.reshape(-1)
    for special in (np.nan, np.inf, -np.inf, -0.0, 0.0):
        if flat.size and rng.random() < 0.3:
            flat[rng.integers(0, flat.size)] = special
    return x


def clip_case():
    """a Clip model, and numpy's answer as a function of x"""
    nodes, inits, ins = [], [], ["x"]
    bounds = []
    for name in ("min", "max"):
        r = rng.random()
        if r < 0.2:
            ins.append("")                                   # omitted
            bounds.append(None)
            continue
        value = np.float32(rng.choice([0.0, 6.0, -1.0, float(rng.standard_normal() * 3), np.inf, -np.inf]))
        if rng.random() < 0.5:
            inits.append(numpy_helper.from_array(np.array(value, np.float32), name))
        else:
            nodes.append(helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.array(value, np.float32), name)))
        ins.append(name)
        bounds.append(value)
    while ins and ins[-1] == "":
        ins.pop()
    nodes.append(helper.make_node("Clip", ins, ["y"], name="clip"))
    big = np.finfo(np.float32).max
    lo = -big if bounds[0] is None else bounds[0]           # numeric_limits lowest() and max()
    hi = big if bounds[1] is None else bounds[1]

    def want(x):
        y = np.where(x < lo, lo, x)                          # numpy's maximum, NaN kept
        return np.asarray(np.where(y > hi, hi, y), np.float32)
    return nodes, inits, want, f"min {lo} max {hi}"


stats = dict(ok=0, fail=0)
for k in range(args.count):
    shape = [int(d) for d in rng.integers(1, 6, int(rng.integers(0, 5)))]
    nodes, inits, want, what = clip_case()
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    d = os.path.join(args.work, f"e{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "m.onnx")
    onnx.save(model, path)
    c_file = os.path.join(d, "model.c")
    ok, err = compile_model(args.nullc, path, c_file)
    if not ok:
        stats["fail"] += 1
        print(f"e{k:03d}: nullc refuses ({what}): {err.strip()[:150]}")
        continue
    x = random_input(shape)
    got = run_generated(args.cc, c_file, x, output_shape(c_file), d)
    ref, rerr = run_ort(model, x)
    if ulps(got, want(x)) != 0 or ref is None or ulps(got, ref) != 0:
        stats["fail"] += 1
        print(f"e{k:03d}: {what}: {ulps(got, want(x))} ulp from numpy, "
              f"{'onnxruntime refuses: ' + rerr[:80] if ref is None else str(ulps(got, ref)) + ' ulp from onnxruntime'}")
        continue
    stats["ok"] += 1

print(f"{args.count} random Clip models: {stats['ok']} identical to numpy and onnxruntime, {stats['fail']} failures")
raise SystemExit(1 if stats["fail"] else 0)
