"""Random Conv -> BatchNormalization (-> Relu) models, compiled by nullc twice:
optimised, where the fold-batchnorm pass gives the Conv new weights and a new
bias, and with --no-opt, where the BatchNorm runs as its own kernel. Folding
changes the arithmetic, so the two builds need not agree bit for bit; the
question is whether the folded one is as accurate. The arbiter is the same
model evaluated in float64 by the onnx reference
evaluator (onnxruntime has no float64 Conv). Each build's error is measured as
the largest |y - truth| over the largest |truth|, and the folded build must be
within 2x the unfolded one's error (or within 1e-6). Convolutions vary in
groups (depthwise included), strides, padding and bias; batch-norm
parameters are of realistic size, now and then with a zero variance."""
import os
import subprocess
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, output_shape, run_generated

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 200, "fold_batchnorm")
rng = np.random.default_rng(1108)


def random_case():
    group = int(rng.choice([1, 1, 2, 0]))                     # 0: depthwise
    c = int(rng.integers(1, 5)) * (2 if group == 2 else 1)
    group = c if group == 0 else group
    m = c * int(rng.integers(1, 3)) if group == c else int(rng.integers(1, 5)) * group
    k = int(rng.choice([1, 3, 5]))
    h = int(rng.integers(k, 10))
    conv = dict(kernel_shape=[k, k], strides=[int(rng.choice([1, 2]))] * 2, group=group)
    if rng.random() < 0.6:
        conv["pads"] = [k // 2] * 4
    w = rng.standard_normal((m, c // group, k, k)) * np.sqrt(2.0 / (c // group * k * k))
    p = dict(w=w, scale=rng.uniform(0.5, 1.5, m), beta=rng.uniform(-0.2, 0.2, m),
             mean=rng.uniform(-0.2, 0.2, m), var=rng.uniform(0.2, 2.0, m))
    if rng.random() < 0.7:
        p["b"] = rng.standard_normal(m) * 0.1
    if rng.random() < 0.1:
        p["var"][rng.integers(0, m)] = 0.0
    eps = float(rng.choice([1e-5, 1e-3]))
    return [1, c, h, h], p, conv, eps, rng.random() < 0.5


def make_model(shape, p, conv, eps, relu, dtype):
    np_t = np.float32 if dtype == TensorProto.FLOAT else np.float64
    nodes = [helper.make_node("Conv", ["x", "w"] + (["b"] if "b" in p else []), ["c"], name="conv", **conv),
             helper.make_node("BatchNormalization", ["c", "scale", "beta", "mean", "var"], ["bn" if relu else "y"],
                              name="bn", epsilon=eps)]
    if relu:
        nodes.append(helper.make_node("Relu", ["bn"], ["y"], name="relu"))
    inits = [numpy_helper.from_array(np.asarray(np.asarray(v, np.float32), np_t), k) for k, v in p.items()]
    g = helper.make_graph(nodes, "cbn", [helper.make_tensor_value_info("x", dtype, shape)],
                          [helper.make_tensor_value_info("y", dtype, None)], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    return model


def nullc(path, c_file, *flags):
    r = subprocess.run([args.nullc, path, "-o", c_file, *flags], capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


from onnx.reference import ReferenceEvaluator
stats = dict(ok=0, fail=0, folded=0)
ratios = []
worst = {"opt": 0.0, "noopt": 0.0}
for k in range(args.count):
    shape, p, conv, eps, relu = random_case()
    d = os.path.join(args.work, f"f{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "m.onnx")
    onnx.save(make_model(shape, p, conv, eps, relu, TensorProto.FLOAT), path)
    x = np.asarray(rng.standard_normal(shape), np.float32)
    outs = {}
    for variant, flags in (("opt", []), ("noopt", ["--no-opt"])):
        c_file = os.path.join(d, f"{variant}.c")
        ok, log = nullc(path, c_file, *flags)
        if not ok:
            break
        if variant == "opt" and "fold-batchnorm 1" in log:
            stats["folded"] += 1
        vd = os.path.join(d, variant)
        os.makedirs(vd, exist_ok=True)
        outs[variant] = run_generated(args.cc, c_file, x, output_shape(c_file), vd).astype(np.float64)
    if len(outs) != 2:
        stats["fail"] += 1
        print(f"f{k:03d}: nullc refuses: {log.strip()[:150]}")
        continue
    m64 = make_model(shape, p, conv, eps, relu, TensorProto.DOUBLE)
    truth = np.asarray(ReferenceEvaluator(m64).run(None, {"x": x.astype(np.float64)})[0],
                       np.float64).reshape(outs["opt"].shape)
    scale = max(float(np.max(np.abs(truth))), 1e-30)
    err = {v: float(np.max(np.abs(o - truth))) / scale for v, o in outs.items()}
    ratios.append(err["opt"] / max(err["noopt"], 1e-12))
    worst = {v: max(worst[v], err[v]) for v in worst}
    if err["opt"] > max(2 * err["noopt"], 1e-6):
        stats["fail"] += 1
        print(f"f{k:03d}: {shape} {conv}: folded error {err['opt']:.2e}, unfolded {err['noopt']:.2e}")
        continue
    stats["ok"] += 1

print(f"{args.count} random Conv + BatchNormalization models: {stats['folded']} folded, {stats['ok']} as accurate "
      f"as the unfolded build, {stats['fail']} failures; folded error / unfolded error: median "
      f"{np.median(ratios):.2f}, largest {np.max(ratios):.2f}; largest error, relative to the output's scale: "
      f"folded {worst['opt']:.2e}, unfolded {worst['noopt']:.2e}")
raise SystemExit(1 if stats["fail"] or stats["folded"] != args.count else 0)
