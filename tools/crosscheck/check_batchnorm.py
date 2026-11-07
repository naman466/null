"""Random single-node BatchNormalization models (inference), on inputs that
include NaN, infinities, -0.0 and large values, with ranks 2 to 5, any number
of channels, epsilon given or left at its default, zero variances, and now and
then a negative one.

The arbiter is the specification's formula,
    y = (x - mean) / sqrt(var + epsilon) * scale + B
evaluated in float64, in that order, and rounded to float32 once: nullc must
match it bit for bit. onnxruntime folds the formula into x * a + b in float32,
so it is held to that form's error bound, 8 * 2^-24 * (|x * a| + |b|), with
a = scale / sqrt(var + epsilon) and b = B - mean * a, and its largest distance
is reported. Where var + epsilon is exactly 0 (epsilon 0 and a zero
variance) the formula divides by zero and gives +-inf, NaN where x = mean;
onnxruntime's folded form gives inf - inf = NaN for the whole channel. The
specification decides, so those channels are left out of the comparison with
onnxruntime, and counted. A tenth of the models are invalid (a parameter with the wrong
number of values, or the training outputs asked for) and nullc must refuse them."""
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort, ulps

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 300, "batchnorm")
rng = np.random.default_rng(1106)
NAMES = ["scale", "B", "mean", "var"]


def random_input(shape):
    x = np.asarray(rng.standard_normal(shape) * rng.choice([1, 5, 100]), np.float32)
    flat = x.reshape(-1)
    for special in (np.nan, np.inf, -np.inf, -0.0):
        if rng.random() < 0.2:
            flat[rng.integers(0, flat.size)] = special
    return x


def random_case():
    rank = int(rng.integers(2, 6))
    shape = [1 if rng.random() < 0.5 else int(rng.integers(1, 4))] + [int(rng.integers(1, 6)) for _ in range(rank - 1)]
    c = shape[1]
    p = {"scale": rng.standard_normal(c) * rng.choice([0.5, 1, 3]),
         "B": rng.standard_normal(c),
         "mean": rng.standard_normal(c) * rng.choice([0.1, 1, 10]),
         "var": np.abs(rng.standard_normal(c)) * rng.choice([0.01, 1, 100])}
    if rng.random() < 0.2:
        p["var"][rng.integers(0, c)] = 0.0
    if rng.random() < 0.05:
        p["var"][rng.integers(0, c)] = -1.0                       # NaN for that channel, as the formula says
    p = {k: np.asarray(v, np.float32) for k, v in p.items()}
    attrs = {}
    if rng.random() < 0.6:
        attrs["epsilon"] = float(rng.choice([1e-5, 1e-3, 0.1, 0.0]))
    invalid = None
    if rng.random() < 0.1:
        if rng.random() < 0.5:
            name = NAMES[int(rng.integers(0, 4))]
            p[name] = np.asarray(np.append(p[name], 1.0), np.float32)
            invalid = f"{name} has {c + 1} values for {c} channels"
        else:
            invalid = "training outputs"
    return shape, p, attrs, invalid


def make_model(shape, p, attrs, invalid):
    outputs = ["y"] + (["running_mean", "running_var"] if invalid == "training outputs" else [])
    node = helper.make_node("BatchNormalization", ["x"] + NAMES, outputs, name="bn", **attrs)
    g = helper.make_graph([node], "bn", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
                          [numpy_helper.from_array(p[k], k) for k in NAMES])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    return model


def per_channel(v, shape):
    return v.astype(np.float64).reshape([1, -1] + [1] * (len(shape) - 2))


stats = dict(ok=0, fail=0, refused=0, zero_sd=0)
worst_ort = 0.0
for k in range(args.count):
    shape, p, attrs, invalid = random_case()
    model = make_model(shape, p, attrs, invalid)
    d = os.path.join(args.work, f"b{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "m.onnx")
    onnx.save(model, path)
    c_file = os.path.join(d, "model.c")
    ok, err = compile_model(args.nullc, path, c_file)
    if invalid:
        if ok:
            stats["fail"] += 1
            print(f"b{k:03d}: nullc accepted an invalid model ({invalid})")
        else:
            stats["refused"] += 1
        continue
    if not ok:
        stats["fail"] += 1
        print(f"b{k:03d}: nullc refuses {shape}: {err.strip()[:150]}")
        continue
    x = random_input(shape)
    got = run_generated(args.cc, c_file, x, output_shape(c_file), d)
    eps = np.float64(np.float32(attrs.get("epsilon", 1e-5)))
    s, b, m, v = (per_channel(p[n], shape) for n in NAMES)
    with np.errstate(all="ignore"):
        sd = np.sqrt(v + eps)
        truth = np.asarray((x.astype(np.float64) - m) / sd * s + b, np.float32)
        a = s / sd
        bound = 8 * 2.0 ** -24 * (np.abs(x * a) + np.abs(b - m * a))
    ref, rerr = run_ort(model, x)
    if ulps(got, truth) != 0:
        stats["fail"] += 1
        print(f"b{k:03d}: {shape} {attrs}: nullc {ulps(got, truth)} ulp from the formula in float64")
        continue
    if ref is None:
        stats["fail"] += 1
        print(f"b{k:03d}: onnxruntime refuses: {rerr[:100]}")
        continue
    zero_sd = np.broadcast_to(sd == 0, truth.shape)
    stats["zero_sd"] += int(np.any(sd == 0))
    finite = np.isfinite(truth) & np.isfinite(bound)
    if not np.array_equal(np.isnan(ref[~finite & ~zero_sd]), np.isnan(truth[~finite & ~zero_sd])):
        stats["fail"] += 1
        print(f"b{k:03d}: {shape}: onnxruntime puts NaN elsewhere")
        continue
    diff = np.abs(ref[finite].astype(np.float64) - truth[finite])
    if diff.size:
        worst_ort = max(worst_ort, float(np.max(diff / np.maximum(bound[finite], 1e-30))))
    if (diff > bound[finite]).any():
        stats["fail"] += 1
        print(f"b{k:03d}: {shape} {attrs}: onnxruntime outside its error bound")
        continue
    stats["ok"] += 1

print(f"{args.count} random BatchNormalization models: {stats['ok']} identical to the formula in float64, "
      f"{stats['refused']} invalid ones refused, {stats['fail']} failures; "
      f"{stats['zero_sd']} with a channel where var + epsilon = 0, left out of the comparison with onnxruntime; "
      f"onnxruntime at most {worst_ort:.2f} of its error bound")
raise SystemExit(1 if stats["fail"] else 0)
