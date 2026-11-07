"""Random single-node elementwise models, compiled by nullc and run, on inputs
that include NaN, infinities, -0.0 and large values, checked against numpy
and onnxruntime.

The operators take turns: Clip, Sigmoid, Tanh, Exp, Log, Sqrt, Abs, Neg, Pow.
Pow's other operand is an initializer, a scalar (as exporters write x ** 2)
or broadcast along the last dims, and is the exponent, or, a quarter of the
time, the base with x as the exponent. Exponents include 0, 1/3, negative
and non-integer values, so negative bases give NaN as numpy's do.

Clip, Abs, Neg and Sqrt must be identical to numpy and to onnxruntime (with
-0.0 and +0.0 treated as equal, and any NaN as any NaN): the first three are
exact, and sqrt is correctly rounded by IEEE 754. Clip's bounds come from
initializers or Constant nodes, missing on either side, infinite, or crossed
(min > max). A missing Clip bound is the largest finite float, as the onnx
spec defines it (and onnxruntime does; the onnx reference evaluator keeps
infinities instead). Log and Sqrt get mostly non-negative inputs.

Sigmoid, Tanh, Exp, Log and Pow call the C library's expf, tanhf, logf and
powf, which are not correctly rounded, so for them the arbiter is the true
value: numpy in float64, rounded once to float32. nullc must be within
MAX_ULPS of it everywhere, tiny results included, and so must onnxruntime's
Exp, Log and Pow.
onnxruntime's Sigmoid cannot be held to that: it is a fast approximation with
an absolute error of about 6e-8 (its sigmoid(-15) is 3.58e-7 against a true
3.06e-7, and below about -17 it returns 0), so Sigmoid and Tanh, whose
outputs lie in [-1, 1], hold onnxruntime only to an absolute MAX_ABS. The
script reports the largest distance each was seen at."""
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort, ulps

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 450, "elementwise")
rng = np.random.default_rng(1105)
MAX_ULPS = 4
MAX_ABS = 2.5e-7          # 4 ulp at 1, the largest output either can give


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


def sigmoid64(x):
    x = x.astype(np.float64)
    with np.errstate(over="ignore"):
        e = np.exp(-np.abs(x))                               # never overflows
        y = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return np.asarray(np.where(np.isnan(x), np.nan, y), np.float32)


def in64(f):
    """f evaluated in float64, rounded once to float32"""
    def g(x):
        with np.errstate(all="ignore"):
            return np.asarray(f(x.astype(np.float64)), np.float32)
    return g


TRUTH = {"Sigmoid": sigmoid64, "Tanh": in64(np.tanh), "Exp": in64(np.exp), "Log": in64(np.log),
         "Sqrt": in64(np.sqrt), "Abs": np.abs, "Neg": np.negative}
EXACT = {"Abs", "Neg", "Sqrt"}         # sqrt in float64 rounded once is the correctly rounded float32 sqrt
BOUNDED = {"Sigmoid", "Tanh"}          # outputs within [-1, 1]: onnxruntime held to MAX_ABS


def unary_case(op):
    """a one-input model, and the true answer"""
    return [helper.make_node(op, ["x"], ["y"], name=op.lower())], [], TRUTH[op], op


def pow_case(shape):
    """a Pow model with an initializer as one operand, and the true answer"""
    tail = shape[len(shape) - int(rng.integers(0, len(shape) + 1)):] if shape else []
    special = [2.0, 3.0, 0.5, -1.0, 0.0, -2.0, 1.0 / 3.0, 1.5]
    v = np.array([rng.choice(special) if rng.random() < 0.6 else rng.standard_normal() * 3
                  for _ in range(int(np.prod(tail)) if tail else 1)], np.float32).reshape(tail)
    base_is_init = rng.random() < 0.25
    if base_is_init:
        v = np.asarray(v * 2, np.float32)                       # bases, negative ones included
    ins = ["w", "x"] if base_is_init else ["x", "w"]
    nodes = [helper.make_node("Pow", ins, ["y"], name="pow")]
    inits = [numpy_helper.from_array(v, "w")]

    def want(x):
        a, b = (v, x) if base_is_init else (x, v)
        with np.errstate(all="ignore"):
            return np.asarray(np.power(a.astype(np.float64), b.astype(np.float64)), np.float32)
    return nodes, inits, want, f"Pow({'w, x' if base_is_init else 'x, w'}), w {list(tail)}"


stats = dict(ok=0, fail=0)
OPS = ["Clip", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Abs", "Neg", "Pow"]
worst = {op: [0, 0.0 if op in BOUNDED else 0] for op in OPS[1:]}         # nullc in ulps; onnxruntime absolute (BOUNDED) or in ulps
for k in range(args.count):
    shape = [int(d) for d in rng.integers(1, 6, int(rng.integers(0, 5)))]
    op = OPS[k % len(OPS)]
    nodes, inits, want, what = (clip_case() if op == "Clip" else pow_case(shape) if op == "Pow"
                                else unary_case(op))
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
    if op in ("Log", "Sqrt") and rng.random() < 0.7:
        x = np.where(np.isnan(x), x, np.abs(x)).astype(np.float32)   # mostly in the domain; -0.0 becomes +0.0
    got = run_generated(args.cc, c_file, x, output_shape(c_file), d)
    ref, rerr = run_ort(model, x)
    if ref is None:
        stats["fail"] += 1
        print(f"e{k:03d}: {what}: onnxruntime refuses: {rerr[:80]}")
        continue
    truth = want(x)
    if op == "Clip" or op in EXACT:
        if op != "Clip":
            worst[op] = [max(worst[op][0], ulps(got, truth)), max(worst[op][1], ulps(ref, truth))]
        bad = ulps(got, truth) != 0 or ulps(got, ref) != 0
    elif op in BOUNDED:
        worst[op][0] = max(worst[op][0], ulps(got, truth))
        ok_nan = np.array_equal(np.isnan(ref), np.isnan(truth))
        diff = float(np.abs(np.where(np.isnan(truth), 0, ref.astype(np.float64) - truth)).max()) if ok_nan else np.inf
        worst[op][1] = max(worst[op][1], diff)
        bad = ulps(got, truth) > MAX_ULPS or diff > MAX_ABS
    else:
        worst[op] = [max(worst[op][0], ulps(got, truth)), max(worst[op][1], ulps(ref, truth))]
        bad = ulps(got, truth) > MAX_ULPS or ulps(ref, truth) > MAX_ULPS
    if bad:
        stats["fail"] += 1
        print(f"e{k:03d}: {what}: nullc {ulps(got, truth)} ulp from numpy, onnxruntime {ulps(ref, truth)} ulp")
        continue
    stats["ok"] += 1

for op, (n, o) in worst.items():
    print(f"{op}: largest distance from the true value, nullc {n} ulp, onnxruntime "
          + (f"{o:.2g} absolute" if op in BOUNDED else f"{o} ulp"))
print(f"{args.count} random models, {len(OPS)} operators: {stats['ok']} pass, {stats['fail']} failures")
raise SystemExit(1 if stats["fail"] else 0)
