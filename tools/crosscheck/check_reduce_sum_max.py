"""Random single-node ReduceSum and ReduceMax models: ranks 0 to 5, some dims
empty, axes as an input (ReduceSum from opset 13, ReduceMax from 18) or as an
attribute (ReduceMax at opset 13), negative axes, keepdims and
noop_with_empty_axes, and inputs with NaN, infinities, -0.0 and sums that
overflow. Some models are invalid (an axis twice or out of range, or ReduceSum
with the axes attribute it lost in opset 13) and nullc must refuse them.

The arbiter is the specification. A sum is computed in float64 and rounded to
float32 once, and nullc must match it bit for bit; the sum of an empty set is
0. A max is exact: numpy's max, which the specification says it follows, so a
NaN anywhere gives NaN, and the max of an empty set is -inf. nullc must match
both, with any NaN equal to any NaN and -0.0 equal to +0.0.

onnxruntime and the onnx reference evaluator are compared too. A sum of
theirs, added in float32, is reported in ulps and against the float32
summation bound n * 2^-24 * sum|x|. A max must match exactly, except where
onnxruntime deviates from the specification in one of its two known ways,
each modelled here so that any other deviation fails: its max keeps a NaN only
if it is the first element of the set (it starts from the first and replaces
it only by a larger value, and nothing is larger than NaN, or than anything
when the current value is NaN), and on an empty input it ignores every
negative axis, passing the input through if none is left. The second applies
to sums as well."""
import collections
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort, ulps

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 400, "reduce_sum_max")
rng = np.random.default_rng(1212)


def random_config():
    op = str(rng.choice(["ReduceSum", "ReduceMax"]))
    rank = int(rng.integers(0, 6))
    shape = [int(rng.integers(0 if rng.random() < 0.08 else 1, 6)) for _ in range(rank)]
    opset = int(rng.choice([13, 18]))
    cfg = dict(op=op, shape=shape, opset=opset, attrs={}, axes=None, as_attr=op == "ReduceMax" and opset == 13,
               invalid=False)
    form = rng.choice(["axes", "axes", "axes", "none", "empty"])
    if form == "axes" and rank > 0:
        k = int(rng.integers(1, rank + 1))
        axes = [int(a) for a in rng.choice(rank, k, replace=False)]
        axes = [a - rank if rng.random() < 0.4 else a for a in axes]
        if rng.random() < 0.04:
            axes.append(axes[0])                                     # repeated
            cfg["invalid"] = True
        if rng.random() < 0.04:
            axes.append(rank if rng.random() < 0.5 else -rank - 1)   # out of range
            cfg["invalid"] = True
        cfg["axes"] = axes
        if op == "ReduceSum" and rng.random() < 0.05:
            cfg["as_attr"] = True                                    # gone since opset 13
            cfg["invalid"] = True
    elif form == "empty":
        cfg["axes"] = []
    if rng.random() < 0.7:
        cfg["attrs"]["keepdims"] = int(rng.integers(0, 2))
    if not cfg["as_attr"] and rng.random() < 0.3:
        cfg["attrs"]["noop_with_empty_axes"] = int(rng.integers(0, 2))
    return cfg


def make_model(cfg):
    inits, inputs, attrs = [], ["x"], dict(cfg["attrs"])
    if cfg["axes"] is not None:
        if not cfg["as_attr"]:
            inits.append(numpy_helper.from_array(np.array(cfg["axes"], np.int64), "axes"))
            inputs.append("axes")
        elif cfg["axes"]:            # an empty attribute means the same as none
            attrs["axes"] = cfg["axes"]
    node = helper.make_node(cfg["op"], inputs, ["y"], name="r", **attrs)
    g = helper.make_graph([node], "reduce", [helper.make_tensor_value_info("x", TensorProto.FLOAT, cfg["shape"])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], initializer=inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", cfg["opset"])])
    model.ir_version = 8
    return model


def reduced(cfg, axes):
    """the dims reduced for these axes: none or empty ones mean all of them,
    unless noop_with_empty_axes"""
    rank = len(cfg["shape"])
    norm = sorted(a % rank for a in axes or [])
    if not norm and not cfg["attrs"].get("noop_with_empty_axes", 0):
        norm = list(range(rank))
    return norm


def first_wins_max(x, norm):
    """onnxruntime's max: the first element of each set, replaced only by a larger one"""
    kept = [d for d in range(x.ndim) if d not in norm]
    n = int(np.prod([x.shape[d] for d in norm]))
    flat = np.transpose(x, kept + norm).reshape([x.shape[d] for d in kept] + [n])
    if flat.shape[-1] == 0:
        return np.full(flat.shape[:-1], -np.inf, np.float32)
    m = flat[..., 0].copy()
    for j in range(1, flat.shape[-1]):
        m = np.where(flat[..., j] > m, flat[..., j], m)
    return m


def evaluate(cfg, x, ort=False):
    """the specification's output, or onnxruntime's with ort=True"""
    axes = cfg["axes"]
    if ort and x.size == 0 and axes:
        axes = [a for a in axes if a >= 0]
        if not axes:
            return x.copy()
    norm = reduced(cfg, axes)
    if not norm:
        return x.copy()
    keep = bool(cfg["attrs"].get("keepdims", 1))
    shape = [1 if i in norm else d for i, d in enumerate(x.shape) if keep or i not in norm]
    if cfg["op"] == "ReduceSum":
        return x.astype(np.float64).sum(axis=tuple(norm)).astype(np.float32).reshape(shape)
    if ort:
        return first_wins_max(x, norm).reshape(shape)
    if any(x.shape[d] == 0 for d in norm):     # numpy refuses the max of an empty set; the specification says -inf
        return np.full(shape, -np.inf, np.float32)
    return np.max(x, axis=tuple(norm)).reshape(shape)


def random_input(shape, k):
    x = np.asarray(rng.standard_normal(shape) * rng.choice([1, 1e3, 1e-3]), np.float32)
    flat = x.reshape(-1)
    if k % 5 == 0 and flat.size:
        for special in (np.nan, np.inf, -np.inf, -0.0):
            if rng.random() < 0.4:
                flat[rng.integers(0, flat.size)] = special
    if k % 11 == 0:                  # sums that overflow float32 but not float64
        x = np.asarray(np.full(shape, 3e38, np.float32) * np.sign(x + 0.5).astype(np.float32), np.float32)
    return x


def reference(model, x):
    try:
        return np.asarray(ReferenceEvaluator(model).run(None, {"x": x})[0], np.float32), None
    except Exception as e:
        return None, str(e)


def bucket(u):
    return "0" if u == 0 else "1" if u == 1 else "2-4" if u <= 4 else "5-16" if u <= 16 else \
        "inf (overflow or specials)" if u == float("inf") else ">16"


stats = collections.Counter()
sum_ulps = {"onnxruntime": collections.Counter(), "onnx reference": collections.Counter()}
bound = {"onnxruntime": 0.0, "onnx reference": 0.0}
known = collections.Counter()        # onnxruntime's modelled deviations, by kind
for k in range(args.count):
    cfg = random_config()
    model = make_model(cfg)
    d = os.path.join(args.work, f"r{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "r.onnx")
    onnx.save(model, path)
    c_file = os.path.join(d, "model.c")
    ok, err = compile_model(args.nullc, path, c_file)
    what = (f"{cfg['op']} {cfg['shape']} opset {cfg['opset']} axes {cfg['axes']}"
            f"{' (attribute)' if cfg['as_attr'] else ''} {cfg['attrs']}")
    if cfg["invalid"]:
        if ok:
            stats["fail"] += 1
            print(f"r{k:03d}: nullc accepted an invalid {what}")
        else:
            stats["refused"] += 1
        continue
    if not ok:
        stats["fail"] += 1
        print(f"r{k:03d}: nullc refuses {what}: {err.strip()[:150]}")
        continue
    x = random_input(cfg["shape"], k)
    want = evaluate(cfg, x)
    got_shape = output_shape(c_file)
    if got_shape != list(want.shape):
        stats["fail"] += 1
        print(f"r{k:03d}: {what}: shape nullc {got_shape}, the specification {list(want.shape)}")
        continue
    got = run_generated(args.cc, c_file, x, got_shape, d)
    if ulps(got, want) != 0:         # exact, any NaN equals any NaN, -0.0 equals +0.0
        stats["fail"] += 1
        print(f"r{k:03d}: {what}: nullc is {ulps(got, want)} ulp from the specification")
        continue
    stats["ok"] += 1

    for who in sum_ulps:
        ref, rerr = run_ort(model, x) if who == "onnxruntime" else reference(model, x)
        expect = evaluate(cfg, x, ort=True) if who == "onnxruntime" else want
        if ref is None or ref.shape != expect.shape:
            stats["fail"] += 1
            print(f"r{k:03d}: {who} on {what}: {rerr or f'shape {list(ref.shape)}'}"[:200])
            continue
        if ref.shape != want.shape or (cfg["op"] == "ReduceMax" and ulps(expect, want) != 0):
            # a known deviation: it must be reproduced exactly
            kind = "an empty input's negative axes" if ref.shape != want.shape else "a NaN not first"
            if ulps(ref, expect) != 0:
                stats["fail"] += 1
                print(f"r{k:03d}: {who} deviates on {what} other than as modelled")
            else:
                known[kind] += 1
            continue
        if cfg["op"] == "ReduceMax":
            if ulps(ref, want) != 0:
                stats["fail"] += 1
                print(f"r{k:03d}: {who}'s max differs on {what}")
            continue
        sum_ulps[who][bucket(ulps(ref, want))] += 1
        norm = reduced(cfg, cfg["axes"])
        if norm and x.size and np.isfinite(x).all() and np.abs(x).max() < 1e30:
            n = int(np.prod([cfg["shape"][a] for a in norm]))
            abs_sum = np.abs(x.astype(np.float64)).sum(axis=tuple(norm)).ravel()
            limit = n * 2.0 ** -24 * abs_sum + 2.0 ** -24 * np.abs(want.astype(np.float64)).ravel()
            e = np.abs(ref.astype(np.float64).ravel() - want.astype(np.float64).ravel())
            bound[who] = max(bound[who], float((e / np.maximum(limit, 1e-45)).max()))

print(f"{args.count} random ReduceSum and ReduceMax models: {stats['ok']} match the specification exactly, "
      f"{stats['refused']} invalid ones refused, {stats['fail']} failures")
print(f"  onnxruntime's known deviations, each reproduced exactly: {dict(known) or 'none'}")
for who in sum_ulps:
    print(f"  {who}'s sums, ulps from the float64 sum: {dict(sorted(sum_ulps[who].items()))}; on finite inputs "
          f"at most {bound[who]:.2f} of the float32 summation bound")
raise SystemExit(1 if stats["fail"] else 0)
