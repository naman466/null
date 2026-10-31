"""Random single-node ReduceMean (opset 13 axes attribute, opset 18 axes input)
and GlobalAveragePool models. The arbiter is the specification's definition:
the mean over the axes, computed in float64 and rounded to float32 once, which
nullc must match bit for bit. onnxruntime and the onnx reference evaluator add
in float32, so they are reported in ulps and against the float32 summation
error bound n * 2^-24 * mean|x|; output shapes are compared with onnx strict
shape inference."""
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort, ulps

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 400, "reduce")
rng = np.random.default_rng(2411)


def random_config():
    op = "GlobalAveragePool" if rng.random() < 0.25 else "ReduceMean"
    rank = int(rng.integers(3, 6)) if op == "GlobalAveragePool" else int(rng.integers(0, 6))
    shape = [int(rng.integers(1, 7)) for _ in range(rank)]
    cfg = dict(op=op, shape=shape, opset=18, attrs={}, axes=None)
    if op == "ReduceMean":
        cfg["opset"] = int(rng.choice([13, 18]))
        form = rng.choice(["axes", "axes", "axes", "none", "empty"])
        if form == "axes" and rank > 0:
            k = int(rng.integers(1, rank + 1))
            axes = [int(a) for a in rng.choice(rank, k, replace=False)]
            axes = [a - rank if rng.random() < 0.4 else a for a in axes]
            if rng.random() < 0.04:
                axes.append(axes[0])                                     # repeated: invalid
            if rng.random() < 0.04:
                axes.append(rank if rng.random() < 0.5 else -rank - 1)   # out of range: invalid
            cfg["axes"] = axes
        elif form == "empty":
            cfg["axes"] = []
        if rng.random() < 0.7:
            cfg["attrs"]["keepdims"] = int(rng.integers(0, 2))
        if cfg["opset"] == 18 and rng.random() < 0.3:
            cfg["attrs"]["noop_with_empty_axes"] = int(rng.integers(0, 2))
    return cfg


def make_model(cfg):
    inits, inputs, attrs = [], ["x"], dict(cfg["attrs"])
    if cfg["axes"] is not None:
        if cfg["opset"] == 18:
            inits.append(numpy_helper.from_array(np.array(cfg["axes"], np.int64), "axes"))
            inputs.append("axes")
        elif cfg["axes"]:            # an empty attribute means the same as none
            attrs["axes"] = cfg["axes"]
    node = helper.make_node(cfg["op"], inputs, ["y"], name="mean", **attrs)
    g = helper.make_graph([node], "reduce", [helper.make_tensor_value_info("x", TensorProto.FLOAT, cfg["shape"])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], initializer=inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", cfg["opset"])])
    model.ir_version = 8
    return model


def spec(cfg):
    """the reduced dims and output shape from the specification, None if invalid"""
    shape, rank = cfg["shape"], len(cfg["shape"])
    if cfg["op"] == "GlobalAveragePool":
        return list(range(2, rank)), shape[:2] + [1] * (rank - 2)
    axes = cfg["axes"] or []
    if any(not -rank <= a < rank for a in axes):
        return None
    norm = [a % rank for a in axes] if rank else []
    if len(set(norm)) != len(norm):
        return None
    if not norm:
        norm = [] if cfg["attrs"].get("noop_with_empty_axes", 0) else list(range(rank))
    keep = cfg["attrs"].get("keepdims", 1)
    out = [1 if i in norm else d for i, d in enumerate(shape) if keep or i not in norm]
    return norm, out


stats = dict(ok=0, rejected=0, fail=0)
others = {"ort": [], "onnx reference": []}          # ulps from the float64 mean, or why not comparable
bound = {"ort": 0.0, "onnx reference": 0.0}         # worst error as a fraction of the float32 summation bound
shape_dev = []
for k in range(args.count):
    cfg = random_config()
    model = make_model(cfg)
    d = os.path.join(args.work, f"red{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "red.onnx")
    onnx.save(model, path)
    shp = tuple(cfg["shape"])
    x = np.asarray(rng.standard_normal(shp) * rng.choice([1, 1e3, 1e-3]), np.float32)
    if k % 9 == 0 and x.size > 1:                     # specials propagate like a sum
        x.ravel()[rng.integers(0, x.size)] = rng.choice([np.inf, -np.inf, np.nan])
    if k % 11 == 0:                                   # sums near the top of the float range
        x = np.asarray(np.full(shp, 3e38, np.float32) * np.sign(x + 0.5).astype(np.float32), np.float32)
    geom = spec(cfg)
    c_file = os.path.join(d, "model.c")
    ok, err = compile_model(args.nullc, path, c_file)
    if geom is None:
        if not ok:
            stats["rejected"] += 1
        else:
            stats["fail"] += 1
            print(f"red{k:03d}: invalid by the spec but nullc compiled it {cfg}")
        if run_ort(model, x)[0] is not None:
            print(f"  red{k:03d}: (onnxruntime accepts the invalid axes {cfg['axes']})")
        continue
    if not ok:
        stats["fail"] += 1
        print(f"red{k:03d}: nullc rejects a valid model: {err.strip()[:160]} {cfg}")
        continue
    axes, out_shape = geom
    want = x.astype(np.float64).mean(axis=tuple(axes), keepdims=True) if axes else x.astype(np.float64)
    want = want.astype(np.float32).reshape(out_shape)
    got_shape = output_shape(c_file)
    if got_shape != out_shape:
        stats["fail"] += 1
        print(f"red{k:03d}: shape nullc {got_shape} spec {out_shape} {cfg}")
        continue
    got = run_generated(args.cc, c_file, x, out_shape, d)
    if ulps(got, want) != 0:      # bit for bit, except that any NaN equals any NaN
        stats["fail"] += 1
        print(f"red{k:03d}: nullc differs from the float64 mean by {ulps(got, want)} ulp {cfg}")
        continue
    stats["ok"] += 1

    finite = bool(np.isfinite(x).all() and (x.size == 0 or np.abs(x).max() < 1e30)) and bool(axes)
    if finite:
        n = int(np.prod([cfg["shape"][a] for a in axes]))
        mean_abs = (np.abs(x.astype(np.float64)).sum(axis=tuple(axes)) / n).ravel()
        limit = n * 2.0 ** -24 * mean_abs + 2.0 ** -24 * np.abs(want.astype(np.float64)).ravel()
    for who in others:
        if who == "ort":
            ref, err = run_ort(model, x)
        else:
            try:
                ref, err = ReferenceEvaluator(model).run(None, {"x": x})[0], None
            except Exception as e:
                ref, err = None, str(e)
        if ref is None:
            others[who].append("rejects: " + err.split("Status Message:")[-1].strip()[:60])
            continue
        ref = np.asarray(ref, np.float32)
        if list(ref.shape) != out_shape:
            others[who].append(f"shape {list(ref.shape)}")
            continue
        others[who].append(ulps(ref, want))
        if finite:
            e = np.abs(ref.astype(np.float64).ravel() - want.astype(np.float64).ravel())
            bound[who] = max(bound[who], float((e / np.maximum(limit, 1e-45)).max()))
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        si = [dd.dim_value for dd in inferred.graph.output[0].type.tensor_type.shape.dim]
        if si != out_shape:
            shape_dev.append((k, si, out_shape))
    except Exception as e:
        shape_dev.append((k, "error", str(e)[:60]))

print(f"{args.count} random reductions: {stats['ok']} valid models bit-identical to the float64 mean, "
      f"{stats['rejected']} invalid ones rejected, {stats['fail']} failures")
for who, items in others.items():
    hist = {}
    for v in items:
        key = v if isinstance(v, str) else "0" if v == 0 else "1" if v == 1 else "2-4" if v <= 4 else \
            "5-16" if v <= 16 else "inf (overflow or specials)" if v == float("inf") else ">16"
        hist[key] = hist.get(key, 0) + 1
    print(f"  {who} vs the float64 mean, in ulps: {hist}; on finite inputs at most "
          f"{bound[who]:.2f} of the float32 summation bound")
print(f"  onnx shape inference differs on {len(shape_dev)}: {shape_dev[:4]}")
raise SystemExit(1 if stats["fail"] else 0)
