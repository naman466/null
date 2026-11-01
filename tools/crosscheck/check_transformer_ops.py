"""Random single LayerNormalization, Transpose and Gather (float data,
constant indices) models, compiled by nullc and run.

Transpose and Gather only move data, so the output must be bit-identical to
numpy and to onnxruntime. LayerNormalization is checked against the
specification's formula computed in float64 and rounded once, which the
runtime also computes in double: at most 1 ulp apart (numpy sums pairwise,
the runtime in order), and the count of exact matches is reported; against
onnxruntime, which computes in float32, within 1e-5. Models onnxruntime
refuses must be refused by nullc too; a scale that broadcasts within the
normalized dims is a documented limitation and only has to be refused
clearly."""
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort, ulps

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 600, "transformer_ops")
rng = np.random.default_rng(1101)


def floats(name, shape):
    return numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32), name)


def random_case(k):
    kind = ["layernorm", "transpose", "gather"][k % 3]
    rank = int(rng.integers(1, 5)) if kind != "transpose" else int(rng.integers(0, 6))
    shape = [int(d) for d in rng.integers(1, 7 if kind != "transpose" else 5, rank)]
    inits, attrs, ins, case = [], {}, ["x"], dict(kind=kind, shape=shape)
    if kind == "layernorm":
        axis = int(rng.integers(-rank, rank))
        normalized = shape[axis % rank:]
        scale_shape = list(normalized)
        r = rng.random()
        if r < 0.05:
            scale_shape = [d + 1 for d in normalized]            # does not broadcast: invalid
        elif r < 0.1:
            scale_shape = [1] * len(normalized)                  # broadcasts: valid, not supported yet
        inits.append(floats("scale", scale_shape))
        ins.append("scale")
        if rng.random() < 0.7:
            inits.append(floats("bias", normalized))
            ins.append("bias")
        eps = float(rng.choice([1e-5, 1e-3, 0.1, 0.0]))
        attrs = dict(axis=axis, epsilon=eps)
        case.update(axis=axis, eps=eps, broadcast=r < 0.1 and r >= 0.05, invalid=r < 0.05)
    elif kind == "transpose":
        perm = [int(p) for p in rng.permutation(rank)]
        if rng.random() < 0.8 and perm:          # an empty perm cannot be written; rank 0 uses the default
            attrs = dict(perm=perm)
    else:
        axis = int(rng.integers(-rank, rank))
        n = shape[axis % rank]
        ishape = [int(d) for d in rng.integers(1, 4, int(rng.integers(0, 3)))]
        idx = rng.integers(-n, n, ishape).astype(np.int64)
        inits.append(numpy_helper.from_array(idx, "idx"))
        ins.append("idx")
        attrs = dict(axis=axis)
        case.update(axis=axis)
    node = helper.make_node({"layernorm": "LayerNormalization", "transpose": "Transpose", "gather": "Gather"}[kind],
                            ins, ["y"], **attrs)
    g = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model, case


def layernorm_f64(x, model, case):
    init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in model.graph.initializer}
    axes = tuple(range(case["axis"] % x.ndim, x.ndim))
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=axes, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=axes, keepdims=True)
    y = (x64 - mean) / np.sqrt(var + float(np.float32(case["eps"]))) * init["scale"]
    if "bias" in init:
        y = y + init["bias"]
    return y.astype(np.float32)


stats = dict(ok=0, rejected=0, limitation=0, fail=0)
ln = dict(exact=0, one_ulp=0, worst_ort=0.0)
for k in range(args.count):
    model, case = random_case(k)
    d = os.path.join(args.work, f"t{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "m.onnx")
    onnx.save(model, path)
    x = rng.standard_normal(case["shape"]).astype(np.float32) * np.float32(rng.choice([1, 100, 0.01]))
    if case["kind"] == "layernorm" and k % 13 == 0:
        x[...] = np.float32(3)                                     # rows with no spread at all
    x = np.asarray(x, np.float32)
    ref, ref_err = run_ort(model, x)
    c_file = os.path.join(d, "model.c")
    ok, err = compile_model(args.nullc, path, c_file)
    if ref is None:
        if not ok:
            stats["rejected"] += 1
        else:
            stats["fail"] += 1
            print(f"t{k:03d} {case['kind']}: onnxruntime refuses ({ref_err[-80:]}), nullc compiles it")
        continue
    if case.get("broadcast"):
        if not ok and "broadcasts within the normalized dims is not supported yet" in err:
            stats["limitation"] += 1
        else:
            stats["fail"] += 1
            print(f"t{k:03d}: a broadcast scale was not refused clearly: {ok} {err.strip()[:120]}")
        continue
    if not ok:
        stats["fail"] += 1
        print(f"t{k:03d} {case['kind']}: onnxruntime runs it, nullc refuses: {err.strip()[:150]}")
        continue
    shape = output_shape(c_file)
    if shape != list(ref.shape):
        stats["fail"] += 1
        print(f"t{k:03d} {case['kind']}: shape nullc {shape}, onnxruntime {list(ref.shape)}")
        continue
    got = run_generated(args.cc, c_file, x, shape, d)
    if case["kind"] == "layernorm":
        want = layernorm_f64(x, model, case)
        u = ulps(got, want)
        both_nan = np.isnan(got) & np.isnan(ref)
        err_ort = float(np.nanmax(np.abs(np.where(both_nan, 0, got.astype(np.float64) - ref)))) if got.size else 0.0
        ln["worst_ort"] = max(ln["worst_ort"], err_ort)
        if u > 1 or not err_ort <= 1e-5 * max(1.0, float(np.nanmax(np.abs(ref))) if ref.size else 1.0):
            stats["fail"] += 1
            print(f"t{k:03d} layernorm: {u} ulp from float64, {err_ort:.1e} from onnxruntime, {case}")
            continue
        ln["exact" if u == 0 else "one_ulp"] += 1
    else:
        if case["kind"] == "transpose":
            perm = [a.ints for a in model.graph.node[0].attribute if a.name == "perm"]
            want = np.transpose(x, list(perm[0]) if perm else None)
        else:
            idx = numpy_helper.to_array(model.graph.initializer[0])
            want = np.take(x, idx, axis=case["axis"])
        if got.tobytes() != want.tobytes() or got.tobytes() != np.ascontiguousarray(ref).tobytes():
            stats["fail"] += 1
            print(f"t{k:03d} {case['kind']}: values differ from numpy or onnxruntime, {case}")
            continue
    stats["ok"] += 1

print(f"{args.count} random models: {stats['ok']} match, {stats['rejected']} refused by both, "
      f"{stats['limitation']} broadcast scales refused as not supported yet, {stats['fail']} failures")
print(f"  layernorm: {ln['exact']} bit-identical to the float64 formula, {ln['one_ulp']} 1 ulp away; "
      f"worst difference from onnxruntime {ln['worst_ort']:.1e}")
raise SystemExit(1 if stats["fail"] else 0)
