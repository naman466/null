"""Random Shape, Gather, Unsqueeze, Squeeze and Concat models, and random
chains of them built on Shape, the way exporters compute Reshape targets.
onnxruntime executes each model, so it is the arbiter: where it runs one,
nullc must accept it with the same output type and shape, and where nullc
computes the output's value at compile time (int64 data), the value must be
exactly what onnxruntime computed; where onnxruntime refuses a model, nullc
should refuse it too. nullc is asked through --dump, so float Gather and
Concat, which have no kernel yet, are checked as well."""
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args

warnings.filterwarnings("ignore")
ort.set_default_logger_severity(4)
args = parse_args(__doc__, 600, "shape_ops")
rng = np.random.default_rng(1031)
DT = {np.float32: TensorProto.FLOAT, np.int64: TensorProto.INT64}


def dims(rank, low=1, high=5):
    return [int(v) for v in rng.integers(low, high, rank)]


def ints(name, values, shape=None):
    a = np.array(values, np.int64)
    return numpy_helper.from_array(a.reshape(shape) if shape is not None else a, name)


def single_op():
    """one random node; returns (nodes, initializers, inputs as {name: array}, output type)"""
    op = rng.choice(["Shape", "Gather", "Gather", "Unsqueeze", "Squeeze", "Concat", "Concat"])
    int_data = rng.random() < 0.5            # int64 data from an initializer: its value is computable
    inits, feeds = [], {}
    out_type = TensorProto.INT64 if op == "Shape" or int_data else TensorProto.FLOAT

    def data(name, shape):
        if int_data:
            inits.append(numpy_helper.from_array(rng.integers(-9, 10, shape).astype(np.int64), name))
        else:
            feeds[name] = rng.standard_normal(shape).astype(np.float32)
        return name

    if op == "Shape":
        x = data("x", dims(int(rng.integers(0, 5))))
        attrs = {k: int(rng.integers(-5, 6)) for k in ("start", "end") if rng.random() < 0.4}
        return [helper.make_node("Shape", [x], ["y"], **attrs)], inits, feeds, out_type
    if op == "Gather":
        r = int(rng.integers(1, 4))
        shape = dims(r)
        axis = int(rng.integers(-r - (rng.random() < 0.05), r + (rng.random() < 0.05)))
        n = shape[axis % r] if -r <= axis < r else 3
        ishape = dims(int(rng.integers(0, 3)), 1, 3)
        idx = rng.integers(-n - (rng.random() < 0.1), n + (rng.random() < 0.1), ishape)
        inits.append(ints("idx", idx.ravel(), ishape))
        return [helper.make_node("Gather", [data("x", shape), "idx"], ["y"], axis=axis)], inits, feeds, out_type
    if op == "Unsqueeze":
        r = int(rng.integers(0, 4))
        k = int(rng.integers(1, 4))
        out_rank = r + k
        axes = [int(a) for a in rng.choice(np.arange(-out_rank, out_rank), k, replace=False)]
        if rng.random() < 0.05:
            axes[-1] = out_rank                                  # out of range
        if rng.random() < 0.05 and k > 1:
            axes[-1] = axes[0] - out_rank if axes[0] >= 0 else axes[0] + out_rank   # the same dim twice
        inits.append(ints("axes", axes))
        return [helper.make_node("Unsqueeze", [data("x", dims(r)), "axes"], ["y"])], inits, feeds, out_type
    if op == "Squeeze":
        r = int(rng.integers(1, 5))
        shape = [1 if rng.random() < 0.5 else int(rng.integers(2, 4)) for _ in range(r)]
        ins = [data("x", shape)]
        if rng.random() < 0.7:
            ones = [d for d in range(r) if shape[d] == 1]
            pick = list(rng.choice(ones, int(rng.integers(0, len(ones) + 1)), replace=False)) if ones else []
            if rng.random() < 0.1:
                pick.append(int(rng.integers(0, r)))             # perhaps not a 1, perhaps twice
            axes = [int(a) - r if rng.random() < 0.4 else int(a) for a in pick]
            if axes:
                inits.append(ints("axes", axes))
                ins.append("axes")
        return [helper.make_node("Squeeze", ins, ["y"])], inits, feeds, out_type
    # Concat
    r = int(rng.integers(1, 4))
    base = dims(r)
    axis = int(rng.integers(-r, r))
    names = []
    for i in range(int(rng.integers(1, 4))):
        s = list(base)
        s[axis] = int(rng.integers(0 if rng.random() < 0.1 else 1, 4))
        if rng.random() < 0.05:
            s[(axis + 1) % r] += 1 if r > 1 else 0                 # a mismatch off the axis
        names.append(data(f"x{i}", s))
    return [helper.make_node("Concat", names, ["y"], axis=axis)], inits, feeds, out_type


def chain():
    """Shape of a float input, then random steps that keep a valid int64 value"""
    x_shape = dims(int(rng.integers(1, 5)), 1, 7)
    feeds = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    nodes, inits = [helper.make_node("Shape", ["x"], ["v0"])], []
    v = np.array(x_shape, np.int64)
    for step in range(int(rng.integers(1, 5))):
        cur, nxt = f"v{step}", f"v{step + 1}"
        kind = rng.choice(["gather", "unsqueeze", "squeeze", "concat"])
        if kind == "gather" and v.ndim >= 1 and v.shape[0] > 0:
            idx = rng.integers(-v.shape[0], v.shape[0], dims(int(rng.integers(0, 2)), 1, 3))
            inits.append(ints(f"i{step}", idx.ravel(), idx.shape))
            nodes.append(helper.make_node("Gather", [cur, f"i{step}"], [nxt], axis=0))
            v = v[idx]
        elif kind == "squeeze" and 1 in v.shape:
            d = v.shape.index(1)
            inits.append(ints(f"a{step}", [d]))
            nodes.append(helper.make_node("Squeeze", [cur, f"a{step}"], [nxt]))
            v = np.squeeze(v, d)
        elif kind == "concat" and v.ndim == 1:
            extra = rng.integers(-3, 30, int(rng.integers(1, 3)))
            inits.append(ints(f"c{step}", extra))
            nodes.append(helper.make_node("Concat", [f"c{step}", cur] if rng.random() < 0.5 else [cur, f"c{step}"],
                                          [nxt], axis=0))
            v = np.concatenate([extra, v]) if nodes[-1].input[0] == f"c{step}" else np.concatenate([v, extra])
        else:
            a = int(rng.integers(-v.ndim - 1, v.ndim + 1))
            inits.append(ints(f"a{step}", [a]))
            nodes.append(helper.make_node("Unsqueeze", [cur, f"a{step}"], [nxt]))
            v = np.expand_dims(v, a)
    nodes[-1].output[0] = "y"
    return nodes, inits, feeds, TensorProto.INT64


def make_model(nodes, inits, feeds, out_type):
    inputs = [helper.make_tensor_value_info(n, DT[a.dtype.type], list(a.shape)) for n, a in feeds.items()]
    g = helper.make_graph(nodes, "g", inputs, [helper.make_tensor_value_info("y", out_type, None)], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def ask_nullc(path):
    """(dtype, shape, value or None) of y from nullc --dump, or (None, error)"""
    r = subprocess.run([args.nullc, path, "--dump"], capture_output=True, text=True)
    if "shape inference stopped" in r.stderr or r.returncode != 0:
        return None, r.stderr.strip().split("stopped: ")[-1][:150]
    m = re.search(r"-> y: (\w+)\[([^\]]*)\]( = (\[[^\]]*\]|-?\d+))?", r.stdout)
    dt, shape = m.group(1), [int(d) for d in m.group(2).split(",") if d.strip()]
    value = None
    if m.group(4):
        value = [int(t) for t in m.group(4).strip("[]").split(",") if t.strip()]
    return (dt, shape, value), None


stats = {"ok": 0, "values": 0, "rejected": 0, "fail": 0}
lenient = []
for k in range(args.count):
    nodes, inits, feeds, out_type = chain() if k % 3 == 2 else single_op()
    model = make_model(nodes, inits, feeds, out_type)
    d = os.path.join(args.work, f"s{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "m.onnx")
    onnx.save(model, path)
    try:
        ref = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
        ref_err = None
    except Exception as e:
        ref, ref_err = None, str(e).split("Status Message:")[-1].strip()[:120]
    got, err = ask_nullc(path)
    what = f"{nodes[0].op_type}" + (f" chain of {len(nodes)}" if k % 3 == 2 else "")
    if ref is None:
        if got is None:
            stats["rejected"] += 1
        else:
            lenient.append(f"s{k:03d} {what}: onnxruntime refuses ({ref_err}), nullc gives {got[0]}{got[1]}")
        continue
    if got is None:
        stats["fail"] += 1
        print(f"s{k:03d} {what}: onnxruntime runs it, nullc refuses: {err}")
        continue
    dt, shape, value = got
    want_dt = {np.dtype(np.float32): "float32", np.dtype(np.int64): "int64"}[ref.dtype]
    if dt != want_dt or shape != list(ref.shape):
        stats["fail"] += 1
        print(f"s{k:03d} {what}: nullc {dt}{shape}, onnxruntime {want_dt}{list(ref.shape)}")
        continue
    computable = ref.dtype == np.int64 and ref.size > 0
    if computable and value is None:
        stats["fail"] += 1
        print(f"s{k:03d} {what}: the int64 value was not computed")
        continue
    if value is not None and value != [int(v) for v in ref.ravel()]:
        stats["fail"] += 1
        print(f"s{k:03d} {what}: nullc computed {value}, onnxruntime {ref.ravel().tolist()}")
        continue
    stats["ok"] += 1
    stats["values"] += value is not None

print(f"{args.count} random models: {stats['ok']} agree with onnxruntime ({stats['values']} of them with a value "
      f"computed at compile time, every one exact), {stats['rejected']} refused by both, {len(lenient)} accepted by "
      f"nullc but refused by onnxruntime, {stats['fail']} failures")
for line in lenient[:10]:
    print("  " + line)
raise SystemExit(1 if stats["fail"] or lenient else 0)
