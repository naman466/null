"""Random single-node float Concat models: ranks 1 to 5, any axis (negative
ones included), 1 to 5 operands, each the graph input x again or an
initializer of matching shape, some of them empty along the axis. Concat
only moves data, so nullc must be bit-identical (NaN and -0.0 included) to
numpy's concatenate and to onnxruntime, and the output shape must be numpy's.
A tenth of the models are invalid (an operand whose size differs off the
axis) and nullc must refuse them."""
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 300, "concat")
rng = np.random.default_rng(1109)


def random_input(shape):
    x = np.asarray(rng.standard_normal(shape), np.float32)
    flat = x.reshape(-1)
    for special in (np.nan, np.inf, -0.0):
        if flat.size and rng.random() < 0.3:
            flat[rng.integers(0, flat.size)] = special
    return x


def bits_equal(a, b):
    return a.shape == b.shape and np.array_equal(np.asarray(a, np.float32).view(np.int32),
                                                 np.asarray(b, np.float32).view(np.int32))


stats = dict(ok=0, fail=0, refused=0)
for k in range(args.count):
    rank = int(rng.integers(1, 6))
    shape = [int(rng.integers(1, 5)) for _ in range(rank)]
    axis = int(rng.integers(-rank, rank))
    a = axis % rank
    names, inits, parts = ["x"], [], []
    for j in range(int(rng.integers(0, 5))):
        if rng.random() < 0.3:
            names.append("x")
            continue
        s = list(shape)
        s[a] = int(rng.choice([0, 1, 2, 3]))
        v = random_input(s)
        inits.append(numpy_helper.from_array(v, f"w{j}"))
        names.append(f"w{j}")
        parts.append((f"w{j}", v))
    invalid = False
    if rank > 1 and rng.random() < 0.1:
        s = list(shape)
        off = (a + 1) % rank
        s[off] += 1
        inits.append(numpy_helper.from_array(random_input(s), "bad"))
        names.insert(int(rng.integers(1, len(names) + 1)), "bad")
        invalid = True
    if rng.random() < 0.5:
        names = [str(n) for n in rng.permutation(names)]                 # x not always first
    g = helper.make_graph([helper.make_node("Concat", names, ["y"], name="cat", axis=axis)], "cat",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    d = os.path.join(args.work, f"c{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "m.onnx")
    onnx.save(model, path)
    c_file = os.path.join(d, "model.c")
    ok, err = compile_model(args.nullc, path, c_file)
    if invalid:
        if ok:
            stats["fail"] += 1
            print(f"c{k:03d}: nullc accepted operands of different sizes off the axis")
        else:
            stats["refused"] += 1
        continue
    if not ok:
        stats["fail"] += 1
        print(f"c{k:03d}: nullc refuses {shape} axis {axis}: {err.strip()[:150]}")
        continue
    x = random_input(shape)
    values = dict(parts, x=x)
    want = np.concatenate([values[n] for n in names], axis=a)
    got = run_generated(args.cc, c_file, x, output_shape(c_file), d)
    ref, rerr = run_ort(model, x)
    if not bits_equal(got, want) or ref is None or not bits_equal(got, ref):
        stats["fail"] += 1
        print(f"c{k:03d}: {shape} axis {axis} {names}: differs from numpy or onnxruntime ({rerr or ''})")
        continue
    stats["ok"] += 1

print(f"{args.count} random Concat models: {stats['ok']} bit-identical to numpy and onnxruntime, "
      f"{stats['refused']} invalid ones refused, {stats['fail']} failures")
raise SystemExit(1 if stats["fail"] else 0)
