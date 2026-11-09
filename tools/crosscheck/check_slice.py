"""Random single-node Slice and Split models. Slice: ranks 1 to 5 (some dims
empty), a random subset of the axes in any order (negative ones included),
starts and ends anywhere from far before the dim to far after it (the int64
extremes exporters write included), steps positive and negative. Split: sizes
as an input (some zero), equal pieces, or opset 18's num_outputs with a
smaller last piece; one output, chosen at random, is the graph's, the others
are not read.

The arbiter is the specification: an emulation of its formulas (negative
starts and ends count from the end, then are clamped to [0, dim] for a
positive step and to [0, dim - 1] and [-1, dim - 1] for a negative one), and
nullc must be bit-identical to it (NaN and -0.0 included). onnxruntime and
numpy's slicing are compared too, and where they deviate from the
specification the case is counted, not failed: numpy, stepping backwards from
a start before the dim, takes nothing where the specification clamps the
start to 0. onnxruntime deviates in two ways, each modelled here so that any
other deviation fails: stepping backwards to an end of INT64_MAX it takes
the whole dim where the specification clamps the end to dim - 1, and it
refuses a num_outputs Split whose last piece is empty, which the
specification's formula, onnx's shape inference and its reference evaluator
all allow. Some models are
invalid (a step of 0, an axis twice or out of range, sizes that do not add
up, num_outputs that cannot fit) and nullc must refuse them."""
import collections
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 400, "slice")
rng = np.random.default_rng(1110)
I64_MIN, I64_MAX = -2**63, 2**63 - 1


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


def ints(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name)


def spec_indices(dim, start, end, step, ort=False):
    """the indices the ONNX Slice specification takes along one dim. ort=True
    models onnxruntime's one deviation: stepping backwards, an end of INT64_MAX
    means "past the first element" rather than being clamped to dim - 1"""
    if ort and step < 0 and end == I64_MAX:
        end = -dim - 1
    if start < 0:
        start += dim
    if end < 0:
        end += dim
    if step > 0:
        start, end = min(max(start, 0), dim), min(max(end, 0), dim)
        return list(range(start, end, step))
    start, end = min(max(start, 0), dim - 1), min(max(end, -1), dim - 1)
    return list(range(start, end, step)) if dim > 0 else []


def take(x, per_dim):
    """x restricted to the given indices along every dim"""
    for d, idx in enumerate(per_dim):
        x = np.take(x, np.asarray(idx, np.int64), axis=d)
    return x


def position(dim):
    """a start or end: inside the dim, just outside, far outside, or an int64 extreme"""
    r = rng.random()
    if r < 0.6:
        return int(rng.integers(-dim - 1, dim + 2))
    if r < 0.8:
        return int(rng.choice([-1000, 1000]))
    return int(rng.choice([I64_MIN, I64_MAX]))


def slice_model():
    rank = int(rng.integers(1, 6))
    shape = [int(rng.integers(0 if rng.random() < 0.1 else 1, 6)) for _ in range(rank)]
    n = int(rng.integers(1, rank + 1))
    dims = [int(d) for d in rng.permutation(rank)[:n]]
    axes = [d - rank if rng.random() < 0.3 else d for d in dims]
    starts = [position(shape[d]) for d in dims]
    ends = [position(shape[d]) for d in dims]
    steps = [int(rng.choice([1, 1, 1, 2, 3, -1, -1, -2, -3, I64_MAX, I64_MIN])) for _ in dims]
    invalid = False
    if rng.random() < 0.1:
        what = rng.integers(0, 3)
        if what == 0:
            steps[0] = 0
        elif what == 1:
            axes.append(axes[0])
            starts.append(0)
            ends.append(1)
            steps.append(1)
        else:
            axes[0] = int(rng.choice([rank, -rank - 1]))
        invalid = True
    node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"], name="s")
    inits = [ints("starts", starts), ints("ends", ends), ints("axes", axes), ints("steps", steps)]

    def spec(x, ort=False):
        per_dim = [list(range(d)) for d in shape]
        for a, s, e, st in zip(axes, starts, ends, steps):
            per_dim[a % rank] = spec_indices(shape[a % rank], s, e, st, ort)
        return take(x, per_dim)

    def py(x):
        index = [slice(None)] * rank
        for a, s, e, st in zip(axes, starts, ends, steps):
            index[a % rank] = slice(s, e, st)
        return x[tuple(index)]
    what = f"Slice {shape} starts {starts} ends {ends} axes {axes} steps {steps}"
    return shape, node, inits, 13, spec, py, lambda x: spec(x, True), invalid, what


def split_model():
    rank = int(rng.integers(1, 5))
    shape = [int(rng.integers(1, 6)) for _ in range(rank)]
    axis = int(rng.integers(-rank, rank))
    a = axis % rank
    form = str(rng.choice(["input", "equal", "num_outputs"]))
    invalid = False
    if form == "equal":
        pieces = int(rng.integers(1, 4))
        shape[a] = pieces * int(rng.integers(0, 3))
        sizes = [shape[a] // pieces] * pieces
    elif form == "num_outputs":
        pieces = int(rng.integers(1, 4))
        shape[a] = int(rng.integers(max(1, pieces - 1), 3 * pieces + 1))
        chunk = -(-shape[a] // pieces)
        sizes = [chunk] * (pieces - 1) + [shape[a] - chunk * (pieces - 1)]
        invalid = sizes[-1] < 0
    else:
        pieces = int(rng.integers(1, 5))
        cuts = sorted(int(v) for v in rng.integers(0, shape[a] + 1, pieces - 1))
        sizes = [e - s for s, e in zip([0] + cuts, cuts + [shape[a]])]
        if rng.random() < 0.1:
            sizes[int(rng.integers(0, pieces))] += 1
            invalid = True
    outs = [f"o{k}" for k in range(len(sizes))]
    chosen = int(rng.integers(0, len(sizes)))
    outs[chosen] = "y"
    inits, opset, attrs, inputs = [], 13, dict(axis=axis), ["x"]
    if form == "input":
        inputs.append("sizes")
        inits.append(ints("sizes", sizes))
    elif form == "num_outputs":
        attrs["num_outputs"] = len(sizes)
        opset = 18
    node = helper.make_node("Split", inputs, outs, name="s", **attrs)

    def spec(x):
        begin = sum(sizes[:chosen])
        per_dim = [list(range(d)) for d in shape]
        per_dim[a] = list(range(begin, begin + max(sizes[chosen], 0)))
        return take(x, per_dim)
    what = f"Split {shape} axis {axis} {form} {sizes} output {chosen}"
    # onnxruntime refuses num_outputs whose last piece is empty, which the specification's
    # formula, onnx's shape inference and its reference evaluator all allow
    ort_refuses = form == "num_outputs" and sizes[-1] == 0
    return shape, node, inits, opset, spec, spec, (lambda x: None) if ort_refuses else spec, invalid, what


stats = collections.Counter()
deviations = collections.Counter()
for k in range(args.count):
    shape, node, inits, opset, spec, py, ort_model, invalid, what = slice_model() if k % 2 == 0 else split_model()
    g = helper.make_graph([node], "slice", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    d = os.path.join(args.work, f"s{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "m.onnx")
    onnx.save(model, path)
    c_file = os.path.join(d, "model.c")
    ok, err = compile_model(args.nullc, path, c_file)
    if invalid:
        if ok:
            stats["fail"] += 1
            print(f"s{k:03d}: nullc accepted an invalid {what}")
        else:
            stats["refused"] += 1
        continue
    if not ok:
        stats["fail"] += 1
        print(f"s{k:03d}: nullc refuses {what}: {err.strip()[:150]}")
        continue
    x = random_input(shape)
    want = spec(x)
    got = run_generated(args.cc, c_file, x, output_shape(c_file), d)
    if not bits_equal(got, want):
        stats["fail"] += 1
        print(f"s{k:03d}: {what}: nullc {got.shape}, the specification {want.shape}")
        continue
    stats["ok"] += 1
    ref, rerr = run_ort(model, x)
    if ref is None or not bits_equal(ref, want):
        # every deviation must be one of the known two, reproduced exactly
        known = ort_model(x)
        explained = (ref is None and known is None) or (ref is not None and known is not None and bits_equal(ref, known))
        deviations["onnxruntime, known" if explained else "onnxruntime, unexplained"] += 1
        if not explained:
            stats["fail"] += 1
            print(f"s{k:03d}: onnxruntime deviates on {what} in an unexpected way: "
                  f"{None if ref is None else ref.shape} vs {want.shape} {rerr or ''}".rstrip())
    if not bits_equal(py(x), want):
        deviations["numpy"] += 1

print(f"{args.count} random Slice and Split models: {stats['ok']} bit-identical to the specification, "
      f"{stats['refused']} invalid ones refused, {stats['fail']} failures; onnxruntime deviates on "
      f"{deviations['onnxruntime, known']} in the two known ways, numpy's slicing on {deviations['numpy']}")
raise SystemExit(1 if stats["fail"] else 0)
