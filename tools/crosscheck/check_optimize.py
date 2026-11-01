"""Random graphs full of work for the optimiser: elementwise layers with
Identity and Dropout nodes between them, an Identity sometimes writing the
output, branches whose results nobody reads, and weights nothing uses. Each
graph is compiled twice, optimised (the default) and with --no-opt, and run
on the same input: the two outputs must be bit-identical, both must agree
with onnxruntime (allclose, 1e-5 relative and 1e-6 absolute), and the
optimised graph must not have more nodes."""
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, output_shape, run_generated, run_ort

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 300, "optimize")
rng = np.random.default_rng(1102)
SHAPE = [2, 3]


def random_graph():
    nodes, inits = [], []
    pool = ["x"]                                      # float32[2, 3] tensors
    for i in range(int(rng.integers(3, 6))):          # weights, broadcast [3] or full [2, 3]
        shape = [3] if rng.random() < 0.5 else SHAPE
        inits.append(numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32), f"w{i}"))
    weights = [t.name for t in inits]
    for i in range(int(rng.integers(4, 15))):
        a = str(rng.choice(pool))
        out = f"t{i}"
        op = rng.choice(["Relu", "Add", "Sub", "Mul", "Identity", "Identity", "Dropout"])
        if op in ("Add", "Sub", "Mul"):
            b = str(rng.choice(pool + weights))
            nodes.append(helper.make_node(str(op), [a, b], [out], name=f"n{i}"))
        elif op == "Dropout":
            outs = [out, f"mask{i}"] if rng.random() < 0.5 else [out]
            nodes.append(helper.make_node("Dropout", [a], outs, name=f"n{i}"))
        else:
            nodes.append(helper.make_node(str(op), [a], [out], name=f"n{i}"))
        pool.append(out)
    # the output: usually the last tensor, sometimes an earlier one, so later
    # nodes become dead; through an Identity named y, or renamed to y directly
    src = pool[-1] if rng.random() < 0.6 else str(rng.choice(pool[1:]))
    if rng.random() < 0.5:
        nodes.append(helper.make_node("Identity", [src], ["y"], name="out"))
    else:
        for n in nodes:
            n.output[:] = ["y" if o == src else o for o in n.output]
            n.input[:] = ["y" if t == src else t for t in n.input]
    if rng.random() < 0.3:
        inits.append(numpy_helper.from_array(np.ones(5, np.float32), "never_read"))
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPE)], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def nodes_and_report(path, flags):
    r = subprocess.run([args.nullc, path, *flags], capture_output=True, text=True)
    count = re.search(r": (\d+) nodes?,", r.stdout)
    report = re.search(r"optimised: (.*)", r.stdout)
    return int(count.group(1)) if count else None, report.group(1) if report else ""


stats = dict(ok=0, fail=0, before=0, after=0, changed=0)
for k in range(args.count):
    model = random_graph()
    d = os.path.join(args.work, f"o{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "m.onnx")
    onnx.save(model, path)
    x = rng.standard_normal(SHAPE).astype(np.float32)
    ref, err = run_ort(model, x)
    outs = {}
    for variant, flags in (("opt", []), ("noopt", ["--no-opt"])):
        c_file = os.path.join(d, f"{variant}.c")
        r = subprocess.run([args.nullc, path, *flags, "-o", c_file], capture_output=True, text=True)
        if r.returncode != 0:
            err = r.stderr.strip()[:150]
            break
        wd = os.path.join(d, variant)
        os.makedirs(wd, exist_ok=True)
        outs[variant] = run_generated(args.cc, c_file, x, output_shape(c_file), wd)
    if len(outs) != 2 or ref is None:
        stats["fail"] += 1
        print(f"o{k:03d}: did not compile or run ({err})")
        continue
    after, report = nodes_and_report(path, [])
    before, _ = nodes_and_report(path, ["--no-opt"])
    same = outs["opt"].tobytes() == outs["noopt"].tobytes()
    close = np.allclose(outs["opt"], ref, rtol=1e-5, atol=1e-6, equal_nan=True)
    if not same or not close or after > before:
        stats["fail"] += 1
        print(f"o{k:03d}: identical {same}, matches onnxruntime {close}, nodes {before} -> {after} ({report})")
        continue
    stats["ok"] += 1
    stats["before"] += before
    stats["after"] += after
    stats["changed"] += report != "nothing to change"

print(f"{args.count} random graphs: {stats['ok']} with optimised and unoptimised outputs bit-identical and matching "
      f"onnxruntime, {stats['fail']} failures; the optimiser changed {stats['changed']} of them, "
      f"{stats['before']} nodes in all down to {stats['after']}")
raise SystemExit(1 if stats["fail"] else 0)
