"""Random single-MaxPool models. The arbiter is an emulation of the ONNX
specification's formulas, because onnxruntime, the onnx reference evaluator
and onnx shape inference each deviate from it in corners of pooling. A maximum
does no arithmetic, so nullc must match the emulation bit for bit; where each
reference differs is counted and reported."""
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto, shape_inference
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 500, "maxpool")
rng = np.random.default_rng(2410)


def random_config():
    kh, kw = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    attrs = dict(kernel_shape=[kh, kw], strides=[int(rng.integers(1, 4)), int(rng.integers(1, 4))])
    if rng.random() < 0.3:
        attrs["dilations"] = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
    mode = rng.choice(["NOTSET", "NOTSET", "NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER", "OMIT"])
    if mode == "NOTSET":
        attrs["pads"] = [int(rng.integers(0, kh)), int(rng.integers(0, kw)), int(rng.integers(0, kh)), int(rng.integers(0, kw))]
        if rng.random() < 0.05:   # sometimes too much padding, which must be refused
            attrs["pads"][0] = kh
    elif mode != "OMIT":
        attrs["auto_pad"] = str(mode)
    if rng.random() < 0.4:
        attrs["ceil_mode"] = 1
    return dict(n=int(rng.integers(1, 3)), c=int(rng.integers(1, 4)), h=int(rng.integers(1, 12)),
                w=int(rng.integers(1, 12)), attrs=attrs)


def make_model(cfg):
    node = helper.make_node("MaxPool", ["x"], ["y"], name="pool", **cfg["attrs"])
    g = helper.make_graph([node], "pool",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [cfg["n"], cfg["c"], cfg["h"], cfg["w"]])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def spec(cfg):
    """output size and padding straight from the onnx MaxPool specification, None if invalid"""
    a = cfg["attrs"]
    k = a["kernel_shape"]; st = a["strides"]; dl = a.get("dilations", [1, 1])
    mode = a.get("auto_pad", "NOTSET")
    ceil_mode = a.get("ceil_mode", 0)
    ins = [cfg["h"], cfg["w"]]
    begin, end, out = [0, 0], [0, 0], [0, 0]
    for i in range(2):
        eff = (k[i] - 1) * dl[i] + 1
        if mode == "NOTSET":
            pads = a.get("pads", [0, 0, 0, 0])
            begin[i], end[i] = pads[i], pads[i + 2]
            span = ins[i] + begin[i] + end[i]
            if span < eff:
                return None
            if ceil_mode:
                o = -(-(span - eff) // st[i]) + 1
                if (o - 1) * st[i] >= ins[i] + begin[i]:      # window starting in the end padding
                    o -= 1
            else:
                o = (span - eff) // st[i] + 1
        elif mode == "VALID":
            if ins[i] < eff:
                return None
            o = -(-(ins[i] - eff + 1) // st[i])
        else:
            o = -(-ins[i] // st[i])
            total = max(0, (o - 1) * st[i] + eff - ins[i])
            begin[i] = total // 2 if mode == "SAME_UPPER" else total - total // 2
            end[i] = total - begin[i]
        out[i] = o
    return out, begin, dl


def emulate(x, cfg, geom):
    (oh_n, ow_n), (pt, pl), dl = geom
    a = cfg["attrs"]
    kh, kw = a["kernel_shape"]; sh, sw = a["strides"]
    y = np.full((cfg["n"], cfg["c"], oh_n, ow_n), -np.inf, np.float32)
    for oh in range(oh_n):
        for ow in range(ow_n):
            for r in range(kh):
                ih = oh * sh - pt + r * dl[0]
                if not 0 <= ih < cfg["h"]:
                    continue
                for s_ in range(kw):
                    iw = ow * sw - pl + s_ * dl[1]
                    if 0 <= iw < cfg["w"]:
                        y[:, :, oh, ow] = np.maximum(y[:, :, oh, ow], x[:, :, ih, iw])
    return y


stats = dict(ok=0, rejected=0, fail=0)
deviations = {"ort": [], "onnx reference": [], "onnx shape inference": []}
for k in range(args.count):
    cfg = random_config()
    model = make_model(cfg)
    d = os.path.join(args.work, f"pool{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "pool.onnx")
    onnx.save(model, path)
    shp = (cfg["n"], cfg["c"], cfg["h"], cfg["w"])
    # some inputs with many ties, some with -inf, so equal maxima and edge values are exercised
    x = rng.standard_normal(shp).astype(np.float32)
    if k % 3 == 0:
        x = np.round(x).astype(np.float32) + np.float32(0)      # + 0 turns -0.0 into +0.0
    if k % 7 == 0:
        x[rng.random(shp) < 0.2] = -np.inf
    a = cfg["attrs"]
    too_much_pad = "pads" in a and (a["pads"][0] >= a["kernel_shape"][0] or a["pads"][2] >= a["kernel_shape"][0]
                                    or a["pads"][1] >= a["kernel_shape"][1] or a["pads"][3] >= a["kernel_shape"][1])
    geom = None if too_much_pad else spec(cfg)
    c_file = os.path.join(d, "model.c")
    ok, err = compile_model(args.nullc, path, c_file)
    if geom is None or min(geom[0]) < 1:
        if not ok:
            stats["rejected"] += 1
        else:
            stats["fail"] += 1
            print(f"pool{k:03d}: invalid by the spec but nullc compiled it cfg {cfg}")
        continue
    if not ok:
        stats["fail"] += 1
        print(f"pool{k:03d}: nullc rejects a valid model: {err.strip()[:140]} cfg {cfg}")
        continue
    want = emulate(x, cfg, geom)
    got_shape = output_shape(c_file)
    if got_shape != list(want.shape):
        stats["fail"] += 1
        print(f"pool{k:03d}: shape nullc {got_shape} spec {list(want.shape)} cfg {cfg}")
        continue
    got = run_generated(args.cc, c_file, x, got_shape, d)
    if got.tobytes() != want.tobytes():
        stats["fail"] += 1
        print(f"pool{k:03d}: {int((got != want).sum())} values differ from the spec emulation cfg {cfg}")
        continue
    stats["ok"] += 1
    # where do the other implementations stand on this valid model?
    ref, err = run_ort(model, x)
    if ref is None or ref.shape != want.shape or ref.tobytes() != want.tobytes():
        deviations["ort"].append((k, a, "rejects: " + err.split("Status Message:")[-1].strip()[:50] if ref is None
                                  else f"shape {list(ref.shape)}" if ref.shape != want.shape else "values"))
    try:
        rv = ReferenceEvaluator(model).run(None, {"x": x})[0]
        if rv.shape != want.shape or rv.tobytes() != want.tobytes():
            deviations["onnx reference"].append((k, a, f"shape {list(rv.shape)}" if rv.shape != want.shape else "values"))
    except Exception:
        deviations["onnx reference"].append((k, a, "crashes"))
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        si = [dd.dim_value for dd in inferred.graph.output[0].type.tensor_type.shape.dim]
    except Exception:
        si = None
    if si != got_shape:
        deviations["onnx shape inference"].append((k, a, f"shape {si}"))

print(f"{args.count} random maxpools: {stats['ok']} valid models bit-identical to the spec emulation, "
      f"{stats['rejected']} invalid ones rejected, {stats['fail']} failures")
for who, items in deviations.items():
    kinds = {}
    for k, a, what in items:
        key = ("auto_pad=" + a.get("auto_pad", "NOTSET")) + (" dilated" if max(a.get("dilations", [1, 1])) > 1 else "") + \
              (" ceil_mode" if a.get("ceil_mode") else "") + ": " + what.split(" [")[0].split(" ")[0]
        kinds[key] = kinds.get(key, 0) + 1
    print(f"  {who} differs on {len(items)} of them: " + ", ".join(f"{v}x {key}" for key, v in sorted(kinds.items())))
raise SystemExit(1 if stats["fail"] else 0)
