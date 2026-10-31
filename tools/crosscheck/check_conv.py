"""Random single-Conv models. Each output shape must equal onnx strict shape
inference, each output must be bit-identical to a numpy float32 emulation of
the runtime's documented summation order (acc = 0; channels, rows, columns
ascending; padding skipped; bias last), and close to onnxruntime."""
import os
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto, shape_inference

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort

args = parse_args(__doc__, 400, "conv")
rng = np.random.default_rng(5010)


def random_config():
    group = int(rng.choice([1, 1, 1, 2, 3, 4]))
    cg, mg = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    c, m = cg * group, mg * group
    kh, kw = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    h, w = int(rng.integers(1, 14)), int(rng.integers(1, 14))
    attrs = dict(group=group, strides=[int(rng.integers(1, 4)), int(rng.integers(1, 4))],
                 dilations=[int(rng.integers(1, 3)), int(rng.integers(1, 3))])
    mode = rng.choice(["NOTSET", "NOTSET", "NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER", "OMIT"])
    if mode == "NOTSET":
        attrs["pads"] = [int(p) for p in rng.integers(0, 3, 4)]
    elif mode != "OMIT":
        attrs["auto_pad"] = str(mode)
    if rng.random() < 0.5:
        attrs["kernel_shape"] = [kh, kw]
    return dict(n=int(rng.integers(1, 3)), c=c, h=h, w=w, m=m, kh=kh, kw=kw, bias=bool(rng.random() < 0.7), attrs=attrs)


def make_model(cfg):
    wshape = [cfg["m"], cfg["c"] // cfg["attrs"]["group"], cfg["kh"], cfg["kw"]]
    wt = rng.standard_normal(wshape).astype(np.float32)
    inits = [helper.make_tensor("w", TensorProto.FLOAT, wshape, wt.ravel())]
    ins = ["x", "w"]
    bt = None
    if cfg["bias"]:
        bt = rng.standard_normal(cfg["m"]).astype(np.float32)
        inits.append(helper.make_tensor("b", TensorProto.FLOAT, [cfg["m"]], bt))
        ins.append("b")
    node = helper.make_node("Conv", ins, ["y"], name="conv", **cfg["attrs"])
    g = helper.make_graph([node], "conv",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [cfg["n"], cfg["c"], cfg["h"], cfg["w"]])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, wt, bt


def pads_of(cfg):
    a = cfg["attrs"]
    eff = [(cfg["kh"] - 1) * a["dilations"][0] + 1, (cfg["kw"] - 1) * a["dilations"][1] + 1]
    mode = a.get("auto_pad", "NOTSET")
    if mode == "NOTSET":
        return a.get("pads", [0, 0, 0, 0])
    if mode == "VALID":
        return [0, 0, 0, 0]
    begin, end = [], []
    for i, size in enumerate([cfg["h"], cfg["w"]]):
        s = a["strides"][i]
        out = -(-size // s)
        total = max(0, (out - 1) * s + eff[i] - size)
        b = total // 2 if mode == "SAME_UPPER" else total - total // 2
        begin.append(b)
        end.append(total - b)
    return [begin[0], begin[1], end[0], end[1]]


def emulate(x, wt, bt, cfg, out_h, out_w):
    a = cfg["attrs"]
    group, (sh, sw), (dh, dw) = a["group"], a["strides"], a["dilations"]
    pt, pl, _, _ = pads_of(cfg)
    cg, mg = cfg["c"] // group, cfg["m"] // group
    n = cfg["n"]
    y = np.zeros((n, cfg["m"], out_h, out_w), np.float32)
    oh = np.arange(out_h)[:, None]
    ow = np.arange(out_w)[None, :]
    for m in range(cfg["m"]):
        acc = np.zeros((n, out_h, out_w), np.float32)
        for ci in range(cg):
            c_in = (m // mg) * cg + ci
            for r in range(cfg["kh"]):
                ih = oh * sh - pt + r * dh
                for s in range(cfg["kw"]):
                    iw = ow * sw - pl + s * dw
                    valid = (ih >= 0) & (ih < cfg["h"]) & (iw >= 0) & (iw < cfg["w"])
                    ihc, iwc = np.clip(ih, 0, cfg["h"] - 1), np.clip(iw, 0, cfg["w"] - 1)
                    prod = (x[:, c_in, ihc, iwc] * wt[m, ci, r, s]).astype(np.float32)
                    acc = np.where(valid, (acc + prod).astype(np.float32), acc)   # skipped, not + 0
        if bt is not None:
            acc = (acc + bt[m]).astype(np.float32)
        y[:, m] = acc
    return y


stats = dict(ok=0, rejected_both=0, onnx_lenient=0, no_ort=0, fail=0)
worst_rel_ort = 0.0
for k in range(args.count):
    cfg = random_config()
    model, wt, bt = make_model(cfg)
    d = os.path.join(args.work, f"conv{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "conv.onnx")
    onnx.save(model, path)
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        want = [dd.dim_value for dd in inferred.graph.output[0].type.tensor_type.shape.dim]
        onnx_ok = all(v > 0 for v in want)
    except Exception:
        want, onnx_ok = None, False
    c_file = os.path.join(d, "model.c")
    ok, err = compile_model(args.nullc, path, c_file)
    if not onnx_ok or not ok:
        if not onnx_ok and not ok:
            stats["rejected_both"] += 1
        elif onnx_ok and "larger than the padded input" in err and \
                "Invalid input shape" in str(run_ort(model, np.zeros((cfg["n"], cfg["c"], cfg["h"], cfg["w"]), np.float32))[1]):
            # onnx's inference truncates (in - k) / s toward zero and reports a size; ort and null refuse
            stats["onnx_lenient"] += 1
        else:
            stats["fail"] += 1
            print(f"conv{k:03d} disagreement: onnx {want} / nullc {err.strip()[:150]} cfg {cfg}")
        continue
    got_shape = output_shape(c_file)
    if got_shape != want:
        stats["fail"] += 1
        print(f"conv{k:03d} shape: nullc {got_shape} onnx {want} cfg {cfg}")
        continue
    x = rng.standard_normal((cfg["n"], cfg["c"], cfg["h"], cfg["w"])).astype(np.float32)
    got = run_generated(args.cc, c_file, x, want, d)
    emu = emulate(x, wt, bt, cfg, want[2], want[3])
    bit_exact = got.tobytes() == emu.tobytes()
    ref, err = run_ort(model, x)
    if ref is None:   # ort has no dilation with SAME_*; the emulation still checks these
        assert "Dilation not supported" in err, err
        stats["no_ort"] += 1
        ref = emu
    scale = np.maximum(np.abs(ref).astype(np.float64), 1.0)
    rel = float((np.abs(got.astype(np.float64) - ref) / scale).max())
    worst_rel_ort = max(worst_rel_ort, rel)
    if not bit_exact or rel > 1e-5:
        stats["fail"] += 1
        print(f"conv{k:03d} values: bit-exact {bit_exact}, vs ort {rel:.1e} cfg {cfg}")
    else:
        stats["ok"] += 1

print(f"{args.count} random convs: {stats['ok']} match (shape == onnx, bit-exact == emulation, close to ort), "
      f"{stats['rejected_both']} rejected by both, {stats['onnx_lenient']} too-small inputs onnx sizes but ort and "
      f"null reject, {stats['fail']} failures, {stats['no_ort']} dilated SAME convs ort cannot run; "
      f"worst error vs ort {worst_rel_ort:.1e}")
raise SystemExit(1 if stats["fail"] else 0)
