"""Random single-AveragePool models (opset 19, so dilations are allowed): the
same windows as check_maxpool.py (kernels, strides, dilations, explicit pads
or auto_pad, ceil_mode), with count_include_pad on or off, and inputs with
ties, infinities, NaN and large values.

The arbiter is an emulation of the specification: the output size and padding
from its formulas (as for MaxPool), and each output the sum of the window's
pixels, in float64, divided by a count and rounded to float32 once. Without
count_include_pad the count is the number of pixels in the window; with it,
the number of window positions inside the padded image, which leaves out the
part of a ceil_mode window that hangs past the end padding. nullc must match
the emulation bit for bit (any NaN equal to any NaN).

onnxruntime and the onnx reference evaluator add in float32, so they are held
to the float32 summation bound (or allowed to overflow where the float32 sum
would). onnxruntime deviates from the specification in three known ways, and
any other deviation fails: where auto_pad=SAME_* asks for a negative total
padding it does not clamp it to 0 but splits it, which can shift the windows
by one (modelled here exactly); it gets dilated SAME_* pooling wrong (as for MaxPool); and it sizes
auto_pad=VALID with ceil_mode differently. The reference evaluator's
differences are only listed: it refuses ceil_mode with auto_pad and negative
SAME padding, and disagrees in other corners of ceil_mode and dilation."""
import collections
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import parse_args, compile_model, output_shape, run_generated, run_ort, ulps

warnings.filterwarnings("ignore")
args = parse_args(__doc__, 500, "avgpool")
rng = np.random.default_rng(1313)


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
    if rng.random() < 0.5:
        attrs["count_include_pad"] = int(rng.integers(0, 2))
    return dict(n=int(rng.integers(1, 3)), c=int(rng.integers(1, 4)), h=int(rng.integers(1, 12)),
                w=int(rng.integers(1, 12)), attrs=attrs)


def make_model(cfg):
    node = helper.make_node("AveragePool", ["x"], ["y"], name="pool", **cfg["attrs"])
    g = helper.make_graph([node], "pool",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [cfg["n"], cfg["c"], cfg["h"], cfg["w"]])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    return model


def spec(cfg, clamp=True):
    """output size and padding (before and after) from the specification's formulas, None if invalid.
    clamp=False models onnxruntime: a negative SAME padding is kept"""
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
            if pads[i] >= k[i] or pads[i + 2] >= k[i]:
                return None
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
            total = (o - 1) * st[i] + eff - ins[i]
            total = max(0, total) if clamp else total
            # the padding before the image: total // 2 for SAME_UPPER and total - total // 2 for
            # SAME_LOWER. onnxruntime's own forms, (total + 1) / 2 for LOWER and C's division,
            # which truncates, agree for any total >= 0 and differ only for the negative ones it keeps
            if clamp:
                begin[i] = total // 2 if mode == "SAME_UPPER" else total - total // 2
            else:
                begin[i] = int(total / 2) if mode == "SAME_UPPER" else int((total + 1) / 2)
            end[i] = total - begin[i]
        out[i] = o
    return out, begin, end, dl


def emulate(x, cfg, geom):
    (oh_n, ow_n), (pt, pl), (pb, pr), dl = geom
    a = cfg["attrs"]
    kh, kw = a["kernel_shape"]; sh, sw = a["strides"]
    include = a.get("count_include_pad", 0)
    x64 = x.astype(np.float64)
    y = np.zeros((cfg["n"], cfg["c"], oh_n, ow_n), np.float32)
    abs_sum = np.zeros(y.shape, np.float64)
    pixels = np.zeros(y.shape, np.float64)
    counts = np.zeros(y.shape, np.float64)
    for oh in range(oh_n):
        for ow in range(ow_n):
            total = np.zeros((cfg["n"], cfg["c"]), np.float64)
            count = 0
            seen = 0
            for r in range(kh):
                ih = oh * sh - pt + r * dl[0]
                for s_ in range(kw):
                    iw = ow * sw - pl + s_ * dl[1]
                    if 0 <= ih < cfg["h"] and 0 <= iw < cfg["w"]:
                        total += x64[:, :, ih, iw]
                        with np.errstate(all="ignore"):
                            abs_sum[:, :, oh, ow] += np.abs(x64[:, :, ih, iw])
                        count += 1
                        seen += 1
                    elif include and -pt <= ih < cfg["h"] + pb and -pl <= iw < cfg["w"] + pr:
                        count += 1
            with np.errstate(all="ignore"):
                y[:, :, oh, ow] = (total / count if count else total * np.nan).astype(np.float32)
            pixels[:, :, oh, ow] = seen
            counts[:, :, oh, ow] = count
    return y, abs_sum, pixels, counts


def ort_known(cfg):
    """which of onnxruntime's known deviations this model can meet, if any"""
    a = cfg["attrs"]
    mode = a.get("auto_pad", "NOTSET")
    if mode.startswith("SAME") and max(a.get("dilations", [1, 1])) > 1:
        return "dilated SAME_*"
    if mode == "VALID" and a.get("ceil_mode"):
        return "VALID with ceil_mode"
    if mode.startswith("SAME") and spec(cfg, clamp=False)[1:3] != spec(cfg)[1:3]:
        return "negative SAME padding"
    return None


def describe(a):
    return (("auto_pad=" + a.get("auto_pad", "NOTSET")) + (" dilated" if max(a.get("dilations", [1, 1])) > 1 else "")
            + (" ceil_mode" if a.get("ceil_mode") else "") + (" count_include_pad" if a.get("count_include_pad") else ""))


stats = collections.Counter()
close = {"onnxruntime": collections.Counter(), "onnx reference": collections.Counter()}
deviations = {"onnxruntime": collections.Counter(), "onnx reference": collections.Counter()}
for k in range(args.count):
    cfg = random_config()
    model = make_model(cfg)
    d = os.path.join(args.work, f"pool{k:03d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "pool.onnx")
    onnx.save(model, path)
    shp = (cfg["n"], cfg["c"], cfg["h"], cfg["w"])
    x = rng.standard_normal(shp).astype(np.float32)
    if k % 3 == 0:
        x = np.round(x).astype(np.float32)
    if k % 7 == 0:
        x[rng.random(shp) < 0.1] = rng.choice([np.inf, -np.inf, np.nan])
    if k % 11 == 0:
        x = (x * np.float32(1e38)).astype(np.float32)          # sums past the float range
    geom = spec(cfg)
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
    want, abs_sum, pixels, counts = emulate(x, cfg, geom)
    got_shape = output_shape(c_file)
    if got_shape != list(want.shape):
        stats["fail"] += 1
        print(f"pool{k:03d}: shape nullc {got_shape} spec {list(want.shape)} cfg {cfg}")
        continue
    got = run_generated(args.cc, c_file, x, got_shape, d)
    if ulps(got, want) != 0:
        stats["fail"] += 1
        print(f"pool{k:03d}: nullc is {ulps(got, want)} ulp from the spec emulation cfg {cfg}")
        continue
    stats["ok"] += 1
    a = cfg["attrs"]
    for who in close:
        if who == "onnxruntime":
            ref, rerr = run_ort(model, x)
        else:
            try:
                ref, rerr = np.asarray(ReferenceEvaluator(model).run(None, {"x": x})[0], np.float32), None
            except Exception as e:
                ref, rerr = None, str(e)
        known = ort_known(cfg) if who == "onnxruntime" else None
        expect, e_abs, e_pix, e_cnt = want, abs_sum, pixels, counts
        if known == "negative SAME padding":
            expect, e_abs, e_pix, e_cnt = emulate(x, cfg, spec(cfg, clamp=False))
        if ref is None or list(ref.shape) != list(expect.shape):
            what = "refuses " + rerr.split("Status Message:")[-1].strip()[:50] if ref is None else "shape"
            if who == "onnxruntime" and known in ("dilated SAME_*", "VALID with ceil_mode"):
                close[who]["known: " + known] += 1
            elif who == "onnxruntime":
                stats["fail"] += 1
                print(f"pool{k:03d}: onnxruntime {what} on {cfg}, not a known deviation")
            else:
                deviations[who][describe(a) + ": " + what] += 1
        else:
            # adding n floats in float32 is within n * 2^-24 * sum|x| of the exact sum (n - 1 additions,
            # a small margin), and dividing and rounding adds half an ulp. a sum past the float range
            # overflows in float32 but not in float64
            with np.errstate(all="ignore"):
                limit = (e_pix * 2.0 ** -24 * e_abs) / np.maximum(e_cnt, 1) + 2.0 ** -24 * np.abs(expect.astype(np.float64))
                err = np.abs(ref.astype(np.float64) - expect.astype(np.float64))
            same = (err <= limit) | (np.isnan(ref) & np.isnan(expect)) | (ref == expect)
            overflow = ~same & ~np.isfinite(ref) & (e_abs > np.finfo(np.float32).max)
            if not (same | overflow).all():
                if who == "onnxruntime" and known == "dilated SAME_*":
                    close[who]["known: " + known] += 1
                elif who == "onnxruntime":
                    stats["fail"] += 1
                    print(f"pool{k:03d}: onnxruntime differs on {int((~(same | overflow)).sum())} values, "
                          f"not a known deviation, cfg {cfg}")
                else:
                    deviations[who][describe(a) + ": values"] += 1
            elif known == "negative SAME padding" and ulps(expect, want) != 0:
                close[who]["known: " + known] += 1
            elif same.all():
                u = ulps(ref, want)
                close[who]["0" if u == 0 else "1" if u == 1 else "within the summation bound"] += 1
            else:
                close[who]["the float32 sum overflows"] += 1

print(f"{args.count} random average pools: {stats['ok']} valid models bit-identical to the spec emulation, "
      f"{stats['rejected']} invalid ones refused, {stats['fail']} failures")
for who in close:
    print(f"  {who}: {dict(sorted(close[who].items()))}; otherwise differs or refuses on "
          f"{sum(deviations[who].values())}: {dict(sorted(deviations[who].items())) or 'none'}")
raise SystemExit(1 if stats["fail"] else 0)
