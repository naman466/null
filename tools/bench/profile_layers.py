"""Where a compiled model spends its time: one line per step.

Compiles a model with nullc, puts a timer call before every step of the
generated model_run (each step begins with a "/* <node>: <Op> */" comment),
builds it with tools/bench/profile_model.c and prints, per step, the mean
milliseconds per run, the share of the total and, for convolutions and
matrix products, the GFLOP/s reached. A summary by operator follows.

    .venv/Scripts/python -I tools/bench/profile_layers.py tests/fixtures/pytorch/resnet18_small__legacy.onnx
    .venv/Scripts/python -I tools/bench/profile_layers.py --torchvision            # full-size ResNet-18 and MobileNetV2
    .venv/Scripts/python -I tools/bench/profile_layers.py model.onnx --cflags "-O2 -march=native -ffp-contract=off"
    .venv/Scripts/python -I tools/bench/profile_layers.py model.onnx --reference-kernels --top 10

Steps that move no data (a Reshape that only renames its input, say) are
left out. Only the standard library is needed, except for --torchvision.
"""
import argparse, os, re, shlex, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "tools", "bench")

STEP = re.compile(r"    /\* (.*): (\w+) \*/$")


def int_arrays(src):
    """static const int64_t name[n] = {...}: the generated shape tables"""
    return {m.group(1): [int(v) for v in m.group(2).split(",") if v.strip()]
            for m in re.finditer(r"static const int64_t (\w+)\[\d*\] = \{([^}]*)\};", src)}


def conv_params(src):
    """null_conv2d_params name = {...}: flops and a short description"""
    out = {}
    for m in re.finditer(r"static const null_conv2d_params (\w+) = \{([^}]*)\};", src):
        n, c, h, w, mm, kh, kw, g, sh, sw, dh, dw, pt, pl, oh, ow, relu = (int(v) for v in m.group(2).split(","))
        flops = 2 * n * mm * oh * ow * (c // g) * kh * kw
        what = f"{c}->{mm} {kh}x{kw}" + (f"/{sh}" if sh > 1 or sw > 1 else "") + (f" g{g}" if g > 1 else "") + f" {oh}x{ow}"
        out[m.group(1)] = (flops, what)
    return out


def args_of(line, kernel):
    m = re.search(kernel + r"\((.*)\);\s*$", line)
    return [a.strip() for a in m.group(1).split(",")] if m else None


def step_work(lines, convs, arrays):
    """(flops, description) of a step's convolution or matrix product, if it has one"""
    for ln in lines:
        k = re.search(r"&(\w+)\);\s*$", ln)
        if "null_conv2d" in ln and k and k.group(1) in convs:
            return convs[k.group(1)]
        a = args_of(ln, r"null_gemm(?:_ref)?_f32")
        if a and len(a) >= 7 and all(x.isdigit() for x in a[4:7]):
            m, n, kk = (int(x) for x in a[4:7])
            return 2 * m * n * kk, f"{m}x{kk} * {kk}x{n}"
        a = args_of(ln, r"null_matmul(?:_ref)?_f32")
        if a and len(a) >= 8 and all(x.isdigit() for x in a[3:7]):
            m, n, kk, rank = (int(x) for x in a[3:7])
            batch = 1
            for d in arrays.get(a[7], [])[:rank]:
                batch *= d
            return 2 * batch * m * n * kk, (f"{batch}x " if batch > 1 else "") + f"{m}x{kk} * {kk}x{n}"
    return 0, ""


def instrument(src):
    """the source with null_prof_mark(i) before every step of model_run, and the step tables"""
    start = src.index("void model_run(")
    end = src.index("\n}\n", start)
    lines = src[start:end].split("\n")
    convs, arrays = conv_params(src), int_arrays(src)
    steps, body, cur = [], [], None
    for ln in lines:
        m = STEP.match(ln)
        if m:
            cur = {"name": m.group(1), "op": m.group(2), "lines": []}
            body.append("    null_prof_mark(%d);" % len(steps))
            steps.append(cur)
        elif cur is not None:
            cur["lines"].append(ln)
        body.append(ln)
    body.append("    null_prof_mark(%d);" % len(steps))
    for s in steps:
        s["flops"], s["what"] = step_work(s["lines"], convs, arrays)
        # a step whose lines are all comments moved no data
        s["moves"] = any(ln.strip() and not ln.strip().startswith("/*") for ln in s["lines"])
    tables = "void null_prof_mark(int i);\nconst int null_prof_n = %d;\n" % len(steps)
    return src[:start] + tables + "\n".join(body) + src[end:], steps


def fmt_ms(ms):
    return f"{ms:.3f}" if ms < 10 else f"{ms:.1f}"


def profile(args, path, label):
    tag = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0])
    c_file = os.path.join(args.work, tag + "_prof.c")
    exe = os.path.join(args.work, tag + "_prof.exe")
    extra = ["--reference-kernels"] if args.reference_kernels else []
    r = subprocess.run([args.nullc, path, "--batch", "1", *extra, "-o", c_file], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"nullc failed on {path}:\n{r.stdout}{r.stderr}")
    with open(c_file, encoding="utf-8") as f:
        src, steps = instrument(f.read())
    with open(c_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    libm = [] if sys.platform == "win32" else ["-lm"]
    subprocess.run([args.cc, "-std=c99", *shlex.split(args.cflags), c_file,
                    os.path.join(HERE, "profile_model.c"), "-o", exe, *libm], check=True)
    r = subprocess.run([exe, str(args.runs)], check=True, capture_output=True, text=True)
    for m in re.finditer(r"step (\d+) ms ([0-9.]+)", r.stdout):
        steps[int(m.group(1))]["ms"] = float(m.group(2))

    total = sum(s["ms"] for s in steps)
    shown = [s for s in steps if s["moves"]]
    if args.top:
        shown = sorted(shown, key=lambda s: -s["ms"])[:args.top]
    width = min(max([len(s["name"]) for s in shown] + [4]), 60)
    print(f"{label}: {total:.3f} ms per run, mean of {args.runs} (cflags {args.cflags}"
          + (", reference kernels" if args.reference_kernels else "") + ")\n")
    print(f"{'step':<{width}}  {'op':<18} {'work':<28} {'ms':>9} {'share':>6} {'GFLOP/s':>8}")
    for s in shown:
        gf = f"{s['flops'] / (s['ms'] * 1e6):.1f}" if s["flops"] and s["ms"] > 0 else ""
        name = s["name"] if len(s["name"]) <= width else "..." + s["name"][-(width - 3):]
        print(f"{name:<{width}}  {s['op']:<18} {s['what']:<28} {fmt_ms(s['ms']):>9} "
              f"{100 * s['ms'] / total if total else 0:5.1f}% {gf:>8}")

    print(f"\n{'by operator':<20} {'steps':>5} {'ms':>9} {'share':>6} {'GFLOP/s':>8}")
    ops = {}
    for s in steps:
        if s["moves"]:
            o = ops.setdefault(s["op"], [0, 0.0, 0])
            o[0] += 1
            o[1] += s["ms"]
            o[2] += s["flops"]
    for op, (n, ms, flops) in sorted(ops.items(), key=lambda kv: -kv[1][1]):
        gf = f"{flops / (ms * 1e6):.1f}" if flops and ms > 0 else ""
        print(f"{op:<20} {n:>5} {fmt_ms(ms):>9} {100 * ms / total if total else 0:5.1f}% {gf:>8}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nullc", default=os.path.join(ROOT, "build-win", "nullc"))
    p.add_argument("--cc", default="gcc")
    p.add_argument("--cflags", default="-O2 -ffp-contract=off")
    p.add_argument("--runs", type=int, default=10, help="timed runs to average over")
    p.add_argument("--work", default=os.path.join(ROOT, "build-win", "bench"))
    p.add_argument("--reference-kernels", action="store_true", help="profile the plain loops instead")
    p.add_argument("--top", type=int, default=0, help="show only the N slowest steps")
    p.add_argument("--torchvision", action="store_true", help="full-size ResNet-18 and MobileNetV2")
    p.add_argument("models", nargs="*", help=".onnx files")
    args = p.parse_args()
    os.makedirs(args.work, exist_ok=True)
    models = [(os.path.basename(m), m) for m in args.models]
    if args.torchvision:
        sys.path.insert(0, HERE)
        from bench import export_torchvision
        models += export_torchvision(args.work)
    if not models:
        p.error("give a model, or --torchvision")
    for label, path in models:
        profile(args, path, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
