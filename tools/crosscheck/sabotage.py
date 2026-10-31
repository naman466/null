"""Break one thing at a time, rebuild, and see what notices: ctest and,
optionally, one of the random cross-checks.

    python tools/crosscheck/sabotage.py tools/crosscheck/sabotages/reduce.py

A spec file defines SABOTAGES, a list of (name, file relative to the repo,
text to replace, replacement), and may define CHECK = (script, count) to run a
cross-check after ctest. Every sabotage must replace non-empty text that
occurs exactly once, the sabotaged build must succeed, and each file is put
back byte for byte, verified by hash, even if something fails."""
import argparse
import hashlib
import os
import re
import runpy
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves it out
from common import ROOT

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("spec", help="a sabotage spec file")
p.add_argument("--build", default=os.path.join(ROOT, "build-win"), help="the build folder to rebuild and test")
args = p.parse_args()
spec = runpy.run_path(args.spec)
here = os.path.dirname(os.path.abspath(__file__))


def build():
    r = subprocess.run(["cmake", "--build", args.build, "-j", "8"], capture_output=True, text=True)
    return r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]


ok, log = build()
if not ok:
    sys.exit("the unsabotaged build fails, fix that first:\n" + log)
for name, rel, old, new in spec["SABOTAGES"]:
    path = os.path.join(ROOT, rel)
    assert old and new and old != new, f"{name}: empty or no-op sabotage"
    before = open(path, "rb").read()
    text = before.decode("utf-8")
    assert text.count(old) == 1, f"{name}: the text occurs {text.count(old)} times in {rel}, not once"
    open(path, "wb").write(text.replace(old, new, 1).encode("utf-8"))
    try:
        ok, log = build()
        if not ok:
            print(f"{name}: SABOTAGED BUILD FAILED, nothing measured\n{log}")
            continue
        ct = subprocess.run(["ctest", "--test-dir", args.build], capture_output=True, text=True).stdout
        failed = re.findall(r"\d+ - (\S+) \(Failed\)", ct)
        line = f"{name}: ctest failed {failed or 'nothing'}"
        if "CHECK" in spec:
            script, count = spec["CHECK"]
            nullc = os.path.join(args.build, "null_cli" + (".exe" if sys.platform == "win32" else ""))
            out = subprocess.run([sys.executable, "-I", os.path.join(here, script), "--nullc", nullc,
                                  "--work", os.path.join(args.build, "crosscheck", "sabotage"), "--count", str(count)],
                                 capture_output=True, text=True).stdout
            summary = [l for l in out.splitlines() if l.startswith(f"{count} random")]
            line += "; " + (summary[0] if summary else out[-300:])
        print(line, flush=True)
    finally:
        open(path, "wb").write(before)
    assert hashlib.sha256(open(path, "rb").read()).digest() == hashlib.sha256(before).digest(), f"{rel} not restored"
ok, log = build()
assert ok, log
print("every file restored and rebuilt")
