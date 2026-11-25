"""Generate every dataset used in the paper into experiments/data/.

The synthetic datasets are listed in experiments/reproduce/datasets.json (one generator command each). We regenerated
all 87 of them with this script and compared them array by array with the files used for the paper: they are identical
(the regenerated all-excitatory MixRelax files, and the sampling-interval sets derived from them, also store the true
weights W_true and the metadata entries neg_frac, n_neg, graph and m, which the files we used did not hold; the metadata
of the spring network names its generator module). Datasets derived from another one (the coarser sampling intervals of
hetndl.gen.subsample, --src) are built after their sources.
Two benchmarks need a downloaded input (see README.md):
  * Grid-118: the MATPOWER IEEE 118-bus case file at experiments/real/matpower_raw/case118.m (parsed as text);
  * NetSim:   the NetSim simulations sim1.mat ... sim28.mat unpacked into experiments/real/netsim_raw/
              (sim13 is skipped, as in NOGI).

Usage (repository root): python experiments/reproduce/make_datasets.py [--workers 4] [--only REGEX]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE118 = "experiments/real/matpower_raw/case118.m"
CASE118_SHA256 = "bc2e6f22b4b9e776572885ee4b50e4f4ab2ee0c5577e9126e86d906f14c4b5f7"
NETSIM_RAW = "experiments/real/netsim_raw"
NETSIM_SIMS = [k for k in range(1, 29) if k != 13]


def _run(item):
    path, cmd = item
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    r = subprocess.run([sys.executable, "-m"] + cmd + ["--out", path], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=ROOT))
    return path, r.returncode, r.stderr[-500:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="regular expression on the dataset path")
    a = ap.parse_args(argv)
    specs = json.load(open(os.path.join(ROOT, "experiments", "reproduce", "datasets.json")))
    todo = {p: c for p, c in specs.items() if not os.path.exists(os.path.join(ROOT, p))
            and (a.only is None or re.search(a.only, p))}
    if any("powergrid" in c[0] for c in todo.values()):
        case = os.path.join(ROOT, CASE118)
        if not os.path.exists(case):
            print("missing %s (download it first, see README.md); skipping Grid-118" % CASE118)
            todo = {p: c for p, c in todo.items() if "powergrid" not in c[0]}
        elif hashlib.sha256(open(case, "rb").read()).hexdigest() != CASE118_SHA256:
            print("warning: %s differs from the file we used (sha256 %s...)" % (CASE118, CASE118_SHA256[:12]))
    failed = []
    src = lambda c: c[c.index("--src") + 1] if "--src" in c else None      # derived from another dataset (subsample)
    for wave in ([p for p in sorted(todo) if src(todo[p]) is None], [p for p in sorted(todo) if src(todo[p])]):
        ready = [p for p in wave if src(todo[p]) is None or os.path.exists(os.path.join(ROOT, src(todo[p])))]
        failed += [(p, "missing source dataset %s" % src(todo[p])) for p in wave if p not in ready]
        with ThreadPoolExecutor(a.workers) as ex:
            for path, rc, err in ex.map(_run, [(p, todo[p]) for p in ready]):
                print("[%s] %s" % ("ok" if rc == 0 else "FAILED", path), flush=True)
                if rc != 0:
                    failed.append((path, err))
    netsim_missing = [k for k in NETSIM_SIMS if not os.path.exists(os.path.join(ROOT, "experiments/data/netsim_sim%d.npz" % k))]
    if netsim_missing and (a.only is None or re.search(a.only, "netsim")):
        if os.path.isdir(os.path.join(ROOT, NETSIM_RAW)):
            subprocess.run([sys.executable, "-m", "hetndl.gen.netsim", "--raw", NETSIM_RAW, "--out", "experiments/data"],
                           cwd=ROOT, check=True, env=dict(os.environ, PYTHONPATH=ROOT), stdout=subprocess.DEVNULL)
            print("[ok] NetSim (27 simulations)")
        else:
            print("missing %s (download NetSim first, see README.md); skipping NetSim" % NETSIM_RAW)
    for path, err in failed:
        print("FAILED", path, err, sep="\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
