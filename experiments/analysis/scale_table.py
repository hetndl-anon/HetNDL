"""Scalability table: Kuramoto-± (p- 0.5) with N = 100 (main table), 200 and 400 nodes at the same mean
degree. Accuracy: HetNDL default procedure (lower validation MSE of the two members; N = 100 from runs/sel/selected.json,
N >= 200 from runs/scale), AUROC / AUPRC / sign accuracy on true edges / node-constant |rho| (omega_readout.score);
ARNI best of full / greedy by AUROC (the main-table rule, favours ARNI) and Granger. Cost: runs/scale/bench.json
(scale_bench.py: one training iteration alone on an idle A100; per epoch and per 600-epoch fit) and the CPU wall times
of ARNI / Granger recorded by experiments/reproduce/run_jobs.py (runs/timing for N = 100). Run with PYTHONPATH=."""
import glob, json, os, re, numpy as np
from hetndl.snapshot import metrics_of
from experiments.analysis.omega_readout import score as omega_score
R = "experiments/runs"; sel = json.load(open(os.path.join(R, "sel", "selected.json")))


def ds(N, s):
    return "kura_ER%d_K2_s0.3_n0.5%s" % (N, "" if s == 1 else "_g%d" % s)


def ours(N, s):
    if N == 100:
        return sel["%s/add_z_signed_diff_lz0_seed%d" % (ds(N, s), s)]["run"]
    c = [(json.load(open(p))["val_mse_best"], os.path.dirname(p)) for p in glob.glob(os.path.join(R, "scale", ds(N, s), "*", "metrics.json"))
         if not json.load(open(p)).get("failed")]
    return min(c)[1] if len(c) == 2 else None


def best(paths):
    ms = [json.load(open(p)) for p in paths]; ms = [m for m in ms if not m.get("failed")]
    return max(ms, key=lambda m: m["auroc"]) if ms else None


def cpu_times():
    t = {}
    for f in ("experiments/reproduce/run_jobs.log",):                    # wall times written by run_jobs.py
        if os.path.exists(f):
            for line in open(f):
                m = re.match(r"\[done\] (\S+) rc=0 (\d+)s", line)
                if m:
                    t[m.group(1)] = int(m.group(2))
    return t


OUT = {"acc": {}}
print("accuracy (mean over seeds; per-seed AUROC in brackets)")
for N in (100, 200, 400):
    o, a, g = [], [], []
    for s in (1, 2, 3):
        r = ours(N, s)
        if r:
            b = metrics_of(json.load(open(r + "/metrics.json"))); o.append((b["auroc"], b["auprc"], b.get("sign_acc", np.nan), omega_score(r)[0]))
        x = best(glob.glob(os.path.join(R, "arni", ds(N, s), "*", "metrics.json"))); a += [(x["auroc"], x["auprc"])] if x else []
        x = best(glob.glob(os.path.join(R, "granger", ds(N, s), "*", "metrics.json"))); g += [(x["auroc"], x["auprc"])] if x else []
    f = lambda v, k=2: ("%s (%s)" % (" / ".join("%.3f" % x for x in np.mean(v, 0)[:k]), " ".join("%.3f" % x[0] for x in v))) if v else "pending"
    print("  N %4d  HetNDL AUROC/AUPRC/sign/|rho| %s | ARNI %s | Granger %s" % (N, f(o, 4), f(a), f(g)))
    OUT["acc"][str(N)] = dict(ours=o, arni=a, granger=g)
b = json.load(open(os.path.join(R, "scale", "bench.json"))) if os.path.exists(os.path.join(R, "scale", "bench.json")) else None
if b:
    print("\ncost on one idle %s (training iterations only)" % b["device"])
    for k, v in b["results"].items():
        print("  %-12s batch %3d  %6.1f s/epoch  peak %5.1f GiB  600 epochs %5.1f h" % (k, v["batch"], v["sec_per_epoch"], v["peak_gib"], v["est_train_h"]))
t = cpu_times()
if t:
    print("\nCPU wall time (s), seed 1, as recorded by experiments/reproduce/run_jobs.py:")
    for k in sorted(t):
        if k.endswith("seed1") and re.search(r"/(arni|granger)/kura_ER(200|400)_", k):
            print("  %-60s %6d" % (k, t[k]))
OUT["bench"] = b; OUT["cpu"] = cpu_times()

if "--json" in __import__("sys").argv:                                   # same numbers for the figure scripts
    _p = __import__("sys").argv[__import__("sys").argv.index("--json") + 1]
    os.makedirs(os.path.dirname(_p), exist_ok=True); json.dump(OUT, open(_p, "w"), indent=1, default=float)
