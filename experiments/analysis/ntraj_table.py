"""Few-trajectory comparison: ours (default procedure, robustness cells of runs/sel/selected.json) vs
ARNI and Granger-lasso run with the SAME trajectory caps and seeds (experiments/launch/gap_cpu.py -> runs/arni_ntraj,
runs/granger_ntraj); the full-data row is the main table. ARNI on Kuramoto = best AUROC of full / greedy per seed (the
main-table rule, a truth-picked advantage for ARNI); on A2 N60 full mode as in the main table. Mean AUROC / AUPRC over
data seeds 1-3. Run from the project root with PYTHONPATH=."""
import glob, json, os, numpy as np
from hetndl.snapshot import metrics_of
R = "experiments/runs"; sel = json.load(open(os.path.join(R, "sel", "selected.json")))
BEN = {"Kuramoto-± p- 0.5": (lambda s: "kura_ER100_K2_s0.3_n0.5" + ("" if s == 1 else "_g%d" % s), (3, 5, 10)),
       "MixOsc N60": (lambda s: "a2_N60_seed%d" % s, (5, 10))}


def best(paths):
    ms = [json.load(open(p)) for p in paths]; ms = [m for m in ms if not m.get("failed")]
    return (max(ms, key=lambda m: m["auroc"])["auroc"], max(ms, key=lambda m: m["auroc"])["auprc"]) if ms else None


def f(v):
    v = [x for x in v if x is not None]
    return "%.3f / %.3f" % tuple(np.mean(v, 0)) if v else "—"


OUT = {}
print("%-20s %-6s %-16s %-16s %-16s" % ("benchmark", "traj", "ours", "ARNI", "Granger"))
for lab, (dsf, ns) in BEN.items():
    for n in ns + ("all",):
        o, a, g = [], [], []
        for s in (1, 2, 3):
            ds = dsf(s); tag = "" if n == "all" else "_n%d" % n
            k = "%s/add_z_signed_diff_lz0%s_seed%d" % (ds, tag, s)
            o.append((sel[k]["auroc"], sel[k]["auprc"]) if k in sel else None)
            if n == "all":
                a.append(best(glob.glob(os.path.join(R, "arni", ds, "*", "metrics.json"))))
                g.append(best(glob.glob(os.path.join(R, "granger", ds, "*", "metrics.json"))))
            else:
                a.append(best(glob.glob(os.path.join(R, "arni_ntraj", ds, "*_n%d_seed%d" % (n, s), "metrics.json"))))
                g.append(best(glob.glob(os.path.join(R, "granger_ntraj", ds, "n%d_seed%d" % (n, s), "metrics.json"))))
        print("%-20s %-6s %-16s %-16s %-16s" % (lab, n, f(o), f(a), f(g)))
        OUT.setdefault(lab, {})[str(n)] = dict(ours=o, arni=a, granger=g)

if "--json" in __import__("sys").argv:                                   # same numbers for the figure scripts
    _p = __import__("sys").argv[__import__("sys").argv.index("--json") + 1]
    os.makedirs(os.path.dirname(_p), exist_ok=True); json.dump(OUT, open(_p, "w"), indent=1, default=float)
