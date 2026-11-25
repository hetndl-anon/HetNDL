"""Ablation rows "one sign member only": for every main-table cell of runs/sel/manifest.json the
non-negative member alone, the plain signed member alone, and the default procedure (lower validation MSE of the two),
with the same grouping as sel_tables.py (grid restarts _r2 / _r3 belong to one cell and are chosen by validation MSE
within each member as well). Mean AUROC / AUPRC over data seeds per benchmark; Netsim = mean over the 27 sims of the
per-sim 3-seed means. Robustness cells (noise / trajectory caps / SNR / smoothing) are excluded. Run from the project
root with PYTHONPATH=."""
import collections, json, os, re, numpy as np
from hetndl.snapshot import metrics_of
from experiments.analysis.paper_seeds import PAPER_SEEDS
R = "experiments/runs"; man = json.load(open(os.path.join(R, "sel", "manifest.json")))
MAIN = re.compile(r"add_z_signed_diff_lz0_seed\d+(_r\d+)?$")


def load(p):
    f = os.path.join(p, "metrics.json")
    if not os.path.exists(f):
        return None
    m = json.load(open(f))
    return None if m.get("failed") else m


groups = collections.defaultdict(list)
for cell, e in man.items():
    if MAIN.search(cell.split("/")[-1]) and int(re.search(r"seed(\d+)", cell.split("/")[-1]).group(1)) in PAPER_SEEDS:
        groups[re.sub(r"_r\d+$", "", cell)].append(e)
rows = collections.defaultdict(lambda: collections.defaultdict(list)); missing = []
for g, entries in sorted(groups.items()):
    best = {}
    for v in ("nonneg", "nowu"):
        c = [(m["val_mse_best"], m) for m in (load(e[v]) for e in entries) if m is not None]
        if c:
            best[v] = min(c, key=lambda x: x[0])
    if len(best) < 2:
        missing.append(g); continue
    pick = min(best, key=lambda v: best[v][0])
    ds = g.split("/")[-2]
    fam = "netsim" if ds.startswith("netsim") else re.sub(r"(_g\d+|_seed\d+)$", "", ds)
    key = (fam, ds if fam == "netsim" else "")
    for lab, v in (("nonneg only", "nonneg"), ("signed only", "nowu"), ("default", pick)):
        b = metrics_of(best[v][1]); rows[key][lab].append((b["auroc"], b["auprc"]))
    rows[key]["picks"].append("n" if pick == "nonneg" else "s")

fams = collections.defaultdict(lambda: collections.defaultdict(list))
for (fam, sim), d in rows.items():
    for lab in ("nonneg only", "signed only", "default"):
        fams[fam][lab].append(np.mean(d[lab], 0))
    fams[fam]["picks"].append("".join(d["picks"]))
print("%-28s %4s  %-15s %-15s %-15s picks" % ("benchmark", "n", "nonneg only", "signed only", "default"))
for fam in sorted(fams):
    d = fams[fam]
    print("%-28s %4d  %-15s %-15s %-15s %s" % (fam, len(d["default"]) if fam == "netsim" else len(d["picks"][0]), *("%.3f / %.3f" % tuple(np.mean(d[l], 0)) for l in ("nonneg only", "signed only", "default")),
                                                " ".join(d["picks"]) if fam != "netsim" else "%d sims" % len(d["picks"])))
if missing:
    print("incomplete cells (a member missing or failed):", missing)
