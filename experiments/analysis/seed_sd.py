"""Spread of the Table-2 AUROC over the three data seeds (supplement, Per-Seed Results). Per setting and method: the sample
standard deviation (ddof 1) of the per-seed AUROC behind each Table-2 cell, from the same runs as baseline_table.py (ours:
runs/sel/selected.json; baselines: baseline_table.run_dirs; ARNI as in Table 2: on the Kuramoto settings the better of
its full and greedy variants per seed, elsewhere the full variant). Printed: the table of standard deviations (means in
parentheses, equal to Table 2), then per method the largest and the median over the twelve settings.
Run from the repository root with PYTHONPATH=."""
import sys
import numpy as np
from experiments.analysis.baseline_table import B, PAPER_NAME, met, run_dirs, score, sel, sel_key

COLS = ("ours", "NOGI", "ARNI", "Granger", "NRI", "AIDD", "COSINE", "MTGNN", "SGODE")


def per_seed_auroc(lab):
    out = {k: [] for k in COLS}
    for ds, s in dict(B)[lab]:
        g = sel_key(lab, ds, s)
        if g not in sel:
            continue
        out["ours"].append(sel[g]["auroc"])
        rd = run_dirs(lab, ds, s)
        for k in COLS[1:]:
            if k == "ARNI":
                v = [score(met(rd[b]))[0] for b in ("ARNI full", "ARNI greedy") if rd[b]]
                v = (max(v) if "Kuramoto" in lab else score(met(rd["ARNI full"]))[0]) if v and rd["ARNI full"] else None
            else:
                v = score(met(rd[k]))[0] if rd[k] else None
            if v is not None:
                out[k].append(v)
    return out


if __name__ == "__main__":
    lines = ["Standard deviation (ddof 1) of the AUROC over the three data seeds; mean in parentheses",
             "%-20s %s" % ("setting", " ".join("%-15s" % k for k in COLS))]
    sd = {k: [] for k in COLS}
    for lab, _ in B:
        a = per_seed_auroc(lab); cells = []
        for k in COLS:
            v = np.array(a[k], float)
            if len(v) == 3:
                sd[k].append(v.std(ddof=1)); cells.append("%.3f (%.3f)" % (v.std(ddof=1), v.mean()))
            else:
                cells.append("n=%d" % len(v))
        lines.append("%-20s %s" % (PAPER_NAME[lab], " ".join("%-15s" % c for c in cells)))
    lines.append("")
    for k in COLS:
        v = np.array(sd[k])
        lines.append("%-8s settings %2d  max %.3f  median %.3f  below 0.0005 in %d" % (k, len(v), v.max(), np.median(v), int((v < 5e-4).sum())))
    txt = "\n".join(lines); print(txt)
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(txt + "\n")
