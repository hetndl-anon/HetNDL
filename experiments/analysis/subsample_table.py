"""Sampling interval x2 and x4. Data <ds>_sub{2,4} (hetndl.gen.subsample): every 2nd / 4th stored state of the
trajectories of kura_ER100_K2_s0.3_n0.5{,_g2,_g3} (Kuramoto-±, p⁻ = 0.5; data seeds 1-3) and a1v2_N30_seed{1,2,3}
(MixRelax), dt x 2 / x 4, nothing else changed. HetNDL: the two candidates (subsample/<ds>/{nonneg,nowu}/<run> of
experiments/reproduce/jobs.json) with the arguments of the main-table members, data replaced; the one with the lower
validation error (val_mse_best) is read (the selection rule of the paper). ARNI: the full variant (the one the paper
reports; subsample/<ds>/arni_full_seed<k>) with the arguments of its main-table run. Reference = the main table:
HetNDL's selected run (runs/sel/selected.json) and ARNI's full run (runs/arni/<ds>/full_seed<k>) on the same data seed.
Per setting and factor: the mean AUROC over the three seeds and its difference to the main table; a difference of at
most 0.02 counts as unchanged. A setting is printed only when all its HetNDL runs exist. Truth enters only through the
scores.
Run from the repository root with PYTHONPATH=. (-> results/supplement/subsample.txt)."""
import glob, json, os
import numpy as np
from hetndl.snapshot import metrics_of
from experiments.analysis.baseline_table import sel, sel_key
R = "experiments/runs"; THR = 0.02
SETTINGS = (("Kuramoto-± p⁻ 0.5", "Kuramoto p⁻ 0.5", [("kura_ER100_K2_s0.3_n0.5", 1), ("kura_ER100_K2_s0.3_n0.5_g2", 2), ("kura_ER100_K2_s0.3_n0.5_g3", 3)]),
            ("MixRelax", "A1 (N30)", [("a1v2_N30_seed%d" % k, k) for k in (1, 2, 3)]))


def met(p):
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return "failed" if m.get("failed") else m


def hetndl(ds):
    c = []
    for member in ("nonneg", "nowu"):
        ps = glob.glob(os.path.join(R, "subsample", ds, member, "*", "metrics.json"))
        if len(ps) != 1:
            return None
        m = met(ps[0])
        if m is None:
            return None
        if m != "failed":
            c.append((m["val_mse_best"], member, m))
    if not c:
        return "failed"
    v, member, m = min(c, key=lambda x: x[0]); b = metrics_of(m)
    return member, b["auroc"], b["auprc"]


def main():
    print("Sampling interval x2 / x4; HetNDL = selected candidate, mean AUROC over data seeds 1-3 against the main table\n")
    rows = []
    for name, lab, cells in SETTINGS:
        ref = [sel[sel_key(lab, ds, k)]["auroc"] for ds, k in cells]
        aref = [met(os.path.join(R, "arni", ds, "full_seed%d" % k, "metrics.json"))["auroc"] for ds, k in cells]
        for f in (2, 4):
            hs = [hetndl("%s_sub%d" % (ds, f)) for ds, _ in cells]
            if any(h is None for h in hs):
                print("%-18s x%d  missing runs" % (name, f)); rows.append(None); continue
            au = [np.nan if h == "failed" else h[1] for h in hs]
            ar = [met(os.path.join(R, "subsample", "%s_sub%d" % (ds, f), "arni_full_seed%d" % k, "metrics.json")) for ds, k in cells]
            ar = [np.nan if (a is None or a == "failed") else a["auroc"] for a in ar]
            d = np.mean(au) - np.mean(ref)
            rows.append((name, f, d, bool(np.isfinite(d) and abs(d) <= THR)))
            print("%-18s x%d  HetNDL %s  mean %.3f (default %.3f, diff %+.3f) picks %s | ARNI %s mean %.3f (default %.3f)" % (
                name, f, " ".join("%.3f" % x for x in au), np.mean(au), np.mean(ref), d,
                ",".join("-" if h == "failed" else h[0] for h in hs), " ".join("%.3f" % x for x in ar), np.mean(ar), np.mean(aref)))
    if all(r is not None for r in rows):
        lower = ["%s x%d (%+.3f)" % (r[0], r[1], r[2]) for r in rows if not r[3]]
        print("\nHetNDL within %.2f of the main table: %d of %d settings and factors%s" % (
            THR, sum(r[3] for r in rows), len(rows), ("; not within: " + ", ".join(lower)) if lower else ""))


if __name__ == "__main__":
    main()
