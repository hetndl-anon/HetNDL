"""The shared-coupling model retrained on the 36 main-table cells with the scale of the coupling fixed (--gauge_scale ref):
after every step (W, G) -> (r W, G / r) with r the RMS of the diffusive coupling on 32 fixed training states, so that the
coupling has unit RMS and lambda_1 ||W||_1 a definite scale; the model function is not changed by this step. Runs gauge/...
of experiments/reproduce/jobs.json: both candidates of every cell (the two members of the cell in
experiments/reproduce/sel_manifest.json), all other arguments unchanged. Written before any AUROC / AUPRC / selection of
these runs was read; at that time only the run states were known (four non-negative runs on Kuramoto-omega had stopped
with a non-finite loss).

Rule fixed before the runs: per setting the mean AUROC of the selected candidate against the main table; "unchanged" if
the difference is at most 0.005 and the same candidates are selected. Made operational here, before reading:
  * selection exactly as the main table (sel_tables.py): per cell the member (non-negative add_z / signed add_z_signed)
    with the lower val_mse_best; scores of the reported snapshot through hetndl.snapshot.metrics_of;
  * main table = experiments/runs/sel/selected.json (the numbers behind Table 2), "same candidates" = the same member
    (n / s) in every cell of the setting;
  * a cell with a failed or missing member is pending and left out of the mean of the fixed-scale runs (as
    sel_tables.py); the comparison is then made on the cells present in both (the main-table mean over all three cells
    is printed too) and the verdict carries the number of cells it rests on;
  * secondary column: the same selection with each failed run's last good snapshot (its `best` block, written by
    train.py before the run stopped);
  * Grid-118: the manifest cell is retrained, i.e. restart 1 of each data seed, while Table 2 takes the lowest validation
    error over three restarts x two members. The verdict follows the rule literally (against Table 2); a like-for-like
    line against restart 1 of the default runs is printed below it.
Also printed: AUPRC, sign accuracy on true edges (settings with inhibitory edges), the number of diverged runs, and the
coupling RMS at the end of training (coupling_rms_last; 1 by construction when the rescaling step works).
Ground truth only in the scores. Run from the project root with PYTHONPATH=. (-> results/supplement/coupling_scale.txt)."""
import json, os, numpy as np
from hetndl.snapshot import metrics_of
from experiments.analysis.baseline_table import B, PAPER_NAME, sel, sel_key
R = "experiments/runs"; TOL = 0.005
man = json.load(open(os.path.join(R, "sel", "manifest.json")))
MEMBERS = ("nonneg", "nowu")


def load(run):
    p = os.path.join(run, "metrics.json")
    if not os.path.exists(p):
        return None, "missing"
    m = json.load(open(p))
    return m, ("failed" if m.get("failed") else "ok")


def pick(runs, allow_failed=False):
    """(variant, metrics of the reported snapshot, val) of the lower-val_mse_best member, or (None, notes) if pending."""
    c, notes = [], []
    for v, run in runs:
        m, st = load(run)
        if st == "missing" or (st == "failed" and not (allow_failed and "best" in m)):
            notes.append("%s %s" % (v, st)); continue
        c.append((m["val_mse_best"], v, m))
    if notes:
        return None, notes
    val, v, m = min(c, key=lambda x: x[0]); b = metrics_of(m)
    return v, dict(auroc=b["auroc"], auprc=b["auprc"], sign=b.get("sign_acc") if b.get("has_neg_edges") else None, val=val)


def scaled_runs(lab, ds, s):
    cell = man[sel_key(lab, ds, s)]
    return [(v, os.path.join(R, "gauge", v, ds, os.path.basename(cell[v]) + "_gs")) for v in MEMBERS]


def f3(x):
    return "  -  " if x is None else "%.3f" % x


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main():
    nfail, rms, verdicts = [], [], []
    print("Fixed coupling scale (--gauge_scale ref) vs Table 2; per data seed: member picked (n = non-negative, "
          "s = signed) and AUROC\n")
    print("%-22s %-34s %-35s %8s  %-15s %s" % ("setting", "Table 2: picks AUROC", "fixed scale: picks AUROC", "diff", "verdict",
                                               "secondary (failed runs' best snapshot): picks AUROC verdict"))
    for lab, cs in B:
        name = PAPER_NAME[lab]; rows = []
        for ds, s in cs:
            t2 = sel[sel_key(lab, ds, s)]
            runs = scaled_runs(lab, ds, s)
            for v, run in runs:
                m, st = load(run)
                if st == "failed":
                    nfail.append("%s seed %d %s (epoch %s: %s)" % (name, s, v, m.get("epochs_run"), m.get("reason", "")[:40]))
                elif st == "ok" and m.get("coupling_rms_last") is not None:
                    rms.append(m["coupling_rms_last"])
            v1, r1 = pick(runs); v2, r2 = pick(runs, allow_failed=True)
            rows.append((s, t2, v1, r1, v2, r2))
        both = [r for r in rows if r[2] is not None]
        t2_all = mean([r[1]["auroc"] for r in rows]); t2_same = mean([r[1]["auroc"] for r in both])
        fx = mean([r[3]["auroc"] for r in both])
        same_pick = all((r[1]["variant"] == "nonneg") == (r[2] == "nonneg") for r in both)
        if not both:
            verdict = "pending"
        else:
            ok = abs(fx - t2_same) <= TOL and same_pick
            verdict = ("unchanged" if ok else "changed") + ("" if len(both) == 3 else " (%d/3)" % len(both))
        verdicts.append((name, verdict))
        sec_ok = [r for r in rows if r[4] is not None]
        sec = mean([r[5]["auroc"] for r in sec_ok])
        sec_pick = all((r[1]["variant"] == "nonneg") == (r[4] == "nonneg") for r in sec_ok)
        sec_v = ("unchanged" if sec_ok and abs(sec - t2_all) <= TOL and sec_pick and len(sec_ok) == 3 else "changed" if len(sec_ok) == 3 else "pending")
        picks = lambda xs: "".join("n" if x == "nonneg" else "s" if x == "nowu" else "." for x in xs)
        print("%-22s %-3s %s (same seeds %s)      %-3s %-31s %8s  %-15s %-3s %s %s" % (
            name, picks([r[1]["variant"] for r in rows]), f3(t2_all), f3(t2_same), picks([r[2] for r in rows]), f3(fx),
            "%+.3f" % (fx - t2_same) if fx is not None else "-", verdict, picks([r[4] for r in rows]), f3(sec), sec_v))
        for s, t2, v1, r1, v2, r2 in rows:
            print("    seed %d  Table 2 %s AUROC %s AUPRC %s sign %s | fixed %s | secondary %s" % (
                s, "n" if t2["variant"] == "nonneg" else "s", f3(t2["auroc"]), f3(t2["auprc"]), f3(t2["sign"]),
                ("%s AUROC %s AUPRC %s sign %s val %.2e" % ("n" if v1 == "nonneg" else "s", f3(r1["auroc"]), f3(r1["auprc"]), f3(r1["sign"]), r1["val"]))
                if v1 else "pending (" + "; ".join(r1) + ")",
                ("%s AUROC %s" % ("n" if v2 == "nonneg" else "s", f3(r2["auroc"]))) if v2 else "pending"))
        if lab.startswith("IEEE"):                                       # like-for-like: restart 1 of the default runs
            lf = []
            for ds, s in cs:
                cell = man[sel_key(lab, ds, s)]; v, r = pick([(v, cell[v]) for v in MEMBERS]); lf.append((v, r))
            print("    like-for-like (restart 1 of the default runs, the cell retrained with the fixed scale): picks %s AUROC %s" % (
                picks([v for v, _ in lf]), f3(mean([r["auroc"] for v, r in lf if v]))))
    print("\nVERDICTS (rule: |mean AUROC difference| <= %.3f and the same members picked):" % TOL)
    for name, v in verdicts:
        print("  %-22s %s" % (name, v))
    n_un = sum(v.startswith("unchanged") for _, v in verdicts); n_ch = sum(v.startswith("changed") for _, v in verdicts)
    print("unchanged %d, changed %d, pending %d of %d settings" % (n_un, n_ch, len(verdicts) - n_un - n_ch, len(verdicts)))
    print("\ndiverged runs (non-finite loss; results, not rerun): %d" % len(nfail))
    for x in nfail:
        print("  " + x)
    if rms:
        print("coupling RMS at the end of training (finished runs with the fixed scale): median %.4f, range %.4f-%.4f (n %d)" % (
            np.median(rms), min(rms), max(rms), len(rms)))


if __name__ == "__main__":
    main()
