"""Worst case and mean rank of every method over the twelve synthetic settings of Table 2. Settings fixed before this script
was first run:
  * values: exactly the numbers of Table 2, i.e. baseline_table.py's per-setting means over data seeds 1-3 (ARNI = the
    better of its full / greedy variants per seed by AUROC on Kuramoto, a truth-picked advantage for ARNI as in Table 2;
    SGODE = the variant with the lower validation loss; COSINE / failed runs left out of their cell's mean);
  * worst case: per method the minimum over the twelve settings of the mean AUROC (and of the mean AUPRC);
  * mean rank: per setting the nine methods are ranked by mean AUROC rounded to three decimals, as printed in Table 2
    (rank 1 = highest; equal rounded values share the average of their ranks); the mean over the twelve settings.
    Same for AUPRC. Secondary, per data seed: the same rank rule on the 36 single-seed values (AUROC), where a method
    without a run on that seed is left out of that seed's ranking;
  * NetSim is not part of the twelve settings and is reported on its own line as in Table 2.
Reading: the paper states the best worst case (the lowest mean AUROC of HetNDL against the highest lowest mean AUROC of
the baselines), with the mean ranks in the supplement, whatever they are. Truth enters only through the scores themselves.
Run from the project root with PYTHONPATH=. ; writes nothing, prints the table (-> results/supplement/worst_case_rank.txt)."""
import numpy as np
from experiments.analysis.baseline_table import B, BASE, PAPER_NAME, sel, met, nogi_grid, nogi_path, one, score, sgode, cosine

METHODS = ("HetNDL",) + BASE


def cell_scores(lab, ds, s):
    """(AUROC, AUPRC) of every method on one data seed, None where missing: the same calls as baseline_table.main()."""
    g = ("ba/%s/add_z_signed_diff_lz0_seed%d" if lab.startswith("BA") else "%s/add_z_signed_diff_lz0_seed%d") % (ds, s)
    o = (sel[g]["auroc"], sel[g]["auprc"])
    nm = nogi_grid(s) if lab.startswith("IEEE") else met(nogi_path(lab, ds, s))
    bs = [score(nm)] + [one(b, ds) for b in ("arni", "granger", "nri", "aidd")] + [cosine(ds, s), one("mtgnn", ds), sgode(ds, s)]
    return [o] + bs


def ranks(v):
    """Average ranks (1 = highest) of the values v rounded to 3 decimals; NaN = absent, gets NaN."""
    v = np.round(np.asarray(v, float), 3); r = np.full(len(v), np.nan); ok = ~np.isnan(v)
    vals = v[ok]; rr = np.empty(len(vals))
    for k, x in enumerate(vals):
        rr[k] = (vals > x).sum() + ((vals == x).sum() + 1) / 2.0
    r[ok] = rr
    return r


def main():
    means = []; seed_rank = []
    for lab, cells in B:
        per = np.array([[np.nan if x is None else x[0] for x in cell_scores(lab, ds, s)] for ds, s in cells])          # AUROC
        prc = np.array([[np.nan if x is None else x[1] for x in cell_scores(lab, ds, s)] for ds, s in cells])
        means.append((PAPER_NAME[lab], np.nanmean(per, 0), np.nanmean(prc, 0)))
        seed_rank += [ranks(row) for row in per]
    A = np.array([m[1] for m in means]); P = np.array([m[2] for m in means])
    RA = np.array([ranks(r) for r in np.round(A, 3)]); RP = np.array([ranks(r) for r in np.round(P, 3)]); RS = np.array(seed_rank)
    print("Twelve synthetic settings of Table 2 (means over data seeds 1-3). Rank 1 = best; ties at 3 decimals share ranks.\n")
    print("%-22s %s" % ("setting (mean AUROC)", " ".join("%8s" % m for m in METHODS)))
    for name, a, _ in means:
        print("%-22s %s" % (name, " ".join("%8.3f" % x for x in a)))
    print()
    rows = [("lowest mean AUROC", A.min(0), "%8.3f"), ("setting of the lowest", None, None),
            ("mean rank, AUROC", RA.mean(0), "%8.2f"), ("lowest mean AUPRC", P.min(0), "%8.3f"), ("mean rank, AUPRC", RP.mean(0), "%8.2f"),
            ("mean rank per seed", np.nanmean(RS, 0), "%8.2f")]
    print("%-22s %s" % ("summary", " ".join("%8s" % m for m in METHODS)))
    for lab, v, f in rows:
        if v is None:
            print("%-22s %s" % (lab, " ".join("%8s" % means[int(k)][0].split(" ")[0][:8] for k in A.argmin(0))))
        else:
            print("%-22s %s" % (lab, " ".join(f % x for x in v)))
    others = A.min(0)[1:]
    print("\nHetNDL lowest mean AUROC %.3f; highest lowest among the eight baselines %.3f (%s)." % (
        A.min(0)[0], others.max(), BASE[int(others.argmax())]))
    order = np.argsort(RA.mean(0))
    print("Order by mean AUROC rank: %s" % ", ".join("%s %.2f" % (METHODS[k], RA.mean(0)[k]) for k in order))
    print("Settings where HetNDL is not ranked first (AUROC): %s" % ", ".join(
        "%s (rank %.1f)" % (means[k][0], RA[k, 0]) for k in range(len(means)) if RA[k, 0] > 1.0))


if __name__ == "__main__":
    main()
