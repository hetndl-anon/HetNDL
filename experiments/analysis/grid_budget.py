"""Equal training budget on Grid-118. HetNDL and NOGI report the best of three training runs by validation error on
Grid-118 (train seeds s, 10 s + 2, 10 s + 3; the seed also sets the trajectory split). NRI, AIDD, MTGNN, SGODE (both
variants) and COSINE get the same two extra runs (grid_budget/... of experiments/reproduce/jobs.json: every other
argument as in the run of seed s). Per method and data seed the reported run is the one with the lowest value of the
method's own validation criterion over its runs: NRI / AIDD val_nll_best, MTGNN val_mae_best, SGODE val_l1_best over
both variants and the three seeds, COSINE val_score (its round choice). Failed runs are left out of the choice. ARNI
and Granger are deterministic fits and keep their single run. baseline_table.grid_pick() applies the same rule for
Table 2; this script checks that both agree.
Output: the Grid-118 row of Table 2 with one training run per neural baseline and with the best of three, and every
count that depends on Table 2, recomputed with the code of baseline_table.py / worst_case_rank.py (cell_scores, ranks;
ties within 1e-3 for W/T/L): the W/T/L of the Grid-118 row and of the twelve settings ("best or tied on N of the 36
data seeds"), every method's lowest mean AUROC and AUPRC, the mean ranks; plus HetNDL's mean AUROC on Grid-118 when only
its first training run (seed s) is used (both candidates, the lower validation error), quoted in the supplement's
section on training restarts. Truth enters only through the scores.
Run from the repository root with PYTHONPATH=. (-> results/supplement/grid_budget.txt)."""
import json, os
import numpy as np
from hetndl.snapshot import metrics_of
import experiments.analysis.baseline_table as bt
from experiments.analysis.baseline_table import B, BASE, PAPER_NAME, sel
from experiments.analysis.worst_case_rank import METHODS, cell_scores, ranks
R = "experiments/runs"
CRIT = {"nri": "val_nll_best", "aidd": "val_nll_best", "mtgnn": "val_mae_best", "sgode": "val_l1_best", "cosine": "val_score"}
COL = {"nri": 4, "aidd": 5, "cosine": 6, "mtgnn": 7, "sgode": 8}           # index in cell_scores(): 0 HetNDL, 1 NOGI, ...


def load(p):
    p = os.path.join(R, p, "metrics.json")
    if not os.path.exists(p):
        return "missing"
    m = json.load(open(p))
    return "failed" if m.get("failed") else m


def candidates(method, s):
    ds = "grid118_seed%d" % s; seeds = (s, 10 * s + 2, 10 * s + 3)
    if method == "sgode":
        paths = ["sgode/%s/%s/seed%d" % (v, ds, k) if k == s else "grid_budget/sgode/%s/%s/seed%d" % (v, ds, k)
                 for v in ("v1", "v3") for k in seeds]
    else:
        paths = ["%s/%s/seed%d" % (method, ds, k) if k == s else "grid_budget/%s/%s/seed%d" % (method, ds, k) for k in seeds]
    return [(p, load(p)) for p in paths]


def pick(method, s):
    c = candidates(method, s)
    if any(m == "missing" for _, m in c):
        return None, c
    ok = [(m[CRIT[method]], p, m) for p, m in c if m != "failed" and m.get(CRIT[method]) is not None and np.isfinite(m[CRIT[method]])]
    return (min(ok, key=lambda x: x[0]) if ok else ("failed", None, None)), c


def summary(get):
    means, seed_rank, wtl = [], [], np.zeros(3, int); per_setting = {}
    for lab, cells in B:
        rows = [get(lab, ds, s) for ds, s in cells]
        per = np.array([[np.nan if x is None else x[0] for x in r] for r in rows]); prc = np.array([[np.nan if x is None else x[1] for x in r] for r in rows])
        means.append((PAPER_NAME[lab], np.nanmean(per, 0), np.nanmean(prc, 0))); seed_rank += [ranks(r) for r in per]
        best = np.nanmax(per[:, 1:], 1); d = per[:, 0] - best
        w = np.array([(d > 1e-3).sum(), (abs(d) <= 1e-3).sum(), (d < -1e-3).sum()]); wtl += w; per_setting[PAPER_NAME[lab]] = w
    A = np.array([m[1] for m in means]); P = np.array([m[2] for m in means])
    RA = np.array([ranks(r) for r in np.round(A, 3)]); RP = np.array([ranks(r) for r in np.round(P, 3)])
    return dict(means=means, A=A, P=P, RA=RA.mean(0), RP=RP.mean(0), RS=np.nanmean(np.array(seed_rank), 0), wtl=wtl, per=per_setting)


def main():
    picks = {}
    print("Grid-118: best of three training runs by each method's own validation criterion\n")
    for method in COL:
        for s in (1, 2, 3):
            pk, c = pick(method, s)
            state = "  ".join("%s:%s" % (p.split("/")[-1] if "sgode" not in p else p.split("/")[-3] + "/" + p.split("/")[-1],
                                         m if isinstance(m, str) else "%.4g->%.3f" % (m[CRIT[method]] if m.get(CRIT[method]) is not None else float("nan"), m["auroc"]))
                              for p, m in c)
            if pk is None:
                print("%-6s seed %d  MISSING RUNS  %s" % (method, s, state)); continue
            if pk[0] == "failed":
                print("%-6s seed %d  all runs failed  %s" % (method, s, state)); picks[(method, s)] = None; continue
            picks[(method, s)] = (pk[2]["auroc"], pk[2]["auprc"])
            assert bt.grid_pick(method, s) == pk[1], (method, s, bt.grid_pick(method, s), pk[1])   # same rule as the main table
            print("%-6s seed %d  pick %-28s AUROC %.3f AUPRC %.3f   [%s criterion -> AUROC: %s]" % (
                method, s, pk[1], pk[2]["auroc"], pk[2]["auprc"], CRIT[method], state))
    if len(picks) < 15:
        print("\nincomplete: %d of 15 method-seed cells have all their runs" % len(picks)); return

    def new(lab, ds, s):
        x = list(cell_scores(lab, ds, s))
        if lab.startswith("IEEE"):
            for method, k in COL.items():
                x[k] = picks[(method, s)]
        return x
    bt.GRID_BUDGET = False; old_s = summary(cell_scores)          # one training run per neural baseline
    bt.GRID_BUDGET = True; new_s = summary(new)                   # best of three (Table 2)
    gi = [k for k, (lab, _) in enumerate(B) if lab.startswith("IEEE")][0]
    print("\nGrid-118 row (mean AUROC / AUPRC over seeds 1-3):")
    for tag, S in (("one run      ", old_s), ("best of three", new_s)):
        print("  %s  %s   W/T/L %s" % (tag, "  ".join("%s %.3f / %.3f" % (m, a, p) for m, a, p in zip(METHODS, S["A"][gi], S["P"][gi])),
                                     "/".join(map(str, S["per"]["Grid-118"]))))
    print("\nCounts that depend on Table 2 (one run -> best of three on Grid-118):")
    print("  twelve settings W/T/L: %s -> %s; best or tied on %d -> %d of 36 data seeds" % (
        "/".join(map(str, old_s["wtl"])), "/".join(map(str, new_s["wtl"])), old_s["wtl"][:2].sum(), new_s["wtl"][:2].sum()))
    for tag, key, f in (("lowest mean AUROC", "A", "%.3f"), ("lowest mean AUPRC", "P", "%.3f")):
        o, n = old_s[key].min(0), new_s[key].min(0)
        print("  %-18s %s" % (tag, "  ".join(("%s " + f + " -> " + f) % (m, a, b) for m, a, b in zip(METHODS, o, n))))
    for tag, key in (("mean rank AUROC", "RA"), ("mean rank AUPRC", "RP"), ("mean rank per seed", "RS")):
        print("  %-18s %s" % (tag, "  ".join("%s %.2f -> %.2f" % (m, a, b) for m, a, b in zip(METHODS, old_s[key], new_s[key]))))
    oth_old, oth_new = old_s["A"].min(0)[1:], new_s["A"].min(0)[1:]
    print("  HetNDL lowest %.3f -> %.3f; highest lowest among the baselines %.3f (%s) -> %.3f (%s)" % (
        old_s["A"].min(0)[0], new_s["A"].min(0)[0], oth_old.max(), BASE[int(oth_old.argmax())], oth_new.max(), BASE[int(oth_new.argmax())]))
    for m, k in (("SGODE", 8), ("MTGNN", 7), ("COSINE", 6), ("NRI", 4), ("AIDD", 5)):
        print("  %-6s Grid-118 %.3f -> %.3f; range over the eight non-Kuramoto settings %.3f-%.3f -> %.3f-%.3f" % (
            m, old_s["A"][gi, k], new_s["A"][gi, k],
            *(lambda v: (v.min(), v.max()))(np.array([a[k] for (nm, a, _) in old_s["means"] if not nm.startswith("Kur")])),
            *(lambda v: (v.min(), v.max()))(np.array([a[k] for (nm, a, _) in new_s["means"] if not nm.startswith("Kur")]))))
    # HetNDL with its first training run only (seed s; both candidates; lower validation error)
    man = json.load(open(os.path.join(R, "sel", "manifest.json"))); first = []
    for s in (1, 2, 3):
        e = man["grid118_seed%d/add_z_signed_diff_lz0_seed%d" % (s, s)]
        c = [(m["val_mse_best"], metrics_of(m)["auroc"]) for m in (json.load(open(os.path.join(e[v], "metrics.json"))) for v in ("nonneg", "nowu")) if not m.get("failed")]
        first.append(min(c)[1])
    full = [sel["grid118_seed%d/add_z_signed_diff_lz0_seed%d" % (s, s)]["auroc"] for s in (1, 2, 3)]
    print("\nHetNDL on Grid-118: first training run only %s, mean %.3f; best of three (reported) %s, mean %.3f" % (
        " ".join("%.3f" % x for x in first), np.mean(first), " ".join("%.3f" % x for x in full), np.mean(full)))


if __name__ == "__main__":
    main()
