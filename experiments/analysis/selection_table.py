"""Per-seed view of the candidate selection (read-only, writes nothing). For every main-table setting and
data seed 1-3: the validation MSE of the two candidates (W >= 0 and signed W, no warm-up) from
experiments/runs/sel/manifest.json, their ratio, the candidate with the lower validation MSE, and the AUROC / AUPRC of
each candidate's protocol snapshot. Grid-118 has three restarts per candidate; the restart with the lowest validation MSE
represents the candidate, as in sel_tables.py and sign_member_table.py. The pick is recomputed here and compared with
experiments/runs/sel/selected.json (written by sel_tables.py). Then, per setting: mean AUROC / AUPRC of "always W >= 0",
"always signed" and the selection, and whether the selection reaches the better of the two fixed choices.
Validation MSE is truth-free; AUROC / AUPRC and the flag "the true graph has inhibitory edges" (has_neg_edges, used only
to group the summary counts) are scores. For an all-excitatory seed on which the signed candidate was selected, the signs
of the selected model's weights are listed (edges of the 2-means cut and true edges). Run from the project root with PYTHONPATH=."""
import collections, json, os, re, numpy as np
from hetndl.snapshot import metrics_of
from hetndl.readout import binarize_2means
from hetndl.gen.kuramoto import load_dataset
from experiments.analysis.baseline_table import B, PAPER_NAME, R, score_matrix, sel, sel_key
man = json.load(open(os.path.join(R, "sel", "manifest.json")))
NAME = {"nonneg": "non-negative", "nowu": "signed"}


def main():
    groups = collections.defaultdict(list)                                 # grid restarts _r2 / _r3 belong to one cell
    for cell, e in man.items():
        groups[re.sub(r"_r\d+$", "", cell)].append(e)
    print("%-20s %4s  %11s %11s %9s  %-12s %-9s  %-15s %-15s" % (
        "setting", "seed", "val W>=0", "val signed", "sgn/nonn", "selected", "selected", "W>=0 candidate", "signed candidate"))
    print("%-20s %4s  %11s %11s %9s  %-12s %-9s  %-15s %-15s" % ("", "", "", "", "", "", ".json", "AUROC / AUPRC", "AUROC / AUPRC"))
    summ, counts, agree, special = {}, collections.Counter(), True, []
    for lab, cells in B:
        name = PAPER_NAME[lab]; acc = collections.defaultdict(list)
        for ds, s in cells:
            g = sel_key(lab, ds, s); best = {}
            for v in NAME:
                c = [json.load(open(os.path.join(e[v], "metrics.json"))) for e in groups[g]]
                assert not any(m.get("failed") for m in c), g
                best[v] = min(c, key=lambda m: m["val_mse_best"])
            pick = min(best, key=lambda v: best[v]["val_mse_best"]); same = pick == sel[g]["variant"] and abs(best[pick]["val_mse_best"] - sel[g]["val"]) < 1e-18
            agree &= same
            sc = {v: (metrics_of(best[v])["auroc"], metrics_of(best[v])["auprc"]) for v in NAME}
            for v in NAME:
                acc[v].append(sc[v])
            acc["sel"].append(sc[pick]); acc["pick"].append(pick[0] if pick == "nonneg" else "s")
            neg = bool(metrics_of(best[pick])["has_neg_edges"]); counts[(neg, pick)] += 1
            if not neg and pick == "nowu":
                special.append("%s seed %d: val MSE W>=0 %.3e, signed %.3e (signed / W>=0 = %.3f); AUROC / AUPRC W>=0 %.3f / %.3f, signed %.3f / %.3f" % (
                    name, s, best["nonneg"]["val_mse_best"], best["nowu"]["val_mse_best"], best["nowu"]["val_mse_best"] / best["nonneg"]["val_mse_best"],
                    *sc["nonneg"], *sc["nowu"]))
                gq = metrics_of(best[pick]).get("gauge", 1); W = score_matrix(os.path.relpath(sel[g]["run"], R)) * (gq if isinstance(gq, int) else 1)
                A = load_dataset("experiments/data/%s.npz" % ds)["A"] > 0; Bn = binarize_2means(W)
                special.append("    selected model: 2-means cut keeps %d pairs (%d true edges), of which with a negative weight %d; true edges with a "
                               "negative weight %d of %d; share of the summed |W| on negative entries %.4f" % (
                                   int(Bn.sum()), int((Bn & A).sum()), int((W[Bn] < 0).sum()), int((W[A] < 0).sum()), int(A.sum()),
                                   float(np.abs(W[W < 0]).sum() / np.abs(W).sum())))
            print("%-20s %4d  %11.3e %11.3e %9.3g  %-12s %-9s  %.3f / %.3f   %.3f / %.3f" % (
                name, s, best["nonneg"]["val_mse_best"], best["nowu"]["val_mse_best"], best["nowu"]["val_mse_best"] / best["nonneg"]["val_mse_best"],
                NAME[pick], "same" if same else "DIFFERENT", *sc["nonneg"], *sc["nowu"]))
        summ[name] = acc
    print("\nrecomputed picks %s experiments/runs/sel/selected.json" % ("all agree with" if agree else "DO NOT all agree with"))
    print("settings with inhibitory edges: %d seeds, signed selected on %d; all-excitatory settings: %d seeds, non-negative selected on %d, signed on %d" % (
        counts[(True, "nowu")] + counts[(True, "nonneg")], counts[(True, "nowu")], counts[(False, "nowu")] + counts[(False, "nonneg")],
        counts[(False, "nonneg")], counts[(False, "nowu")]))
    print("all-excitatory seeds on which the signed candidate was selected:")
    for x in special:
        print("  " + x)
    print("\nmean over the data seeds, AUROC / AUPRC (picks: n = non-negative, s = signed)")
    print("%-20s %-6s %-15s %-15s %-15s %s" % ("setting", "picks", "always W>=0", "always signed", "selection", "selection vs the better fixed choice (AUROC)"))
    ok = 0
    for name, acc in summ.items():
        mn, ms, md = (np.mean(acc[k], 0) for k in ("nonneg", "nowu", "sel")); better = max(mn[0], ms[0]); d = md[0] - better
        ok += d >= -5e-4
        print("%-20s %-6s %.3f / %.3f   %.3f / %.3f   %.3f / %.3f   %s (difference %+.1e)" % (
            name, "".join(acc["pick"]), *mn, *ms, *md, "equal or higher" if d >= 0 else ("equal to 3 decimals" if d >= -5e-4 else "LOWER"), d))
    print("the selection's mean AUROC reaches the better fixed choice (to 3 decimals) in %d of %d settings" % (ok, len(summ)))


if __name__ == "__main__":
    main()
