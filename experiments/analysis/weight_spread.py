"""Spread of the true edge weights next to the recall of the 2-means cut (supplement, "Binary graphs").
For every main-table setting and data seed 1-3: the true weight of edge j -> i as it multiplies the coupling law in the
generator (Kuramoto K / k_i, GLV and Grid-118 the stored W_true, MixRelax / MixOsc / MixOsc-E the constant kappa), its
coefficient of variation and max / min over the true edges, the recall of hetndl.readout.binarize_2means applied to the
TRUE |W| (what the cut keeps when the weights are known exactly) and the recall of the same cut on our reported run
(experiments/runs/sel/selected.json). Ground truth enters only this description and the scores.
Run from the project root with PYTHONPATH=."""
import json, os, numpy as np
from hetndl.readout import binarize_2means
from experiments.analysis.baseline_table import B, PAPER_NAME, R, score_matrix, sel, sel_key
from hetndl.gen.kuramoto import load_dataset


def true_weights(ds):
    """|true weight| matrix W[j, i] (0 off the edges) and the name of its source."""
    z = np.load("experiments/data/%s.npz" % ds, allow_pickle=True); A = (z["A"] > 0).astype(float)
    meta = json.loads(str(z["meta"]))
    if "W_true" in z.files:
        return np.abs(z["W_true"]) * A, "W_true"
    if meta["system"] == "kuramoto":
        k = np.maximum(A.sum(0), 1.0)                                  # unsigned in-degree of the receiver i
        return A * (meta["K"] / k)[None, :], "K / k_i"
    return A * meta["kappa"], "kappa"


def recall(Bn, A):
    off = ~np.eye(len(A), dtype=bool)
    return float((Bn & A)[off].sum()) / max(int(A[off].sum()), 1)


def main():
    print("%-20s %4s %-8s %6s %7s %8s  %12s %11s" % ("setting", "seed", "weights", "edges", "CV", "max/min", "recall truth", "recall ours"))
    means = {}
    for lab, cells in B:
        name = PAPER_NAME[lab]; rows = []
        for ds, s in cells:
            g = sel_key(lab, ds, s)
            if g not in sel:
                continue
            A = load_dataset("experiments/data/%s.npz" % ds)["A"] > 0
            Wt, src = true_weights(ds); w = Wt[A]
            r_true = recall(binarize_2means(Wt), A)
            r_ours = recall(binarize_2means(score_matrix(os.path.relpath(sel[g]["run"], R))), A)
            rows.append((w.std() / w.mean(), w.max() / w.min(), r_true, r_ours))
            print("%-20s %4d %-8s %6d %7.3f %8.2f  %12.3f %11.3f" % (name, s, src, len(w), rows[-1][0], rows[-1][1], r_true, r_ours))
        means[name] = np.array(rows).mean(0)
    print("\nmeans over the data seeds")
    print("%-20s %7s %8s  %12s %11s" % ("setting", "CV", "max/min", "recall truth", "recall ours"))
    for name, v in means.items():
        print("%-20s %7.3f %8.2f  %12.3f %11.3f" % ((name,) + tuple(v)))
    v = np.array(list(means.values()))
    from scipy.stats import spearmanr
    print("\nSpearman over the %d settings: recall ours vs CV %+.3f, recall ours vs recall of the cut on the true weights %+.3f" % (
        len(v), spearmanr(v[:, 3], v[:, 0])[0], spearmanr(v[:, 3], v[:, 2])[0]))


if __name__ == "__main__":
    main()
