"""Weight-weighted recall of the truth-free 2-means cut (Grid-118's true edge weights span a factor of about 500): for every main-table setting and data seed 1-3, with the edge set of
hetndl.readout.binarize_2means(|W|) of our reported run (experiments/runs/sel/selected.json, the cut of Sect. 4.6), print
  recall           = kept true edges / true edges (the recall of binary_graph_table.py),
  weight recall    = sum of the true weights of the kept true edges / sum of the true weights of all true edges,
  strongest half   = share of the stronger half of the true edges (by true weight) that the cut keeps,
and the same three numbers for the cut applied to the TRUE weights (what 2-means keeps when the weights are known).
True weights: weight_spread.true_weights (the weight of edge j -> i as it multiplies the coupling law in the generator:
Kuramoto K / k_i, GLV and Grid-118 the stored W_true, MixRelax / MixOsc / MixOsc-E the constant kappa). Ground truth
enters only these scores. Run from the project root: PYTHONPATH=. python experiments/analysis/weight_coverage.py
> results/supplement/weight_coverage.txt"""
import os
import numpy as np
from hetndl.readout import binarize_2means
from hetndl.gen.kuramoto import load_dataset
from experiments.analysis.baseline_table import B, PAPER_NAME, R, score_matrix, sel, sel_key
from experiments.analysis.weight_spread import true_weights


def scores(Bn, A, Wt):
    off = ~np.eye(len(A), dtype=bool); T = A & off; K = Bn & T
    w = Wt[T]; strong = Wt >= np.median(w)
    return (K.sum() / T.sum(), Wt[K].sum() / w.sum(), (K & strong & T).sum() / (strong & T).sum())


def main():
    print(__doc__.split("\n")[0])
    print("%-20s %4s %8s | %7s %8s %9s | %7s %8s %9s" % ("setting", "seed", "max/min", "recall", "w-recall", "strong 50%",
                                                         "recall", "w-recall", "strong 50%"))
    print("%-34s | %-27s | %s" % ("", "our run", "cut on the true weights"))
    means = {}
    for lab, cells in B:
        name = PAPER_NAME[lab]; rows = []
        for ds, s in cells:
            g = sel_key(lab, ds, s)
            if g not in sel:
                continue
            A = load_dataset("experiments/data/%s.npz" % ds)["A"] > 0
            Wt, _ = true_weights(ds)
            o = scores(binarize_2means(score_matrix(os.path.relpath(sel[g]["run"], R))), A, Wt)
            t = scores(binarize_2means(Wt), A, Wt)
            w = Wt[A & ~np.eye(len(A), dtype=bool)]
            rows.append((w.max() / w.min(),) + o + t)
            print("%-20s %4d %8.1f | %7.3f %8.3f %9.3f | %7.3f %8.3f %9.3f" % ((name, s) + rows[-1]))
        means[name] = np.mean(rows, 0)
    print("\nmeans over the data seeds")
    for name, v in means.items():
        print("%-20s      %8.1f | %7.3f %8.3f %9.3f | %7.3f %8.3f %9.3f" % ((name,) + tuple(v)))


if __name__ == "__main__":
    main()
