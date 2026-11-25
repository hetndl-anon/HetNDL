"""Contingency of HetNDL's class read-out (the node_table.py path) against the true classes on MixOsc N=60, Grid-118 and
MixRelax (BA): which clusters mix true classes, how many nodes lie in a cluster whose majority class is their own, and
which nodes are misassigned on Grid-118 (truth only in this check). Run from the project root with
PYTHONPATH=. (-> experiments/results/review/class_contingency.txt)."""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from hetndl.edge_readout import load_run
from experiments.analysis.baseline_table import B, PAPER_NAME, sel, sel_key
from experiments.analysis.readout_split_check import readout
for lab, cells in B:
    name = PAPER_NAME[lab]
    if name not in ("MixOsc, $N=60$", "MixOsc N=60", "Grid-118", "MixRelax (BA)"):
        continue
    for ds, s in cells:
        m, d, vf, W = load_run(sel[sel_key(lab, ds, s)]["run"]); X = np.asarray(d["data"], np.float32)
        K, l, _ = readout(vf, X, d["meta"]["system"]); cls = np.asarray(d["classes"])
        tab = np.array([[np.sum((cls == c) & (l == k)) for k in np.unique(l)] for c in np.unique(cls)])
        mixed = [k for j, k in enumerate(np.unique(l)) if (tab[:, j] > 0).sum() > 1]
        print("%-12s seed %d K %d ARI %.3f | rows = true classes %s, cols = read-out clusters:\n%s\n  clusters with more than one true class: %s"
              % (name, s, K, adjusted_rand_score(cls, l), list(np.unique(cls)), tab, mixed))
        print("  nodes in a cluster whose majority class is their own: %d of %d" % (tab.max(0).sum(), len(cls)))
        if name == "Grid-118":
            maj = {k: np.bincount(cls[l == k]).argmax() for k in np.unique(l)}
            wrong = [i for i in range(len(cls)) if maj[l[i]] != cls[i]]
            print("  misassigned nodes (index, true class):", [(i, int(cls[i])) for i in wrong])
