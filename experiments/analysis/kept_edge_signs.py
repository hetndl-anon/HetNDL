"""Signs of the true edges that the 2-means cut keeps, in all twelve main settings (Sect. 4.6 and supplement, Binary
Graphs). For every data seed 1-3: our reported run (runs/sel/selected.json), the cut hetndl.readout.binarize_2means on |W|
of the protocol snapshot, the predicted sign = sign of W after the model's own sign convention
(binary_graph_table.signed_scores). Counted: true edges kept by the cut and how many of them have the true sign. Ground
truth (S) enters only the count. Run from the repository root with PYTHONPATH=."""
import os, sys
import numpy as np
from hetndl.readout import binarize_2means
from hetndl.gen.kuramoto import load_dataset
from experiments.analysis.baseline_table import B, PAPER_NAME, R, sel, sel_key, score_matrix
from experiments.analysis.binary_graph_table import signed_scores

if __name__ == "__main__":
    lines, tot = [], [0, 0]
    for lab, cells in B:
        kept = ok = 0
        for ds, s in cells:
            g = sel_key(lab, ds, s)
            if g not in sel:
                continue
            S = load_dataset("experiments/data/%s.npz" % ds)["S"]
            path = os.path.relpath(sel[g]["run"], R)
            Bn = binarize_2means(score_matrix(path)); Ws, _ = signed_scores("ours", path)
            m = Bn & (S != 0); kept += int(m.sum()); ok += int((np.sign(Ws[m]) == np.sign(S[m])).sum())
        tot[0] += kept; tot[1] += ok
        lines.append("%-20s true edges kept by the cut %5d, with the correct sign %5d" % (PAPER_NAME[lab], kept, ok))
    lines.append("all twelve settings: %d kept, %d with the correct sign" % tuple(tot))
    txt = "\n".join(lines); print(txt)
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(txt + "\n")
