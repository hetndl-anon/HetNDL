"""MTGNN scoring: the single code path for the MTGNN graph scores, used by experiments/launch/mtgnn_run.py and,
for runs finished before this rule, by this script (from the saved W.npy = A^T; no retraining).

Why symmetrised: MTGNN's graph constructor (layer.graph_constructor) builds A = ReLU(tanh(alpha (M1 M2^T - M2 M1^T))),
which is antisymmetric before the ReLU, so every node pair is non-zero in at most one direction, and the network
propagates through both A and A^T (net.gtnet: gconv1(x, A) + gconv2(x, A^T)). The learned adjacency therefore encodes
"i and j interact", not a direction; scoring it per direction discards half of it. We score S = W + W^T (identical to
A + A^T), which favours MTGNN; the per-direction scores are kept as auroc_directed / auprc_directed. Chosen
after the first runs had been scored per direction (as-is mean AUROC 0.47-0.65 vs 0.39-0.98 symmetrised); the reason is
architectural and the change is in the baseline's favour.
Usage (project root, PYTHONPATH=.): python experiments/analysis/mtgnn_rescore.py   (rescores every runs/mtgnn run)"""
import glob, json, os
import numpy as np
from hetndl.metrics import auroc_auprc


def mtgnn_scores(W, A_true):
    au_d, ap_d = auroc_auprc(W, A_true)
    au, ap = auroc_auprc(W + W.T, A_true)
    return dict(auroc=au, auprc=ap, auroc_directed=au_d, auprc_directed=ap_d, scoring="symmetrised W + W^T")


def main():
    from hetndl.gen.kuramoto import load_dataset
    for p in sorted(glob.glob("experiments/runs/mtgnn/*/seed*/metrics.json")):
        m = json.load(open(p)); d = os.path.dirname(p)
        if m.get("scoring", "").startswith("symmetrised") or not os.path.exists(os.path.join(d, "W.npy")):
            continue
        W = np.load(os.path.join(d, "W.npy")); A = load_dataset(m["args"]["data"])["A"]
        m.update(mtgnn_scores(W, A)); json.dump(m, open(p, "w"), indent=1, default=float)
        print("rescored", d, "%.3f (directed %.3f)" % (m["auroc"], m["auroc_directed"]))


if __name__ == "__main__":
    main()
