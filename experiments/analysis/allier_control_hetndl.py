"""HetNDL on the positive-control data of the Allier et al. 2026 baseline (experiments/launch/allier_control.py: data of their
model from their simulator, 100 neurons, 20 runs of 500 steps; coupling W_ij tanh(x_j), which reads the sender only, four
neuron types). Protocol = the default of every benchmark, nothing set for these data: the non-negative and the signed
candidate (add_z, add_z_signed; d_z 2, 1000 epochs, dopri5, diffusive coupling, training seed = data seed; the jobs
allier_control_hetndl/... of experiments/reproduce/jobs.json), the reported one has the lower validation MSE, snapshot by
validation MSE. Truth enters only the scores printed here:
  graph AUROC / AUPRC and sign accuracy on the true edges (as scored by hetndl.train at the snapshot), Pearson r between
  the learned and the true weights over all ordered pairs (weights in the model's sign gauge), node classes by the
  read-out of the scalar-state benchmarks (readout_split_check.readout, self-field curves, select_k) against the four
  neuron types, and the R2 of the in-strength (summed signed true weight into a node) on the first three principal
  components of the self-field curves inside the true types (how much of the within-type spread of the self-fields
  follows the in-strength; cf. allier_latents.py).
The rows of their method on the same data are in allier_control.txt (allier_control_table.py).
Run from the project root: PYTHONPATH=. python experiments/analysis/allier_control_hetndl.py"""
import json, os
import numpy as np
from sklearn.metrics import adjusted_rand_score
from hetndl.edge_readout import load_run
from experiments.analysis.readout_split_check import readout, node_fields

R = "experiments/runs/allier_control_hetndl"
VARIANTS = (("nonneg", "add_z"), ("signed", "add_z_signed"))


def r2_within(F, y, cl):
    res = tot = 0.0
    for c in np.unique(cl):
        m = cl == c
        X = np.c_[F[m], np.ones(m.sum())]
        b = np.linalg.lstsq(X, y[m], rcond=None)[0]
        res += ((y[m] - X @ b) ** 2).sum(); tot += ((y[m] - y[m].mean()) ** 2).sum()
    return 1.0 - res / tot


def main():
    print(__doc__.split("\n")[0])
    print("%-4s %-8s %-10s | %-15s %6s %7s | %-9s %s" % ("seed", "selected", "val MSE", "AUROC / AUPRC", "sign", "r(W)", "type ARI", "K; R2 of the in-strength on the self-field curves inside the types"))
    for s in (1, 2):
        ds = "allier_control_N100_seed%d" % s; cands = []
        for v, model in VARIANTS:
            run = os.path.join(R, ds, "%s_diff_lz0_seed%d" % (model, s))
            if os.path.exists(os.path.join(run, "metrics.json")):
                m = json.load(open(os.path.join(run, "metrics.json")))
                if not m.get("failed"):
                    cands.append((m["val_mse_best"], v, run))
        if len(cands) < len(VARIANTS):
            print("%-4d pending (%d of %d candidates finished)" % (s, len(cands), len(VARIANTS))); continue
        val, v, run = min(cands)
        m, d, vf, W = load_run(run); b = m["best"]
        z = np.load("experiments/data/%s.npz" % ds); off = ~np.eye(len(W), dtype=bool)
        g = b.get("gauge", 1); Wg = W * (g if isinstance(g, (int, float)) else 1)
        r = np.corrcoef(Wg[off], z["W_true"][off])[0, 1]
        X = np.asarray(d["data"], np.float32); cls = np.asarray(d["classes"])
        K, lab, _ = readout(vf, X, "a1")                               # the read-out of the scalar-state benchmarks
        lo, hi = np.percentile(X, [5, 95]); curves = node_fields(vf, np.linspace(lo, hi, 64)[:, None])[:, :, 0]
        C = curves - curves.mean(0); pcs = C @ np.linalg.svd(C, full_matrices=False)[2][:3].T
        rin = r2_within(pcs, np.asarray(z["W_true"], float).sum(0), cls)   # in-strength: summed signed weight into the node
        print("%-4d %-8s %-10.3e | %.3f / %.3f  %6.3f %7.3f | %-9.3f K %d; R2 %.3f   (candidates: %s)" % (
            s, v, val, b["auroc"], b["auprc"], b.get("sign_acc", float("nan")), r, adjusted_rand_score(cls, lab), K, rin,
            ", ".join("%s %.3e" % (c[1], c[0]) for c in sorted(cands, key=lambda c: c[1]))))


if __name__ == "__main__":
    main()
