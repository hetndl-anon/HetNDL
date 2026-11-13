"""Class rows of Table 3 (node-level read-outs), printed as LaTeX so that no number is typed by hand.
Rows: MixRelax, MixRelax (BA), MixOsc N=30, MixOsc N=60, Grid-118; data seeds 1-3 (MixOsc N=30: PAPER_SEEDS). Cell = mean
ARI over seeds (minimum in parentheses), K chosen from the data, truth only in the ARI.
  HetNDL   the paper's read-out (readout_split_check.readout, all trajectories) of the reported run;
  Granger  its lag-one self block clustered with select_k (granger_mechanism.py, same refit at the stored alpha);
  ARNI     its fitted vector field with all sources at the node's state, through HetNDL's read-out (variant B of
           arni_same_readout.py, the like-for-like quantity fixed there before any result).
The HetNDL and Granger means are asserted against the values of the single-method scripts (OLD). The Kuramoto |rho| rows
of Table 3 come from omega_readout.py and granger_omega.py. Run from the project root with PYTHONPATH=.
(-> results/table3_classes.txt)."""
import json, os, numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import adjusted_rand_score
from hetndl.baselines.granger import lagged_design
from hetndl.data import add_noise, select_trajectories, trajectory_split
from hetndl.edge_readout import load_run
from hetndl.gen.kuramoto import load_dataset
from hetndl.readout import select_k
from experiments.analysis.baseline_table import B, PAPER_NAME, sel, sel_key
from experiments.analysis.readout_split_check import readout
from experiments.analysis.rollout_baselines import arni_fit
from experiments.analysis.arni_same_readout import ArniField
ROWS = (("MixRelax", "MixRelax"), ("MixRelax (BA)", "MixRelax (BA)"), ("MixOsc N=30", r"MixOsc, $N=30$"),
        ("MixOsc N=60", r"MixOsc, $N=60$"), ("Grid-118", "Grid-118"))
OLD = {"MixRelax": (1.000, 0.000), "MixOsc N=30": (1.000, 1.000), "MixOsc N=60": (0.916, 0.873), "Grid-118": (0.966, 0.762)}


def granger_ari(ds, cls):
    run = [os.path.join("experiments/runs/granger", ds, x) for x in sorted(os.listdir(os.path.join("experiments/runs/granger", ds)))][0]
    m = json.load(open(run + "/metrics.json")); a = m["args"]; d = load_dataset(a["data"])
    keep = select_trajectories(d["data"].shape[0], a.get("n_traj"), a["seed"])
    obs = add_noise(d["data"][keep], np.random.default_rng(a["seed"] + 12345), a.get("noise_snr"), a.get("noise_std"))
    tr, _, _ = trajectory_split(len(keep), a["seed"]); X, Y = lagged_design(obs, tr, m["lags"])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    coef = Lasso(alpha=m["alpha"], max_iter=20000, tol=1e-3).fit((X - mu) / sd, Y).coef_ / sd[None, :]
    N, dim = obs.shape[2], obs.shape[3]; C = coef.reshape(N, dim, m["lags"], N, dim)
    feat = np.stack([C[i, :, 0, i, :].ravel() for i in range(N)])
    return adjusted_rand_score(cls, select_k(feat, k_max=5)[1])


def main():
    res = {}
    for lab, cells in B:
        name = PAPER_NAME[lab]
        if name not in dict(ROWS):
            continue
        for ds, s in cells:
            m, d, vf, W = load_run(sel[sel_key(lab, ds, s)]["run"]); X = np.asarray(d["data"], np.float32)
            system, cls = d["meta"]["system"], d["classes"]
            K, l, _ = readout(vf, X, system); h = adjusted_rand_score(cls, l)
            basis, fits, _, _, _ = arni_fit(os.path.join("experiments/runs/arni", ds, "full_seed%d" % s), load_dataset("experiments/data/%s.npz" % ds))
            Ka, la, _ = readout(ArniField(basis, fits, X.shape[-1], "B"), X, system)
            g = granger_ari(ds, cls)
            res.setdefault(name, []).append((h, g, adjusted_rand_score(cls, la), K))
            print("%-14s seed %d  HetNDL K %d ARI %.3f | Granger %.3f | ARNI (our read-out) K %d ARI %.3f" % (name, s, K, h, g, Ka, res[name][-1][2]), flush=True)
    print()
    for name, tex in ROWS:
        v = np.array([r[:3] for r in res[name]]); mean, mn = v.mean(0), v.min(0)
        if name in OLD:
            assert abs(mean[0] - OLD[name][0]) < 5e-4 and abs(mean[1] - OLD[name][1]) < 5e-4, (name, mean, OLD[name])
        best = max(round(x, 3) for x in mean)
        cell = lambda k: ("\\textbf{%.3f} (%.3f)" if round(mean[k], 3) == best else "%.3f (%.3f)") % (mean[k], mn[k])
        print("%-17s & ARI, classes & %s & %s & %s\\\\   %% K per seed: %s" % (tex, cell(0), cell(1), cell(2), ",".join(str(r[3]) for r in res[name])))


if __name__ == "__main__":
    main()
