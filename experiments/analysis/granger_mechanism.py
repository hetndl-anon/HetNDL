"""Can the linear Granger baseline also read the mechanism classes? (Lasso-VAR nearly matches our topology
on A1 / A2 / A3 / grid.) Refit the lasso at the validation-selected alpha stored in metrics.json, take each node's
self block (target i <- source i, all state dims, lag 1) as its feature, cluster with the truth-free select_k and with
the true K, report ARI. Run from the project root with PYTHONPATH=.; args: granger run dirs."""
import json, sys, numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.metrics import adjusted_rand_score
from hetndl.baselines.granger import lagged_design
from hetndl.data import add_noise, select_trajectories, trajectory_split
from hetndl.gen.kuramoto import load_dataset
from hetndl.readout import select_k
for run in sys.argv[1:]:
    m = json.load(open(run + "/metrics.json")); a = m["args"]; d = load_dataset(a["data"]); c = d["classes"]
    keep = select_trajectories(d["data"].shape[0], a.get("n_traj"), a["seed"])
    obs = add_noise(d["data"][keep], np.random.default_rng(a["seed"] + 12345), a.get("noise_snr"), a.get("noise_std"))
    tr, _, _ = trajectory_split(len(keep), a["seed"]); X, Y = lagged_design(obs, tr, m["lags"])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    coef = Lasso(alpha=m["alpha"], max_iter=20000, tol=1e-3).fit((X - mu) / sd, Y).coef_ / sd[None, :]   # raw units
    N, dim = obs.shape[2], obs.shape[3]
    C = coef.reshape(N, dim, m["lags"], N, dim)
    feat = np.stack([C[i, :, 0, i, :].ravel() for i in range(N)])                     # (N, dim*dim) self block
    K_true = len(np.unique(c)); K_sel, lab, _ = select_k(feat, k_max=5)
    km = KMeans(K_true, n_init=20, random_state=0).fit_predict((feat - feat.mean(0)) / (feat.std(0) + 1e-12))
    print("%s | Granger self-block ARI: true K %.3f | K_sel %d ARI %.3f" % (run.split("runs/")[1], adjusted_rand_score(c, km), K_sel, adjusted_rand_score(c, lab)))
