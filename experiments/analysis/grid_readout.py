"""IEEE-118 swing-equation read-out for trained runs (protocol snapshot via hetndl.snapshot).
Per run: AUROC / AUPRC; learned W vs W_true on the true edges (Pearson, Spearman, best scalar alpha, relative RMSE), and
whether W tracks the 1/M_i scaling (corr with W_true) better than the bare line susceptance K_ij (corr with W_true*M_i);
mechanism read-out from the self term F((theta, omega), z_i): with diffusive coupling F_omega = (P_i - D omega)/M_i
exactly, so the slope dF_omega/d omega = -D/M_i separates generators (M = 1) from loads (M = 0.2) -> KMeans ARI vs the
true classes, and the intercept tracks P_i/M_i -> Spearman with P_i/M_i; z-cluster ARI from metrics for comparison.
Run from the project root with PYTHONPATH=.; args: run dirs."""
from hetndl.snapshot import metrics_of, state_path, array_path
from hetndl.readout import select_k
import json, sys, numpy as np, torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from hetndl.gen.kuramoto import load_dataset
from hetndl.model import build

for run in sys.argv[1:]:
    m = json.load(open(run + "/metrics.json")); a = m["args"]; b = metrics_of(m)
    d = load_dataset(a["data"]); meta = d["meta"]; A, Wt, cls, P = d["A"], d["W_true"], d["classes"], d["omega"]
    N = len(cls); M = np.where(cls == 1, meta["m_gen"], meta["m_load"]); D = meta["damping"]
    W = np.load(array_path(run, "W", m)); W = np.abs(W) if a["model"] == "orig" else W
    e = A > 0; w, wt = W[e], Wt[e]
    alpha = float((w * wt).sum() / max((w * w).sum(), 1e-12))
    rel = float(np.sqrt(np.mean((alpha * w - wt) ** 2)) / np.sqrt(np.mean(wt ** 2)))
    K = Wt * M[None, :]                                           # undo the 1/M_i column scaling -> kappa * K_ij (symmetric)
    r_wt, r_k = pearsonr(w, wt)[0], pearsonr(w, K[e])[0]
    line = "%s | AUROC %.3f AUPRC %.3f | edges: Pearson(W,W_true) %.3f  Pearson(W,K) %.3f  Spearman %.3f  alpha %.3g  relRMSE %.3f" % (
        run.split("runs/")[-1], b["auroc"], b["auprc"], r_wt, r_k, spearmanr(w, wt).correlation, alpha, rel)
    if a["model"] == "orig":
        print(line + " | no self-term read-out (NOGI)"); continue
    dt = float(meta["dt"])
    gp, ode = build(a["model"], N, 2, a["hidden"], a["d_z"], dt, a.get("rtol", 1e-3), a.get("atol", 1e-3), a.get("method", "rk4"),
                    bool(a["mask_diag"]), feat=a.get("feat", "raw"), coupling=a.get("coupling", "free"),
                    z_norm=a.get("z_norm", "none"), edge_z=bool(a.get("edge_z", 0)))
    st = torch.load(state_path(run, m), map_location="cpu"); gp.load_state_dict(st["gp"]); ode.load_state_dict(st["vf"]); vf = ode.vf
    X = d["data"]                                                 # (S, T, N, 2)
    th_mean = X[..., 0].mean((0, 1)); om = np.linspace(np.percentile(X[..., 1], 2), np.percentile(X[..., 1], 98), 33)
    slope, icpt = np.zeros(N), np.zeros(N)
    with torch.no_grad():
        for i in range(N):
            g = torch.tensor(np.stack([np.full_like(om, th_mean[i]), om], -1), dtype=torch.float32)
            h = vf.phi(g)
            f = vf.F(torch.cat([h, vf.z_eff()[i].expand(len(g), -1)], -1)).numpy()[:, 1] if vf.use_z else vf.F(h).numpy()[:, 1]
            slope[i], icpt[i] = np.polyfit(om, f, 1)
    km = KMeans(2, n_init=20, random_state=0).fit_predict(slope[:, None])      # true K (reference only)
    ari_slope = adjusted_rand_score(cls, km)
    K_sel, lab_sel, _ = select_k(slope[:, None], k_max=5)                       # truth-free K (protocol)
    true_slope = -D / M
    print(line + " | F_omega slope: ARI(gen/load, true K) %.3f  K_sel %d ARI(K_sel) %.3f  slope gen %.2f (true %.2f) load %.2f (true %.2f) | Spearman(intercept, P/M) %.3f | z-ARI %.3f  rollout %.3f" % (
        ari_slope, K_sel, adjusted_rand_score(cls, lab_sel), slope[cls == 1].mean(), true_slope[cls == 1][0], slope[cls == 0].mean(), true_slope[cls == 0][0],
        spearmanr(icpt, P / M).correlation, b.get("cluster_ari", float("nan")), b["rollout_nrmse"]))
