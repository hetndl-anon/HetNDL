"""A1 read-out: per-node learned F(x, z_i) vs the three true self-terms on the observed x range.
Reports z std, KMeans-on-F-curves ARI, nearest-true-curve accuracy, per-class mean |F - f_true| (after removing the
best per-run constant offset, since F's constant can trade with G's self-part under free coupling)."""
from hetndl.snapshot import metrics_of, state_path, array_path
from hetndl.readout import select_k
import json, sys, numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from hetndl.gen.kuramoto import load_dataset
from hetndl.gen.a1_mixed import self_term
from hetndl.model import build
for run in sys.argv[1:]:
    m = json.load(open(run + "/metrics.json")); a = m["args"]
    d = load_dataset(a["data"]); c = d["classes"]; N = len(c)
    gp, ode = build(a["model"], N, 1, a["hidden"], a["d_z"], 0.02, 1e-3, 1e-3, "rk4", bool(a["mask_diag"]),
                    feat=a.get("feat", "raw"), coupling=a.get("coupling", "free"), z_norm=a.get("z_norm", "none"),
                    edge_z=bool(a.get("edge_z", 0)))
    st = torch.load(state_path(run, m), map_location="cpu"); gp.load_state_dict(st["gp"]); ode.load_state_dict(st["vf"]); vf = ode.vf
    lo, hi = np.percentile(d["data"], [5, 95]); xs = np.linspace(lo, hi, 64)
    with torch.no_grad():
        h = vf.phi(torch.tensor(xs, dtype=torch.float32).view(-1, 1))
        Fc = np.stack([vf.F(torch.cat([h, vf.z_eff()[i].expand(64, -1)], -1))[:, 0].numpy() for i in range(N)])
    T = np.stack([self_term(xs, np.full(64, k)) for k in range(3)])          # [3, 64]
    nearest = np.argmin(((Fc[:, None] - T[None]) ** 2).mean(-1), 1)
    km = KMeans(3, n_init=20, random_state=0).fit_predict(Fc)      # true K (reference only)
    K_sel, lab_sel, _ = select_k(Fc, k_max=5, curves=True)                        # truth-free K (protocol)
    err = [float(np.abs(Fc[c == k] - T[k]).mean()) for k in range(3)]
    gap = [float(np.abs(T[i] - T[j]).mean()) for i, j in ((0, 1), (0, 2), (1, 2))]
    print("%s  coupling=%s  x∈[%.2f,%.2f]  z std %.3g | ARI(z,metrics) %.3f  ARI(F-curves, true K) %.3f  K_sel %d ARI(K_sel) %.3f  nearest-acc %.3f | per-class |F−f| %s | class gaps %s | AUROC %.3f val %.1e" % (
        run.split("runs/")[1], a.get("coupling"), lo, hi, float(vf.Z.std()), metrics_of(m)["cluster_ari"], adjusted_rand_score(c, km), K_sel, adjusted_rand_score(c, lab_sel),
        float((nearest == c).mean()), np.round(err, 3), np.round(gap, 2), metrics_of(m)["auroc"], metrics_of(m)["val_mse"]))
