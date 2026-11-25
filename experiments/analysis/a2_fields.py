"""A2 read-out: per-node learned self-field F((u, v), z_i) vs the true FHN / Van der Pol fields on a 16 x 16 grid
over the observed (u, v) box (5-95 % per coordinate). Reports z std, KMeans-on-fields ARI, nearest-true-field
accuracy, per-class mean |F - f| (both coordinates) and the mean gap between the two true fields."""
from hetndl.snapshot import metrics_of, state_path, array_path
from hetndl.readout import select_k
import json, sys, numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from hetndl.gen.kuramoto import load_dataset
from hetndl.gen.a2_mixed import self_field
from hetndl.model import build
for run in sys.argv[1:]:
    m = json.load(open(run + "/metrics.json")); a = m["args"]
    d = load_dataset(a["data"]); c = d["classes"]; N = len(c)
    gp, ode = build(a["model"], N, 2, a["hidden"], a["d_z"], 0.05, 1e-3, 1e-3, "rk4", bool(a["mask_diag"]),
                    feat=a.get("feat", "raw"), coupling=a.get("coupling", "free"), z_norm=a.get("z_norm", "none"),
                    edge_z=bool(a.get("edge_z", 0)), edge_emb=int(a.get("edge_emb", 0) or 0))
    st = torch.load(state_path(run, m), map_location="cpu"); gp.load_state_dict(st["gp"]); ode.load_state_dict(st["vf"]); vf = ode.vf; vf.eval()
    X = d["data"].reshape(-1, 2)
    lo, hi = np.percentile(X, 5, 0), np.percentile(X, 95, 0)
    U, V = np.meshgrid(np.linspace(lo[0], hi[0], 16), np.linspace(lo[1], hi[1], 16))
    g = np.stack([U.ravel(), V.ravel()], -1)                                   # (256, 2)
    T = np.stack([self_field(g[:, 0], g[:, 1], np.full(len(g), k)) for k in range(2)])   # (2, 256, 2)
    with torch.no_grad():
        h = vf.phi(torch.tensor(g, dtype=torch.float32))
        if vf.use_z:
            Fc = np.stack([vf.F(torch.cat([h, vf.z_eff()[i].expand(len(g), -1)], -1)).numpy() for i in range(N)])
        else:
            Fc = np.stack([vf.F(h).numpy()] * N) if vf.additive else np.full((N, len(g), 2), np.nan)
    if np.isnan(Fc).any():
        print("%s  non-additive model: no self-field | AUROC %.3f" % (run.split("runs/")[-1], metrics_of(m)["auroc"])); continue
    nearest = np.argmin(((Fc[:, None] - T[None]) ** 2).mean((-1, -2)), 1)
    km = KMeans(2, n_init=20, random_state=0).fit_predict(Fc.reshape(N, -1))      # true K (reference only)
    K_sel, lab_sel, _ = select_k(Fc.reshape(N, -1), k_max=5, curves=True)                        # truth-free K (protocol)
    err = [float(np.abs(Fc[c == k] - T[k]).mean()) for k in range(2)]
    zs = float(vf.Z.std()) if vf.use_z else float("nan")
    print("%s  coupling=%s  z std %.3g | ARI(fields, true K) %.3f  K_sel %d ARI(K_sel) %.3f  nearest-acc %.3f | per-class |F−f| %s | class gap %.2f | AUROC %.3f AUPRC %.3f val %.1e" % (
        run.split("runs/")[-1], a.get("coupling"), zs, adjusted_rand_score(c, km), K_sel, adjusted_rand_score(c, lab_sel), float((nearest == c).mean()),
        np.round(err, 3), float(np.abs(T[0] - T[1]).mean()), metrics_of(m)["auroc"], metrics_of(m)["auprc"], metrics_of(m)["val_mse"]))
