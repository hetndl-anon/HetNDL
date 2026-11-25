"""Stage 1 of symbolic distillation (hetndl env): sample a trained run's learned self-term F(., z_i) per node and
its coupling c(x_i, x_j) on grids, together with the true functions, and write them to experiments/distill/<tag>/.
Stage 2 (experiments/analysis/distill_pysr.py, env `sr`) fits PySR to these grids.

Systems: a1 (x in R), a2 (u, v), kuramoto (theta; F is the constant omega_i, coupling sin(theta_j - theta_i)),
powergrid (theta, omega; swing model: F = (omega, P_i / M_i - D omega / M_i), coupling (0, sin(theta_j - theta_i));
every node's self field is sampled on its OBSERVED states — theta at the node's mean phase, omega over the 2-98 % range —
because the phases of a synchronised grid stay near node-specific values and a theta x omega box would be mostly
off the data; the PySR input is therefore omega only; classes read from the slope of F_omega in omega exactly as
grid_readout.py does, select_k), glv (x > 0; F_i(x) = r_i x - q_i x^2 with q_i = 1 (free coupling, G = x_i x_j) or q_i = 1 - sum_j W_true[j, i]
(diffusive: G(x_i, x_j) - G(x_i, x_i) = x_i x_j - x_i^2 leaves -x_i^2 sum_j W_ji to the self term)).
Classes are read from the per-node F curves with the truth-free number of groups of hetndl.readout.select_k (the same
read-out as a1_fcurves / a2_fields).
The true labels and the true number of classes are used only to score the result.
Run from the project root with PYTHONPATH=.; args: run dirs [--out experiments/distill]."""
from hetndl.snapshot import metrics_of, state_path, array_path
import argparse, json, os, numpy as np, torch
from hetndl.readout import select_k
from sklearn.metrics import adjusted_rand_score
from hetndl.gen.kuramoto import load_dataset
from hetndl.gen.a1_mixed import self_term
from hetndl.gen.a2_mixed import self_field
from hetndl.model import build, gauge_sign


def swing_class_truth(grid, intercept, slope):
    """True swing-model self field of one class on (theta, omega) points: (omega, intercept + slope * omega)."""
    om = np.asarray(grid, float)[:, 1]
    return np.stack([om, intercept + slope * om], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="experiments/distill")
    ap.add_argument("--n_grid", type=int, default=24)
    a = ap.parse_args()
    for run in a.runs:
        m = json.load(open(run + "/metrics.json")); ar = m["args"]
        d = load_dataset(ar["data"]); meta = d["meta"]; sysname = meta["system"]
        data, A, S, classes, omega = d["data"], d["A"], d["S"], d["classes"], d["omega"]
        N, dim = A.shape[0], data.shape[-1]
        gp, ode = build(ar["model"], N, dim, ar["hidden"], ar["d_z"], float(meta["dt"]), 1e-3, 1e-3, "rk4",
                        bool(ar["mask_diag"]), feat=ar.get("feat", "raw"), coupling=ar.get("coupling", "free"),
                        z_norm=ar.get("z_norm", "none"), edge_z=bool(ar.get("edge_z", 0)))
        st = torch.load(state_path(run, m), map_location="cpu"); gp.load_state_dict(st["gp"]); ode.load_state_dict(st["vf"])
        vf = ode.vf
        assert vf.additive and vf.use_z and not vf.edge_z, "distillation expects an additive z model with a shared G"
        X = data.reshape(-1, dim)
        n = a.n_grid
        # ---- self-term grid
        if sysname == "kuramoto":
            g1 = np.linspace(-np.pi, np.pi, n, endpoint=False); grid = g1[:, None]
            n_cls = 0
        elif sysname == "a1":
            lo, hi = np.percentile(X[:, 0], [5, 95]); grid = np.linspace(lo, hi, 64)[:, None]
            n_cls = True
        elif sysname == "glv":
            lo, hi = np.percentile(X[:, 0], [5, 95]); grid = np.linspace(lo, hi, 64)[:, None]
            n_cls = 0
        elif sysname == "powergrid":                                                           # omega only, see docstring
            grid = np.linspace(*np.percentile(X[:, 1], [2, 98]), 64)[:, None]; n_cls = True
        else:  # a2 / a3
            lo, hi = np.percentile(X, 5, 0), np.percentile(X, 95, 0)
            U, V = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
            grid = np.stack([U.ravel(), V.ravel()], -1); n_cls = True
        P = len(grid)
        Z = vf.z_eff().detach()
        with torch.no_grad():
            h = vf.phi(torch.tensor(grid, dtype=torch.float32))
            if sysname == "powergrid":                                                        # (theta_bar_i, omega) per node
                th_bar = data[..., 0].mean((0, 1))
                F_node = np.stack([vf.F(torch.cat([vf.phi(torch.tensor(np.stack([np.full(P, th_bar[i]), grid[:, 0]], -1), dtype=torch.float32)),
                                                   Z[i].expand(P, -1)], -1)).numpy() for i in range(N)])
            else:
                F_node = np.stack([vf.F(torch.cat([h, Z[i].expand(P, -1)], -1)).numpy() for i in range(N)])   # (N, P, dim)
            xs = torch.tensor(X[np.random.default_rng(0).choice(len(X), min(4096, len(X)), replace=False)], dtype=torch.float32)
            sgn = gauge_sign(vf, xs.view(-1, 1, dim))
        out = dict(system=sysname, run=run, grid=grid, F_node=F_node, Z=Z.numpy(), gauge_sign=sgn,
                   W=np.load(array_path(run, "W", m)), A=A, S=S, classes_true=classes, omega=omega)
        if n_cls:
            # classes from EXACTLY the features of the mechanism read-out (a1_fcurves: 64 points on the 5-95 % range;
            # a2_fields: 16 x 16 grid), so that the distilled classes are the reported classes
            if sysname == "a1":
                feat = F_node
            elif sysname == "powergrid":                                                       # grid_readout.py feature
                th_mean = data[..., 0].mean((0, 1)); om = np.linspace(np.percentile(data[..., 1], 2), np.percentile(data[..., 1], 98), 33)
                feat = np.zeros((N, 1))
                with torch.no_grad():
                    for i in range(N):
                        g = torch.tensor(np.stack([np.full_like(om, th_mean[i]), om], -1), dtype=torch.float32)
                        feat[i, 0] = np.polyfit(om, vf.F(torch.cat([vf.phi(g), Z[i].expand(len(g), -1)], -1)).numpy()[:, 1], 1)[0]
            else:
                U16, V16 = np.meshgrid(np.linspace(lo[0], hi[0], 16), np.linspace(lo[1], hi[1], 16))
                g16 = torch.tensor(np.stack([U16.ravel(), V16.ravel()], -1), dtype=torch.float32)
                with torch.no_grad():
                    h16 = vf.phi(g16)
                    feat = np.stack([vf.F(torch.cat([h16, Z[i].expand(len(g16), -1)], -1)).numpy() for i in range(N)])
            K_sel, lab, _ = select_k(feat.reshape(N, -1), k_max=5, curves=sysname != "powergrid")    # truth-free number of groups
            lab = np.unique(lab, return_inverse=True)[1]                                            # drop empty components
            F_cls = np.stack([F_node[lab == k].mean(0) for k in range(lab.max() + 1)])               # (K_sel, P, dim)
            K_true = int(classes.max()) + 1                                                          # scoring only
            if sysname == "a1":
                T = np.stack([self_term(grid[:, 0], np.full(P, k))[:, None] for k in range(K_true)])
            elif sysname == "powergrid":
                Mi = np.where(classes == 1, meta["m_gen"], meta["m_load"]); Dm = float(meta["damping"])
                g2 = np.stack([np.zeros(P), grid[:, 0]], -1)
                T = np.stack([swing_class_truth(g2, float((omega / Mi)[classes == k].mean()), -Dm / float(Mi[classes == k][0])) for k in range(K_true)])
            else:
                T = np.stack([self_field(grid[:, 0], grid[:, 1], np.full(P, k)) for k in range(K_true)])
            # map learned clusters to true classes by nearest mean curve (for scoring only)
            cost = ((F_cls[:, None] - T[None]) ** 2).mean((-1, -2))                                # (K_sel, K_true)
            match = cost.argmin(1)
            out.update(cluster=lab, F_cls=F_cls, F_true=T, cluster_to_true=match, K_sel=int(lab.max() + 1),
                       ari=adjusted_rand_score(classes, lab))
        elif sysname == "glv":   # continuous node parameter: per-node true curves and a quadratic read-out of r_i
            Wt_ = d["W_true"].astype(np.float64)
            q = 1.0 - Wt_.sum(0) if vf.coupling == "diffusive" else np.ones(N)
            xg = grid[:, 0]
            F_true = np.stack([omega[i] * xg - q[i] * xg ** 2 for i in range(N)])[:, :, None]      # (N, P, 1)
            coef = np.stack([np.polyfit(xg, F_node[i, :, 0], 2) for i in range(N)])               # [c2, c1, c0]
            out.update(F_true_node=F_true, q_true=q, quad_coef=coef)
        else:  # kuramoto: F should be the constant omega_i on the whole circle
            Fm, Fs = F_node[:, :, 0].mean(1), F_node[:, :, 0].std(1)
            out.update(F_mean=Fm, F_std=Fs)
        if sysname in ("kuramoto", "glv"):
            # mechanism classes by the same read-out as a1 (the per-node curves on this self-field grid, select_k); the
            # per-node distillation (pernode_export.py) takes them from here. One true class (scoring only).
            # No F_cls is written, so distill_pysr.py keeps its coupling-only (kuramoto) / pooled (glv) path.
            _, lab_u, _ = select_k(F_node.reshape(N, -1), k_max=5, curves=True)
            lab_u = np.unique(lab_u, return_inverse=True)[1]
            out.update(cluster=lab_u, cluster_to_true=np.zeros(lab_u.max() + 1, dtype=np.int64), K_sel=int(lab_u.max() + 1))
        # ---- coupling grid c(x_i, x_j): the first coordinate varies on both ends, others at the median
        if sysname == "kuramoto":
            r = np.linspace(-np.pi, np.pi, n, endpoint=False)
        else:
            r = np.linspace(np.percentile(X[:, 0], 5), np.percentile(X[:, 0], 95), n)
        XI, XJ = np.meshgrid(r, r, indexing="ij")
        med = np.median(X, 0)
        xi = np.tile(med, (n * n, 1)); xj = xi.copy()
        xi[:, 0], xj[:, 0] = XI.ravel(), XJ.ravel()
        with torch.no_grad():
            hi, hj = vf.phi(torch.tensor(xi, dtype=torch.float32)), vf.phi(torch.tensor(xj, dtype=torch.float32))
            c = vf.G(torch.cat([hi, hj], -1))
            if vf.coupling == "diffusive":
                c = c - vf.G(torch.cat([hi, hi], -1))
            c = (sgn * c).numpy()                                                                  # (n*n, dim)
        if sysname == "kuramoto":
            c_true = np.sin(XJ.ravel() - XI.ravel())[:, None]
            w_true = S * (float(meta["K"]) / np.maximum(A.sum(0), 1))[None, :] if "K" in meta else S
        elif sysname == "a1":
            c_true = (np.tanh(XJ.ravel()) - np.tanh(XI.ravel()))[:, None]; w_true = A * float(meta["kappa"])
        elif sysname == "powergrid":
            c_true = np.stack([np.zeros(n * n), np.sin(XJ.ravel() - XI.ravel())], -1); w_true = d["W_true"].astype(np.float64)
        elif sysname == "glv":
            c_true = (XI.ravel() * XJ.ravel() - (XI.ravel() ** 2 if vf.coupling == "diffusive" else 0.0))[:, None]
            w_true = d["W_true"].astype(np.float64)
        else:
            c_true = np.stack([XJ.ravel() - XI.ravel(), np.zeros(n * n)], -1); w_true = A * float(meta["kappa"])
        out.update(c_grid_xi=xi, c_grid_xj=xj, c_learned=c, c_true=c_true, W_true=w_true, coupling=vf.coupling)
        tag = run.rstrip("/").split("runs/")[-1].replace("/", "__")
        od = os.path.join(a.out, tag); os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, "grids.npz"), **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
        json.dump({k: (v if not isinstance(v, np.ndarray) else None) for k, v in out.items() if not isinstance(v, np.ndarray)},
                  open(os.path.join(od, "info.json"), "w"), indent=1)
        if n_cls:
            extra = "ARI(F clusters) %.3f" % out["ari"]
        elif sysname == "glv":
            from scipy.stats import spearmanr
            c1 = out["quad_coef"][:, 1]
            extra = "linear coef of F_i vs r_i: Pearson %.3f Spearman %.3f | quad coef vs -q_i: Pearson %.3f | R2(F learned, F true) %.3f" % (
                float(np.corrcoef(c1, omega)[0, 1]), float(spearmanr(c1, omega).correlation),
                float(np.corrcoef(out["quad_coef"][:, 0], -out["q_true"])[0, 1]) if np.std(out["q_true"]) > 0 else float("nan"),
                1 - float(((F_node - F_true) ** 2).sum() / ((F_true - F_true.mean()) ** 2).sum()))
        else:
            extra = "F const: std/|mean| %.3f, rho(F_mean, omega) %.3f" % (
                float(np.mean(out["F_std"] / (np.abs(out["F_mean"]) + 1e-9))), float(np.corrcoef(out["F_mean"], omega)[0, 1]))
        print("%s -> %s | %s | gauge %+d" % (tag, od, extra, sgn))


if __name__ == "__main__":
    main()
