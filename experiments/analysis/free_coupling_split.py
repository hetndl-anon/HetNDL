"""Free-coupling models with the self-field / coupling split fixed after training (supplement, Node-Level Read-Outs).
Runs: the "free coupling" row of Table 5 (ablation_table.py): per data set the lower-validation-MSE member of
runs/ablation/core/free/{nonneg,nowu}/<ds>/*, on MixRelax (a1v2_N30_seed1-3) and MixOsc N = 30 (a2_N30_seed1-3).
A free model is dx_i/dt = F(x_i, z_i) + sum_j W_ji G(x_i, x_j). With s_i = sum_j W_ji,
    F_can_i(x) = F(x, z_i) + s_i G(x, x)
gives the same vector field in diffusive form, dx_i/dt = F_can_i(x_i) + sum_j W_ji [G(x_i, x_j) - G(x_i, x_i)]
(Lemma 1 with h = -G(x, x)); the script checks this identity numerically on observed states.
Read-out: the class read-out of the paper (grids of a1_fcurves.py and a2_fields.py, hetndl.readout.select_k with
curves=True), applied to the raw F (which reproduces the free-coupling column of Table 5) and to F_can. Truth (classes)
enters only the ARI and the per-class errors.
Run from the repository root with PYTHONPATH=."""
import glob, json, os, sys
import numpy as np, torch
from sklearn.metrics import adjusted_rand_score
from hetndl.snapshot import state_path
from hetndl.readout import select_k
from hetndl.gen.kuramoto import load_dataset
from hetndl.gen.a1_mixed import self_term
from hetndl.gen.a2_mixed import self_field
from hetndl.model import build

A = "experiments/runs/ablation/core/free"
SYSTEMS = {"MixRelax": ["a1v2_N30_seed%d" % s for s in (1, 2, 3)], "MixOsc N=30": ["a2_N30_seed%d" % s for s in (1, 2, 3)]}
TABLE5_FREE = {"MixRelax": 0.167, "MixOsc N=30": 0.637}


def pick(ds):
    """Lower validation MSE of the two sign members (ablation_table.pick)."""
    c = []
    for mem in ("nonneg", "nowu"):
        for p in glob.glob(os.path.join(A, mem, ds, "*", "metrics.json")):
            m = json.load(open(p))
            if not m.get("failed"):
                c.append((m["val_mse_best"], os.path.dirname(p), m))
    assert len(c) == 2, (ds, len(c))
    return min(c, key=lambda x: x[0])[1:]


def load(run, m, node_dim, t_end):
    a = m["args"]; d = load_dataset(a["data"]); N = len(d["classes"])
    gp, ode = build(a["model"], N, node_dim, a["hidden"], a["d_z"], t_end, 1e-3, 1e-3, "rk4", bool(a["mask_diag"]),
                    feat=a.get("feat", "raw"), coupling=a.get("coupling", "free"), z_norm=a.get("z_norm", "none"),
                    edge_z=bool(a.get("edge_z", 0)), edge_emb=int(a.get("edge_emb", 0) or 0))
    assert a.get("coupling") == "free" and not a.get("edge_z") and not a.get("edge_emb"), run
    st = torch.load(state_path(run, m), map_location="cpu"); gp.load_state_dict(st["gp"]); ode.load_state_dict(st["vf"])
    vf = ode.vf; vf.eval(); gp.eval()
    return d, vf, gp.W().detach()


@torch.no_grad()
def fields(vf, W, g):
    """Raw F(x, z_i) and canonical F + s_i G(x, x) on the grid g (P, node_dim): two arrays (N, P, node_dim)."""
    h = vf.phi(torch.tensor(g, dtype=torch.float32)); P = len(h); N = W.shape[0]
    s = W.sum(0)                                                                   # s_i = sum_j W[j, i]
    Gd = vf.G(torch.cat([h, h], -1))                                               # G(x, x), (P, node_dim)
    raw = torch.stack([vf.F(torch.cat([h, vf.z_eff()[i].expand(P, -1)], -1)) for i in range(N)])
    return raw.numpy(), (raw + s.view(N, 1, 1) * Gd.unsqueeze(0)).numpy()


@torch.no_grad()
def identity_gap(vf, W, X):
    """max |forward - (F_can(x_i) + sum_j W_ji [G(x_i, x_j) - G(x_i, x_i)])| on observed states X (B, N, d), and the
    field scale (max |forward|)."""
    x = torch.tensor(X, dtype=torch.float32); B, N, _ = x.shape
    f = vf(x, W)
    h = vf.phi(x)
    hi = h.unsqueeze(2).expand(B, N, N, vf.fdim); hj = h.unsqueeze(1).expand(B, N, N, vf.fdim)
    Gp = vf.G(torch.cat([hi, hj], -1)); Gd = vf.G(torch.cat([h, h], -1))
    Fz = vf.F(torch.cat([h, vf._z(B)], -1))
    s = W.sum(0).view(1, N, 1)
    Wt = W.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
    alt = (Fz + s * Gd) + ((Gp - Gd.unsqueeze(2)) * Wt).sum(2)
    return float((f - alt).abs().max()), float(f.abs().max())


def run_system(name, out):
    res = []
    for ds in SYSTEMS[name]:
        run, m = pick(ds)
        if name == "MixRelax":
            d, vf, W = load(run, m, 1, 0.02)
            lo, hi = np.percentile(d["data"], [5, 95]); g = np.linspace(lo, hi, 64)[:, None]
            T = np.stack([self_term(g[:, 0], np.full(len(g), k)) for k in range(3)])[..., None]   # (3, P, 1)
        else:
            d, vf, W = load(run, m, 2, 0.05)
            X = d["data"].reshape(-1, 2); lo, hi = np.percentile(X, 5, 0), np.percentile(X, 95, 0)
            U, V = np.meshgrid(np.linspace(lo[0], hi[0], 16), np.linspace(lo[1], hi[1], 16))
            g = np.stack([U.ravel(), V.ravel()], -1)
            T = np.stack([self_field(g[:, 0], g[:, 1], np.full(len(g), k)) for k in range(2)])     # (2, P, 2)
        c = d["classes"]; N = len(c)
        rng = np.random.default_rng(0); D = d["data"]
        Xs = D[rng.integers(0, D.shape[0], 16), rng.integers(0, D.shape[1], 16)]                    # (16, N, d)
        gap, scale = identity_gap(vf, W, Xs)
        assert gap <= 1e-4 * max(scale, 1.0), ("identity check failed", run, gap, scale)
        raw, can = fields(vf, W, g)
        row = dict(ds=ds, run=run, member=m["args"]["model"], gap=gap, scale=scale)
        for lab, Fc in (("raw", raw), ("can", can)):
            K, labels, _ = select_k(Fc.reshape(N, -1), k_max=5, curves=True)
            err = [float(np.abs(Fc[c == k] - T[k]).mean()) for k in range(len(T))]
            row[lab] = dict(K=int(K), ari=float(adjusted_rand_score(c, labels)), err=err)
        res.append(row)
        print("%-12s %-16s %-13s | raw K %d ARI %.3f |F-f| %s | canonical K %d ARI %.3f |F-f| %s | identity gap %.1e (field %.2f)" % (
            name, ds, row["member"], row["raw"]["K"], row["raw"]["ari"], np.round(row["raw"]["err"], 3),
            row["can"]["K"], row["can"]["ari"], np.round(row["can"]["err"], 3), gap, scale), file=out, flush=True)
    raw_m = float(np.mean([r["raw"]["ari"] for r in res])); can_m = float(np.mean([r["can"]["ari"] for r in res]))
    control = abs(raw_m - TABLE5_FREE[name]) <= 0.0015
    print("%-12s mean ARI raw %.3f (Table 5: %.3f, %s) | canonical %.3f" % (
        name, raw_m, TABLE5_FREE[name], "reproduced" if control else "NOT reproduced", can_m), file=out, flush=True)
    return control


if __name__ == "__main__":
    class Tee:
        def __init__(self, *f): self.f = f
        def write(self, s): [x.write(s) for x in self.f]
        def flush(self): [x.flush() for x in self.f]
    fh = open(sys.argv[1], "w") if len(sys.argv) > 1 else None
    out = Tee(sys.stdout, fh) if fh else sys.stdout
    ok = True
    for name in SYSTEMS:
        ok &= run_system(name, out)
    if not ok:
        print("the raw read-out does not reproduce Table 5", file=out)
