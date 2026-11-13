"""MixRelax with node coefficients that vary within each law (supplement, Node-Level Read-Outs). Data
a1v2_N30_cs10_seed{1,2,3}: hetndl.gen.a1_mixed with the arguments of a1v2_N30_seed{k} plus --coef_spread 0.1, i.e. every
node keeps its class law with its own coefficients (a, b) = class values x U[0.9, 1.1] factors; graph, classes and initial
states equal those of a1v2_N30_seed{k}. Runs coef_spread/<dataset>/{nonneg,nowu}/<run>: the two candidates with the
arguments of the main-table members of a1v2_N30_seed{k}, data replaced; the candidate with the lower validation error
(val_mse_best) is read, the selection rule of the paper.
Per data set: graph AUROC, the class read-out of the paper for MixRelax (the computation of a1_fcurves.py: learned curves
F(x, z_i) on 64 points over the 5-95 % range of all observed states, hetndl.readout.select_k(curves=True)) with its K and
its ARI against the true laws, and the mean absolute difference between each node's learned self-field and its true
curve (its law with its own coefficients) on that grid. Truth enters only the scores.
Run from the repository root with PYTHONPATH=."""
import glob, json, os, sys
import numpy as np, torch
from sklearn.metrics import adjusted_rand_score
from hetndl.gen.a1_mixed import self_term
from hetndl.gen.kuramoto import load_dataset
from hetndl.model import build
from hetndl.readout import select_k
from hetndl.snapshot import metrics_of, state_path

R = "experiments/runs/coef_spread"
DS = "a1v2_N30_cs10_seed%d"


def pick(ds):
    c = []
    for mem in ("nonneg", "nowu"):
        for p in glob.glob(os.path.join(R, ds, mem, "*", "metrics.json")):
            m = json.load(open(p))
            if not m.get("failed"):
                c.append((m["val_mse_best"], mem, os.path.dirname(p), m))
    return min(c, key=lambda x: x[0]) if len(c) == 2 else None


def read(run, m):
    a = m["args"]; d = load_dataset(a["data"]); N = len(d["classes"])
    gp, ode = build(a["model"], N, 1, a["hidden"], a["d_z"], 0.02, 1e-3, 1e-3, "rk4", bool(a["mask_diag"]),
                    feat=a.get("feat", "raw"), coupling=a.get("coupling", "free"), z_norm=a.get("z_norm", "none"),
                    edge_z=bool(a.get("edge_z", 0)))
    st = torch.load(state_path(run, m), map_location="cpu"); gp.load_state_dict(st["gp"]); ode.load_state_dict(st["vf"]); vf = ode.vf
    lo, hi = np.percentile(d["data"], [5, 95]); xs = np.linspace(lo, hi, 64)
    with torch.no_grad():
        h = vf.phi(torch.tensor(xs, dtype=torch.float32).view(-1, 1))
        Fc = np.stack([vf.F(torch.cat([h, vf.z_eff()[i].expand(64, -1)], -1))[:, 0].numpy() for i in range(N)])
    K, lab, _ = select_k(Fc, k_max=5, curves=True)
    T = np.stack([self_term(xs, np.full(64, d["classes"][i]), np.broadcast_to(d["coef"][i], (64, 2))) for i in range(N)])
    return K, adjusted_rand_score(d["classes"], lab), float(np.abs(Fc - T).mean())


if __name__ == "__main__":
    out, aris = [], []
    for k in (1, 2, 3):
        p = pick(DS % k)
        if p is None:
            out.append("%-22s missing or failed run" % (DS % k)); continue
        val, mem, run, m = p
        K, ari, dist = read(run, m); aris.append(ari)
        out.append("%-22s pick %-6s val %.2e  AUROC %.3f | K %d  ARI %.3f | mean |F - f_i| %.4f" % (
            DS % k, mem, val, metrics_of(m)["auroc"], K, ari, dist))
    if len(aris) == 3:
        out.append("mean ARI %.3f" % np.mean(aris))
    txt = "\n".join(out); print(txt)
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(txt + "\n")
