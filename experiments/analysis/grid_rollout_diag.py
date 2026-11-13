"""Why does HetNDL simulate Grid-118 poorly (Fig. 3d, roll-out NRMSE 0.90) although it identifies the graph?
Diagnostic on the paper seeds, test trajectories, same NRMSE as hetndl.metrics.rollout_nrmse:
  * TRUE swing equations integrated with ONE RK4 step per sample (the integrator HetNDL trains and rolls out with) and
    with 20 RK4 sub-steps (a fine reference) -> how much of the error is the coarse integration step itself;
  * the reported HetNDL model (runs/sel/selected.json), with its own integrator;
  * per component (phase theta, frequency omega) and per horizon.
True field: theta' = omega, omega' = (P_i - D omega) / M_i + sum_j W_true[j, i] sin(theta_j - theta_i) with
W_true[j, i] = kappa K_ij / M_i (hetndl/gen/powergrid.py). Ground truth is used for this diagnosis only.
Run from the project root with PYTHONPATH=. (hetndl env)."""
import json, os
import numpy as np, torch
from experiments.analysis.paper_seeds import PAPER_SEEDS
from hetndl.data import trajectory_split
from hetndl.gen.kuramoto import load_dataset
from hetndl.model import build
from hetndl.snapshot import state_path

H = (1, 5, 10, 20, 50, 100)
sel = json.load(open("experiments/runs/sel/selected.json"))


def nrmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.std(true) + 1e-12))


def true_field(d, Wt=None, P=None):
    meta = d["meta"]; M = np.where(d["classes"] == 1, meta["m_gen"], meta["m_load"]); D = meta["damping"]
    P = d["omega"] if P is None else P; Wt = d["W_true"] if Wt is None else Wt

    def f(x):                                                 # x: (n, N, 2)
        th, om = x[..., 0], x[..., 1]
        coup = np.einsum("ji,nij->ni", Wt, np.sin(th[:, None, :] - th[:, :, None]))   # sum_j W[j,i] sin(th_j - th_i)
        return np.stack([om, (P - D * om) / M + coup], -1)
    return f


def rk4(f, x, h, sub):
    hh = h / sub
    for _ in range(sub):
        k1 = f(x); k2 = f(x + 0.5 * hh * k1); k3 = f(x + 0.5 * hh * k2); k4 = f(x + hh * k3)
        x = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def roll_true(d, te, sub, steps=100, Wt=None, P=None):
    f = true_field(d, Wt, P); x = d["data"][te, 0].astype(np.float64); dt = float(d["meta"]["dt"]); out = []
    for _ in range(steps):
        x = rk4(f, x, dt, sub); out.append(x)
    return np.stack(out, 1)


def roll_model(run, steps=100):
    m = json.load(open(run + "/metrics.json")); ar = m["args"]; d = load_dataset(ar["data"]); data = d["data"]
    N, dim = data.shape[2], data.shape[3]; dt = float(d["meta"]["dt"])
    gp, ode = build(ar["model"], N, dim, ar["hidden"], ar["d_z"], dt, ar.get("rtol", 1e-3), ar.get("atol", 1e-3),
                    ar.get("method", "dopri5"), bool(ar["mask_diag"]), feat=ar.get("feat", "raw"),
                    coupling=ar.get("coupling", "free"), z_norm=ar.get("z_norm", "none"), edge_z=bool(ar.get("edge_z", 0)),
                    step_size=ar.get("step_size"))
    st = torch.load(state_path(run, m), map_location="cpu"); gp.load_state_dict(st["gp"]); ode.load_state_dict(st["vf"])
    gp.eval(); ode.eval(); _, _, te = trajectory_split(data.shape[0], ar["seed"])
    x = torch.from_numpy(data[te, 0]); out = []
    with torch.no_grad():
        W = gp.W()
        for _ in range(steps):
            x = ode(x, W); out.append(x.numpy())
    return np.stack(out, 1), d, te


def report(lab, pred, true):
    hs = " ".join("h%d %.3f" % (h, nrmse(pred[:, :h], true[:, :h])) for h in H)
    print("  %-26s %s | theta %.3f omega %.3f" % (lab, hs, nrmse(pred[..., 0], true[..., 0]), nrmse(pred[..., 1], true[..., 1])))


for s in PAPER_SEEDS:
    key = [k for k in sel if k.startswith("grid118_seed%d/" % s)][0]
    pred, d, te = roll_model(sel[key]["run"])
    true = d["data"][te, 1:101]
    print("seed %d  (max in-strength kappa K / M = %.0f, dt = %.2f)" % (s, d["W_true"].sum(0).max(), d["meta"]["dt"]))
    report("HetNDL (reported model)", pred, true)
    report("true eqs, 1 RK4 step", roll_true(d, te, 1), true)
    report("true eqs, 20 RK4 sub-steps", roll_true(d, te, 20), true)
    rng = np.random.default_rng(s)
    for eps in (0.03, 0.10):                                    # sensitivity: true model with slightly wrong weights
        Wp = d["W_true"] * (1 + eps * rng.standard_normal(d["W_true"].shape)) * (d["W_true"] != 0)
        report("true eqs, W x (1+%.0f%% noise)" % (100 * eps), roll_true(d, te, 20, Wt=Wp), true)
    P = d["omega"]; Pp = P + P.std() * rng.standard_normal(P.shape) * 0.5   # node constants with a moderate error
    report("true eqs, P_i + 0.5 sd noise", roll_true(d, te, 20, P=Pp), true)
