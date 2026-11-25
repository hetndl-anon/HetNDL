"""Where does the Grid-118 roll-out error of HetNDL sit? (supplement, "Roll-out on Grid-118").
Checks the statement that over 100 steps a few strongly coupled buses, whose edges and weights the model identifies,
carry most of its frequency error. No training; the reported model of each paper seed
(runs/sel/selected.json) is rolled out for 100 steps from the initial states of the test trajectories with its own
integrator (same roll-out as grid_rollout_diag.py / hetndl.metrics.rollout_nrmse). Per bus: share of the summed squared
frequency error (over test trajectories and the 100 steps); the five buses with the largest share; their rank in true
in-strength sum_j W_true[j, i]; and, for their incoming edges, whether HetNDL ranks them (AUROC of the receiver's
column) and the relative error of the learned weights after the one scalar alpha that removes the scale symmetry
(fitted over all true edges, as in grid_readout.py). Ground truth is used for this diagnosis only.
Run from the project root with PYTHONPATH=. (hetndl env)."""
import json
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from experiments.analysis.paper_seeds import PAPER_SEEDS
from hetndl.data import trajectory_split
from hetndl.gen.kuramoto import load_dataset
from hetndl.model import build
from hetndl.snapshot import state_path

sel = json.load(open("experiments/runs/sel/selected.json"))


def roll_model(run, steps=100):                    # identical to grid_rollout_diag.roll_model (that module runs on import)
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
        Wl = W.numpy()
    return np.stack(out, 1), d, te, Wl


shares = []
for s in PAPER_SEEDS:
    key = [k for k in sel if k.startswith("grid118_seed%d/" % s)][0]
    pred, d, te, Wl = roll_model(sel[key]["run"])
    true = d["data"][te, 1:101]
    e2 = ((pred[..., 1] - true[..., 1]) ** 2).sum((0, 1))               # per bus, summed over trajectories and steps
    share = e2 / e2.sum(); top = np.argsort(-share)[:5]
    Wt = d["W_true"]; A = Wt != 0; instr = Wt.sum(0); rank = (-instr).argsort().argsort() + 1   # 1 = strongest
    alpha = float((np.abs(Wl[A]) @ np.abs(Wt[A])) / (np.abs(Wl[A]) @ np.abs(Wl[A])))
    print("seed %d (reported run %s)" % (s, sel[key]["run"]))
    print("  top-5 share of the summed squared frequency error: %.3f; top-1 %.3f; top-10 %.3f"
          % (share[top].sum(), share[top[0]], np.sort(share)[::-1][:10].sum()))
    for i in top:
        col = np.delete(np.arange(len(instr)), i)
        y = A[col, i].astype(int); sc = np.abs(Wl[col, i])
        au = roc_auc_score(y, sc) if 0 < y.sum() < len(y) else float("nan")
        rel = np.abs(alpha * np.abs(Wl[A[:, i], i]) - np.abs(Wt[A[:, i], i])) / np.abs(Wt[A[:, i], i])
        print("   bus %3d  share %.3f  inertia %.1f  in-degree %d  in-strength %.1f (rank %d of %d)  incoming-edge AUROC %.3f"
              "  rel. weight error mean %.3f" % (i, share[i], 1.0 if d["classes"][i] == 1 else 0.2, A[:, i].sum(), instr[i],
                                                 rank[i], len(instr), au, rel.mean()))
    shares.append(share[top].sum())
print("top-5 share over the paper seeds: %s; pooled range %.3f-%.3f" % (np.round(shares, 3), min(shares), max(shares)))
