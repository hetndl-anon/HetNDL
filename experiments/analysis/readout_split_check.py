"""Do the mechanism read-outs change when their state grids are built from the training trajectories only?
The read-outs place their evaluation grids on quantiles of the observed states of ALL trajectories (training, validation
and test; states only, no labels). This check rebuilds the same features from the TRAINING trajectories alone
(hetndl.data.select_trajectories + trajectory_split with the run's seed, as in train.py) for the reported runs of
MixRelax, MixRelax (BA), MixOsc N=30 / N=60 and Grid-118 (data seeds 1-3, experiments/runs/sel/selected.json) and compares
the number of groups K of hetndl.readout.select_k, the ARI against the true classes and the two partitions.
Features, copied from the read-out scripts (the "all" column must reproduce their K and ARI):
  MixRelax (a1_fcurves.py)  F(x, z_i) on 64 points over the 5-95 % range of the states, select_k(curves=True);
  MixOsc   (a2_fields.py)   F((u, v), z_i) on a 16 x 16 grid over the 5-95 % range of each coordinate, select_k(curves=True);
  Grid-118 (grid_readout.py) slope in omega of the second component of F at the node's mean phase, omega on 33 points over
                            its 2-98 % range, select_k on the slope.
Truth (classes) only in the ARI. Prints one line per run and a final line IDENTICAL or DIFFERENT.
Run from the project root with PYTHONPATH=."""
import numpy as np, torch
from sklearn.metrics import adjusted_rand_score
from hetndl.data import select_trajectories, trajectory_split
from hetndl.edge_readout import load_run
from hetndl.readout import select_k
from experiments.analysis.baseline_table import B, PAPER_NAME, sel, sel_key
SETTINGS = ("MixRelax", "MixRelax (BA)", "MixOsc N=30", "MixOsc N=60", "Grid-118")


def node_fields(vf, g):
    """F(g, z_i) for every node i on the grid points g (P, dim) -> (N, P, dim)."""
    with torch.no_grad():
        h = vf.phi(torch.tensor(g, dtype=torch.float32))
        return np.stack([vf.F(torch.cat([h, vf.z_eff()[i].expand(len(g), -1)], -1)).numpy() for i in range(vf.N)])


def readout(vf, X, system):
    """(K, labels, grid range as text) from the states X (S, T, N, dim) of the chosen trajectories."""
    N = X.shape[2]
    if system == "a1":
        lo, hi = np.percentile(X, [5, 95]); xs = np.linspace(lo, hi, 64)
        feat = node_fields(vf, xs[:, None])[:, :, 0]; rng = "x [%.3f, %.3f]" % (lo, hi); curves = True
    elif system == "a2":
        Xf = X.reshape(-1, 2); lo, hi = np.percentile(Xf, 5, 0), np.percentile(Xf, 95, 0)
        U, V = np.meshgrid(np.linspace(lo[0], hi[0], 16), np.linspace(lo[1], hi[1], 16))
        feat = node_fields(vf, np.stack([U.ravel(), V.ravel()], -1)).reshape(N, -1)
        rng = "u [%.3f, %.3f] v [%.3f, %.3f]" % (lo[0], hi[0], lo[1], hi[1]); curves = True
    elif system == "powergrid":
        th_mean = X[..., 0].mean((0, 1)); om = np.linspace(np.percentile(X[..., 1], 2), np.percentile(X[..., 1], 98), 33)
        feat = np.zeros((N, 1))
        with torch.no_grad():
            for i in range(N):
                g = torch.tensor(np.stack([np.full_like(om, th_mean[i]), om], -1), dtype=torch.float32)
                feat[i, 0] = np.polyfit(om, vf.F(torch.cat([vf.phi(g), vf.z_eff()[i].expand(len(g), -1)], -1)).numpy()[:, 1], 1)[0]
        rng = "omega [%.3f, %.3f]" % (om[0], om[-1]); curves = False
    else:
        raise ValueError(system)
    K, lab, _ = select_k(feat, k_max=5, curves=curves)
    return K, lab, rng


def main():
    print("%-14s %4s  %-5s  %-46s %2s %6s | %-46s %2s %6s | %s" % (
        "setting", "seed", "train", "all trajectories: grid range", "K", "ARI", "training trajectories only: grid range", "K", "ARI", "partitions"))
    same = True; n = 0
    for lab, cells in B:
        name = PAPER_NAME[lab]
        if name not in SETTINGS:
            continue
        for ds, s in cells:
            m, d, vf, W = load_run(sel[sel_key(lab, ds, s)]["run"]); a = m["args"]; X = np.asarray(d["data"], np.float32)
            keep = select_trajectories(X.shape[0], a.get("n_traj"), a["seed"]); tr, _, _ = trajectory_split(len(keep), a["seed"])
            system = d["meta"]["system"]
            Ka, la, ra = readout(vf, X, system); Kt, lt, rt = readout(vf, X[keep][tr], system)
            cls = d["classes"]; aa, at = adjusted_rand_score(cls, la), adjusted_rand_score(cls, lt)          # scoring only
            eq = adjusted_rand_score(la, lt) == 1.0 or (Ka == 1 and Kt == 1)
            ok = Ka == Kt and eq and abs(aa - at) < 1e-12; same &= ok; n += 1
            print("%-14s %4d  %2d/%-2d  %-46s %2d %6.3f | %-46s %2d %6.3f | %s" % (
                name, s, len(tr), X.shape[0], ra, Ka, aa, rt, Kt, at, "same" if eq else "differ"))
    print("\n%d runs: K, ARI and the node partition with training-trajectory grids vs all-trajectory grids: %s" % (n, "IDENTICAL" if same else "DIFFERENT"))


if __name__ == "__main__":
    main()
