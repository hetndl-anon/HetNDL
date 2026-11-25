"""Positive control for the Allier et al. 2026 baseline: the training budget is calibrated on data of their own model
class before the benchmark runs. Data from THEIR simulator (neural_gnn.generators.
graph_data_generator.data_generate in the patched copy; signal model PDE_N2 = their Eq. 1,
dx_i/dt = -x_i/tau_i + s_i tanh(x_i) + g sum_j W_ij tanh(x_j)) with their sparse configuration
config/signal/signal_fig_supp_8.yaml (Cauchy weights, four neuron types with their parameters, no noise), scaled to the
size of our benchmarks: N = 100 neurons, 10 % of the ordered pairs connected, 20 runs of 500 frames (dt 0.01).
Converted to a HetNDL dataset experiments/data/allier_control_N100_seed<k>.npz: data (20, 500, 100, 1) = their signal
column 3; A, S from their connectivity W[i, j] (weight of j -> i), i.e. our A[j, i] = (W[i, j] != 0); classes = their
neuron types; system name 'neural_assembly'. experiments/launch/allier_run.py then runs their trainer on it exactly as on
our benchmarks; the smallest training budget that recovers this control is used for the benchmark reruns (truth enters
only the scores, and only on this control). --frames gives longer runs (data-size control, same seeds, hence the
same graph and neuron types as the 500-frame control of that seed).
Calibration rule (fixed before any control run): training settings in order of budget, (C0) supp8 = their
sparse configuration, 10 epochs; (C1) supp8_fig2lr = the same with the learning rates of their main
experiment (signal_fig_2.yaml: W 1e-3, networks 5e-4, latents 7.5e-4), 10 epochs; (C2) C1 with 20 epochs; (C3) fig2 = their
main experiment's training block, 10 epochs (budget of C1). The benchmark reruns use the first of C0, C1, C2 whose graph
AUROC is >= 0.95 on both control seeds; C3 replaces C1 only if C1 fails and C3 passes.
usage (project root): <neural-gnn python> -I experiments/launch/allier_control.py --seed k [--device cuda:0]"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCHED = os.path.join(ROOT, "experiments", "external", "neural_gnn", "patched")
CONFIG = os.path.join(ROOT, "experiments", "external", "neural_gnn", "src", "config", "signal", "signal_fig_supp_8.yaml")
sys.path.insert(0, ROOT); sys.path.insert(1, PATCHED)
import numpy as np
import yaml

N, FILL, RUNS, FRAMES = 100, 0.1, 20, 500


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=FRAMES)   # data-size control: longer runs, towards the 10^5 frames of their paper
    a = ap.parse_args()
    import torch, random, matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm
    if not hasattr(matplotlib.cm, "get_cmap"):                    # their plotting code; removed in matplotlib 3.9
        matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap
    from neural_gnn.config import NeuralGraphConfig
    from neural_gnn.generators.graph_data_generator import data_generate
    from neural_gnn.utils import set_device
    name = "allier_control_N%d_seed%d" % (N, a.seed) if a.frames == FRAMES else "allier_control_N%d_T%d_seed%d" % (N, a.frames, a.seed)
    work = os.path.join(ROOT, "experiments", "runs", "allier", "control", "gen", name)
    os.makedirs(os.path.join(work, "config"), exist_ok=True)
    cfg = yaml.safe_load(open(CONFIG))
    cfg["dataset"] = name; cfg["description"] = "positive control for the HetNDL comparison (their model, our data size)"
    cfg["simulation"].update(n_neurons=N, connectivity_filling_factor=FILL, n_frames=a.frames, seed=a.seed)   # their generator seeds from simulation.seed
    cfg["training"].update(n_runs=RUNS, seed=a.seed, device=a.device)
    p = os.path.join(work, "config", "gen.yaml"); yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
    config = NeuralGraphConfig.from_yaml(p); config.config_file = name
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    cwd = os.getcwd(); os.chdir(work)
    try:
        data_generate(config, visualize=False, device=set_device(a.device), bSave=True)
    finally:
        os.chdir(cwd)
    g = os.path.join(work, "graphs_data", name)
    xs = [np.load(os.path.join(g, "x_list_%d.npy" % r)) for r in range(RUNS)]
    X = np.stack([x[:, :, 3] for x in xs])[..., None].astype(np.float64)       # (R, T, N, 1)
    Wt = torch.load(os.path.join(g, "connectivity.pt"), map_location="cpu").numpy().astype(np.float64)
    W = Wt.T                                                                    # ours: W[j, i] = theirs[i, j]
    A = (W != 0).astype(float); np.fill_diagonal(A, 0)
    types = xs[0][0, :, 6].astype(int)
    meta = dict(system="neural_assembly", dt=float(config.simulation.delta_t), T=float(xs[0].shape[0] * config.simulation.delta_t),
                N=N, fill=FILL, runs=RUNS, source="neural-gnn generator, signal_fig_supp_8 with N, fill, frames, runs overridden",
                seed=a.seed)
    out = os.path.join(ROOT, "experiments", "data", name + ".npz")
    np.savez_compressed(out, data=X, A=A, S=np.sign(W) * A, omega=np.zeros(N), classes=types, W_true=W, meta=json.dumps(meta))
    print("saved", os.path.relpath(out, ROOT), X.shape, "edges", int(A.sum()), "negative", int((np.sign(W) * A < 0).sum()),
          "types", np.bincount(types).tolist(), "state range %.2f..%.2f" % (X.min(), X.max()))


if __name__ == "__main__":
    main()
