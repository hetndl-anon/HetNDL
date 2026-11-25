"""Allier et al. 2026 baseline (arXiv:2602.13325, "Graph neural networks uncover structure and functions underlying the
activity of simulated neural assemblies"; official code github.com/saalfeldlab/neural-gnn, commit 4880f20b,
Janelia Open-Source License; obtained as described in README.md and run from the patched copy
experiments/external/neural_gnn/patched, see patches.diff) on one HetNDL dataset. Protocol, fixed before any run:
  * code and settings: the authors' trainer (neural_gnn.models.graph_trainer.data_train) with their configuration for
    sparse connectivity, config/signal/signal_fig_supp_8.yaml: 10 epochs, the first 2 without the L1 penalty and then
    coeff_W_L1 1e-5; monotone transfer function (coeff_edge_diff 100, first phase); phi(a, 0) = 0 (coeff_lin_phi_zero 1);
    learning rates 1e-4; data augmentation 50; clustering of the latents every 4 epochs (their App. A.1). Batch 8, the
    batch of their main experiment (config/signal/signal_fig_2.yaml), instead of that file's batch 1: with batch 1 their
    trainer runs one frame per iteration and the 105 runs would need about 33 h on our two GPUs (measured before
    any result was looked at).
    Message function of their Eq. 3, psi*(a_i, a_j, x_j) (model PDE_N5, input size 1 + 2 + 2); --model PDE_N8 adds the
    receiver's state (a variant in their code that the paper does not use).
  * graph not given: with_connectivity_mask False (their default), so every ordered pair i != j carries a learned
    weight. The data folder holds no ground truth: connectivity.pt is zeros (read only by their progress plots) and the
    neuron-type column is zeros (one type).
  * patches (patches.diff): P1 the clustering step groups nodes by its own clusters, as in their App. A.1 (Step 4); the
    upstream code maps clusters onto the true neuron types. P2 every sample draws one training trajectory (upstream:
    always run 0) and the iterations per epoch count the frames of all training trajectories (their formula, 0.2 x
    frames x augmentation / batch). P3 phase states use wrapped differences.
  * data: same trajectory split as every method (hetndl.data.trajectory_split, seed = data seed; NetSim: the training
    seed), training trajectories only; NetSim trajectory caps as for NRI / AIDD / COSINE / MTGNN (--n_traj). Scalar
    states only (their trainer reads one signal column): Kuramoto (phases wrapped to [-pi, pi), wrapped differences),
    MixRelax, GLV, NetSim. Target = their finite difference (x[t+1] - x[t]) / dt. The function profiles of the
    clustering step are sampled on the observed range of the training states (their default [-5, 5] is the range of
    their neural activity).
  * snapshot: the last epoch (their practice; their trainer has no validation step). The initial W is saved too
    (W_init.npy, recorded by wrapping their model constructor without drawing random numbers) with the relative change
    and the correlation of the final W with it, as evidence that training moved the weights.
  * disk: their trainer writes about 2.8 GB of progress images per run (20 plots per epoch); a thread deletes the images
    in the sub-folders of its tmp_training that are older than 3 minutes, and the rest after training (the per-epoch
    summaries stay). The images feed no computation; their end-of-epoch montage reads the newest ones and skips
    missing files.
  * read-outs saved, no scores: W in our orientation (theirs W[i, j] weighs j -> i, so our W[j, i] = theirs[i, j],
    W = theirs^T); latents a; node types by K-means on a with K in 2..10 chosen by the silhouette score (their Sect. 3).
    Scoring (truth only there): experiments/analysis/allier_table.py.
Output runs/allier/<model>/<dataset>/seed<k>/{metrics.json, W.npy, a.npy, labels.npy, work/ (their data and log folders)}.
Run from the repository root with the environment of experiments/external/neural_gnn/requirements.txt (Python 3.10 or
later; ALLIER_PYTHON for experiments/reproduce/run_jobs.py):
python -I experiments/launch/allier_run.py --data <npz> --seed k --device cuda:0 [--model PDE_N5] [--n_traj n]
       [--epochs E --aug A (smoke only)]"""
import argparse, copy, glob, json, os, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCHED = os.path.join(ROOT, "experiments", "external", "neural_gnn", "patched")
CONFIG = os.path.join(ROOT, "experiments", "external", "neural_gnn", "src", "config", "signal", "signal_fig_supp_8.yaml")
FIG2 = os.path.join(ROOT, "experiments", "external", "neural_gnn", "src", "config", "signal", "signal_fig_2.yaml")
sys.path.insert(0, ROOT); sys.path.insert(1, PATCHED)
import numpy as np
import yaml
UPSTREAM = "saalfeldlab/neural-gnn@4880f20b554cde4aee2c6e2f780858c430b17b0b"
SCALAR = ("kuramoto", "a1", "glv", "netsim", "neural_assembly")   # neural_assembly: the positive control (allier_control.py)


def prune_images(root, stop, age=180.0):
    """Delete progress images (*.tif) in the sub-folders of root older than `age` seconds, every 60 s until stop is set;
    after stop, delete the remaining ones."""
    while True:
        done = stop.wait(60); now = time.time()
        for f in glob.glob(os.path.join(root, "*", "**", "*.tif"), recursive=True):
            try:
                if done or now - os.path.getmtime(f) > age:
                    os.remove(f)
            except OSError:
                pass
        if done:
            return


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--model", default="PDE_N5", choices=["PDE_N5", "PDE_N8"])
    ap.add_argument("--n_traj", type=int, default=None); ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--aug", type=int, default=None)                # smoke only: data augmentation loop
    ap.add_argument("--train", default="supp8", choices=["supp8", "supp8_fig2lr", "fig2"])   # training settings (calibration)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from hetndl.data import select_trajectories, trajectory_split
    from hetndl.gen.kuramoto import load_dataset
    ds = os.path.splitext(os.path.basename(a.data))[0]
    out = os.path.abspath(a.out or os.path.join(ROOT, "experiments", "runs", "allier", a.model, ds, "seed%d" % a.seed))
    work = os.path.join(out, "work"); os.makedirs(os.path.join(work, "graphs_data", "data"), exist_ok=True)
    d = load_dataset(os.path.join(ROOT, a.data) if not os.path.isabs(a.data) else a.data)
    system = d["meta"].get("system"); assert system in SCALAR, "scalar states only: %s" % system
    X = np.asarray(d["data"], np.float64)
    assert X.shape[3] == 1, X.shape
    keep = select_trajectories(X.shape[0], a.n_traj, a.seed); X = X[keep]
    tr, va, te = trajectory_split(len(keep), a.seed)
    phase = system == "kuramoto"
    if phase:                                                       # periodic state: wrap to [-pi, pi)
        X = (X + np.pi) % (2 * np.pi) - np.pi
    Xtr = X[tr, :, :, 0]                                            # (M_train, T, N)
    M, T, N = Xtr.shape
    dt = float(d["meta"]["dt"])
    for r in range(M):                                              # their x layout, columns 0..7 (generator docstring)
        x = np.zeros((T, N, 8), np.float32)
        x[:, :, 0] = np.arange(N); x[:, :, 3] = Xtr[r]; x[:, :, 5] = 1.0
        np.save(os.path.join(work, "graphs_data", "data", "x_list_%d.npy" % r), x)
    import torch
    torch.save(torch.zeros(N, N), os.path.join(work, "graphs_data", "data", "connectivity.pt"))
    lo, hi = (-np.pi, np.pi) if phase else (float(Xtr.min()), float(Xtr.max()))

    cfg = yaml.safe_load(open(CONFIG))
    cfg["description"] = "HetNDL benchmark %s seed %d (%s)" % (ds, a.seed, a.model)
    cfg["dataset"] = "data"
    sim = cfg["simulation"]
    sim.update(n_neurons=N, n_neuron_types=1, n_frames=T, delta_t=dt)
    gm = cfg["graph_model"]
    gm.update(signal_model_name=a.model, input_size=1 + 2 + 2 if a.model == "PDE_N5" else 1 + 1 + 2 + 2,
              input_size_update=1 + 2, embedding_dim=2)
    cfg["plotting"].update(xlim=[lo, hi], mlp0_xlim=[lo, hi], mlp1_xlim=[lo, hi])
    if a.train == "fig2":                                           # their main experiment's training block
        cfg["training"] = yaml.safe_load(open(FIG2))["training"]
    tc = cfg["training"]
    if a.train == "supp8_fig2lr":                                   # sparse schedule, learning rates of their main experiment
        tc.update(learning_rate_W_start=1e-3, learning_rate_start=5e-4, learning_rate_embedding_start=7.5e-4)
    tc.update(n_runs=M, seed=a.seed, device=a.device, with_connectivity_mask=False, batch_size=8)
    if a.epochs is not None:
        tc["n_epochs"] = a.epochs
    if a.aug is not None:
        tc["data_augmentation_loop"] = a.aug
    os.makedirs(os.path.join(work, "config"), exist_ok=True)
    cfg_path = os.path.join(work, "config", "run.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    if phase:
        os.environ["HETNDL_PHASE_STATE"] = "1"

    import matplotlib
    matplotlib.use("Agg")
    from neural_gnn.config import NeuralGraphConfig
    from neural_gnn.models.graph_trainer import data_train
    from neural_gnn.utils import set_device
    from neural_gnn.models.Signal_Propagation import Signal_Propagation
    W0 = {}
    _init = Signal_Propagation.__init__
    def _record_init(self, *args, **kw):                            # observer only: copies the initial W, draws no random number
        _init(self, *args, **kw)
        w = getattr(self, "W", None)
        if isinstance(w, torch.nn.Parameter) and "W" not in W0:
            W0["W"] = w.detach().cpu().clone()
    Signal_Propagation.__init__ = _record_init
    config = NeuralGraphConfig.from_yaml(cfg_path)
    config.config_file = "run"
    device = set_device(a.device)
    t0 = time.time(); cwd = os.getcwd(); os.chdir(work)
    stop = threading.Event()
    pruner = threading.Thread(target=prune_images, args=(os.path.join(work, "log", "run", "tmp_training"), stop), daemon=True)
    pruner.start()
    try:
        data_train(config=config, erase=True, best_model=None, style="color", device=device)
    finally:
        os.chdir(cwd); stop.set(); pruner.join()
    wall = time.time() - t0
    n_ep = config.training.n_epochs
    ckpt = os.path.join(work, "log", "run", "models", "best_model_with_%d_graphs_%d.pt" % (M - 1, n_ep - 1))
    sd = torch.load(ckpt, map_location="cpu")["model_state_dict"]
    W = sd["W"].numpy().T.astype(np.float64)                        # ours: W[j, i] = theirs[i, j]
    A = sd["a"].numpy().astype(np.float64)[:N]
    np.save(os.path.join(out, "W.npy"), W); np.save(os.path.join(out, "a.npy"), A)
    conv = {}
    if "W" in W0:                                                  # convergence evidence: how far W moved from its initialisation
        Wi = W0["W"].numpy().T.astype(np.float64); np.save(os.path.join(out, "W_init.npy"), Wi)
        off = ~np.eye(N, dtype=bool)
        conv = dict(w_rel_change=float(np.linalg.norm((W - Wi)[off]) / np.linalg.norm(Wi[off])),
                    w_corr_init=float(np.corrcoef(W[off], Wi[off])[0, 1]))

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best = (-np.inf, 1, np.zeros(N, int))
    for k in range(2, min(10, N - 1) + 1):                          # their Sect. 3: K = 2..10, max silhouette
        if np.unique(A.round(12), axis=0).shape[0] < k:
            break
        lab = KMeans(n_clusters=k, n_init=10, random_state=a.seed).fit_predict(A)
        s = silhouette_score(A, lab)
        if s > best[0]:
            best = (s, k, lab)
    np.save(os.path.join(out, "labels.npy"), best[2])
    m = dict(method="allier2026", model=a.model, upstream=UPSTREAM, patches="P1 P2 P3", config="signal_fig_supp_8", train=a.train,
             epochs=n_ep, n_train=M, T=T, N=N, dt=dt, phase=phase, xlim=[lo, hi], n_traj=len(keep),
             K=int(best[1]), silhouette=float(best[0]), wall_sec=wall, checkpoint=os.path.relpath(ckpt, out), **conv, args=vars(a))
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}, default=float))


if __name__ == "__main__":
    main()
