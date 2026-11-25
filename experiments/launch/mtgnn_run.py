"""MTGNN baseline (Wu et al., KDD 2020, "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural
Networks"; official code github.com/nnzhan/MTGNN, master zip sha256 6d2fa2e1...423c, commit f811746, kept unmodified in
experiments/external/mtgnn/src) on one HetNDL dataset. Protocol:
  * same trajectory split as every method (hetndl.data.trajectory_split, seed = data seed); windows of 12 input and 12
    output steps are cut inside each trajectory, never across two; Netsim trajectory caps as for NRI / AIDD / COSINE.
  * inputs: the observed states; phases encoded as (cos, sin), the a-priori rule used for HetNDL and NOGI (raw unwrapped
    phases drift and would handicap a forecaster); every input feature standardised with training statistics. Target =
    the first state feature, loss in original units through the official StandardScaler / masked-MAE Trainer (MTGNN's
    convention: all features in, one target channel out).
  * model and training = the official multi-step defaults (gcn_depth 2, dropout 0.3, subgraph_size k = 20 capped at N,
    node_dim 40, conv / residual channels 32, skip 64, end 128, 3 layers, propalpha 0.05, tanhalpha 3, batch 64, Adam
    1e-3, weight decay 1e-4, clip 5, node permutation every 100 iterations, 100 epochs) with the official Trainer and its
    curriculum learning. The official run (METR-LA: 375 iterations per epoch x 100 epochs = 37,500 iterations, a new
    horizon every 2,500) reaches the full 12-step horizon after 73 % of training; our datasets are smaller, so the step
    is scaled to keep that fraction: step = round(2500 x total iterations / 37,500) (fixed before any run); snapshot = lowest validation MAE (the official rule). No ground truth enters the run.
  * graph: the dense learned adjacency gc.fullA (before the top-k mask, so that every pair has a score). MTGNN
    propagates out[v] = sum_w A[v, w] x[w] (layer.nconv), so A[v, w] weighs w -> v and our W[j, i] (j -> i) = A[i, j],
    i.e. W = A^T. Scored with experiments/analysis/mtgnn_rescore.mtgnn_scores: symmetrised W + W^T as the main score
    (MTGNN's adjacency is non-zero in one direction per pair and is used in both; see that file), per-direction
    scores kept as *_directed. MTGNN learns no signs.
Output runs/mtgnn/<dataset>/seed<k>/{metrics.json, W.npy, train.log}. Run from the project root:
python -I experiments/launch/mtgnn_run.py --data <npz> --seed k --device cuda:0 [--n_traj n] [--epochs E (smoke only)]"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "experiments", "external", "mtgnn", "src")
sys.path.insert(0, ROOT); sys.path.append(SRC)                     # project first; the external code only for its modules
import numpy as np
import torch
from hetndl.data import select_trajectories, trajectory_split
from hetndl.gen.kuramoto import load_dataset
from experiments.analysis.mtgnn_rescore import mtgnn_scores
from net import gtnet                                              # official model (unmodified)
from trainer import Trainer                                       # official trainer (unmodified)
from util import StandardScaler                                   # official scaler (unmodified)
L_IN = L_OUT = 12


def windows(X, idx):
    """X (S, T, N, F) -> inputs (M, L_IN, N, F) and targets (M, L_OUT, N, F), cut inside each trajectory."""
    xs, ys = [], []
    for s in idx:
        T = X.shape[1]
        for t in range(T - L_IN - L_OUT + 1):
            xs.append(X[s, t:t + L_IN]); ys.append(X[s, t + L_IN:t + L_IN + L_OUT])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n_traj", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ds = os.path.splitext(os.path.basename(a.data))[0]
    out = a.out or os.path.join(ROOT, "experiments", "runs", "mtgnn", ds, "seed%d" % a.seed); os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "train.log"), "w")
    d = load_dataset(a.data); X = np.asarray(d["data"], np.float64)
    keep = select_trajectories(X.shape[0], a.n_traj, a.seed); X = X[keep]
    if d["meta"].get("system") == "kuramoto":                      # periodic feature rule for phases
        X = np.concatenate([np.cos(X), np.sin(X)], -1)
    tr, va, te = trajectory_split(len(keep), a.seed)
    N, F = X.shape[2], X.shape[3]
    mu, sd = X[tr].mean((0, 1, 2)), X[tr].std((0, 1, 2)) + 1e-8
    scaler = StandardScaler(mean=float(mu[0]), std=float(sd[0]))
    xtr, ytr = windows((X - mu) / sd, tr); _, ytr_raw = windows(X, tr)
    xva, _ = windows((X - mu) / sd, va); _, yva_raw = windows(X, va)
    dev = torch.device(a.device)
    model = gtnet(True, True, 2, N, dev, predefined_A=None, dropout=0.3, subgraph_size=min(20, N), node_dim=40,
                  dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128,
                  seq_length=L_IN, in_dim=F, out_dim=L_OUT, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
    total = a.epochs * int(np.ceil(len(xtr) / 64)); cl_step = max(1, int(round(2500 * total / 37500)))
    eng = Trainer(model, 0.001, 0.0001, 5, cl_step, L_OUT, scaler, dev, True)
    # official tensors: x (B, F, N, L_IN); y (B, F, N, L_OUT) and the target channel y[:, 0] in original units
    to = lambda arr: torch.tensor(arr).transpose(1, 3).to(dev)
    best, best_ep, best_state, it, t0 = np.inf, -1, None, 0, time.time()
    rng = np.random.default_rng(a.seed)
    for ep in range(1, a.epochs + 1):
        order = rng.permutation(len(xtr)); tl = []
        for b in range(0, len(order), 64):
            sel = order[b:b + 64]
            if it % 100 == 0:
                perm = torch.tensor(np.random.permutation(N)).to(dev)
            tx = to(xtr[sel])[:, :, perm, :]; ty = to(ytr_raw[sel])[:, :, perm, :]
            tl.append(eng.train(tx, ty[:, 0, :, :], perm)[0]); it += 1
        vl = []
        with torch.no_grad():
            for b in range(0, len(xva), 64):
                vl.append(eng.eval(to(xva[b:b + 64]), to(yva_raw[b:b + 64])[:, 0, :, :])[0])
        v = float(np.mean(vl)); log.write("epoch %d train_mae %.5f val_mae %.5f %.0fs\n" % (ep, np.mean(tl), v, time.time() - t0)); log.flush()
        if v < best:
            best, best_ep = v, ep; best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        A = model.gc.fullA(model.idx).cpu().numpy()
    W = A.T                                                         # W[j, i] = A[i, j]: weight of j -> i
    sc = mtgnn_scores(W, d["A"])
    np.save(os.path.join(out, "W.npy"), W)
    m = dict(method="mtgnn", **sc, sign_acc=float("nan"), best_epoch=best_ep, val_mae_best=best,
             epochs=a.epochs, cl_step=cl_step, n_traj=len(keep), wall_sec=time.time() - t0, k=min(20, N), upstream="nnzhan/MTGNN@f811746", args=vars(a))
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}, default=float))


if __name__ == "__main__":
    main()
