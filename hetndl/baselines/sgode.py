"""SGODE baseline (Chen et al., "Signed Graph Neural Ordinary Differential Equation for Modeling Continuous-time
Dynamics", AAAI 2024; arXiv:2312.11198v1). Our re-implementation of the paper's continuous-dynamics model (SGODE
embedded into NDCN, its Eq. 7): the public repository only has the traffic model (SGODE-RNN) and no licence, so nothing
is copied; the ODE function follows the paper and the order of operations of the official ODEFunc_SGODE
(model_pytorch_ode.py of that repository), the skeleton follows NDCN (Zang and Wang, KDD 2020;
github.com/calvin-zcx/ndcn, neural_dynamics.py / heat_dynamics.py and their defaults). It is run on every benchmark of
the main table.

Everything below was fixed BEFORE any run was scored; no ground truth enters a run except the scores
written at the end.
  * model: H(0) = Linear(F, h) - Tanh - Linear(h, h) of the first snapshot of a window (NDCN input layer);
    dH/dt = ReLU(W_h (K H + c * H) + b_h) with K (n x n), c (n,) the per-node self term of the paper (B(t) = b H(t) W_h);
    X(t) = Linear(h, F) of H(t) (NDCN output layer). h = 20 and dropout 0 (NDCN defaults, which the paper keeps for the
    synthetic dynamics), embedding size 10 (paper, appendix).
  * variants: v1 = K is a free n x n matrix; v3 = K = ReLU(E1 E2^T) - ReLU(E3 E4^T) (the sigma of the paper's Eq. 3 / 7 is
    the ReLU of the cited AGCRN / NDCN layers, as in the official code). Embeddings and c start as in the official code
    (softmax over the nodes of U(0, 1) numbers); v1's K starts at the value of v3's K at its initialisation, so that both
    variants start from the same distribution. v2 (K = E1 E2^T) is not run.
  * data: the same trajectory split as every method (hetndl.data.trajectory_split, seed = data seed) and the NetSim
    trajectory caps of NRI / AIDD / COSINE / MTGNN. Inputs = the observed states, phases as (cos, sin) (the a-priori
    rule used for every method), every feature standardised with training statistics. A window = one snapshot and the
    next 12 snapshots (the horizon used for the other forecaster, MTGNN), cut inside one trajectory, stride 1.
  * training: L1 loss over the 12 predicted snapshots (the paper's Eq. 13), Adam, learning rate 0.01 (paper), weight
    decay 1e-4 (the most frequent value of the paper's Table 8 for SGODEv1 and for SGODEv3; the paper tunes it per
    dataset, we cannot), batches of 256 windows, and the per-benchmark rules shared by the neural ODE methods of this
    repository: 1000 epochs (600 on Kuramoto, 200 on NetSim); Dormand-Prince (rtol = atol = 1e-3) on MixRelax / MixOsc /
    MixOsc-E, fixed-step RK4 with one step per sampling interval on Kuramoto, GLV, Grid-118 and with step 0.25 on NetSim
    (protocol()). Checkpoint = lowest validation loss (12-step L1 on the validation trajectories), every epoch.
  * graph: (K H)_i = sum_j K[i, j] H_j, so K[i, j] weighs j -> i and our W[j, i] = K[i, j], W = K^T. Score = |W| per
    ordered pair (hetndl.metrics.auroc_auprc; K is a full directed matrix, so no symmetrisation). Signs: sign(W) against
    the true signs on the true edges; K acts on hidden states, so its overall sign is a gauge (K, c, W_h -> -K, -c, -W_h
    leaves the model unchanged) and sign_acc_flip = max(acc, 1 - acc) is reported next to the raw value, which favours
    SGODE. Reading K as an interaction graph is our addition: the paper reports forecasting error only.
  * reported column: per run the variant with the lower validation loss (experiments/analysis/sgode_table.py); both
    variants are listed in the supplementary material.
  * sanity value (no truth): val_mae_persistence, the same 12-step L1 of repeating the first snapshot.
Output <out>/{metrics.json, W.npy, train.log}; default out = experiments/runs/sgode/<variant>/<dataset>/seed<k>.
Run from the project root: python -m hetndl.baselines.sgode --data <npz> --seed k --variant v1|v3 --device cuda:0
[--n_traj n] [--epochs E (smoke tests only)]"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from ..data import select_trajectories, trajectory_split
from ..gen.kuramoto import load_dataset
from ..metrics import auroc_auprc, sign_accuracy

HORIZON, HIDDEN, EMBED, LR, WEIGHT_DECAY, BATCH = 12, 20, 10, 0.01, 1e-4, 256


def protocol(system: str):
    """(epochs, solver, step size or None, periodic features) of a benchmark family: the rules of the main runs."""
    if system == "kuramoto":
        return 600, "rk4", None, True
    if system in ("a1", "a2", "a3"):
        return 1000, "dopri5", None, False
    if system in ("glv", "powergrid"):
        return 1000, "rk4", None, False
    if system == "netsim":
        return 200, "rk4", 0.25, False
    raise ValueError("no protocol for system %r" % system)


def _embed(n: int, d: int) -> torch.Tensor:
    return torch.softmax(torch.rand(n, d), dim=0)                  # official initialisation


class SGODEFunc(nn.Module):
    def __init__(self, n: int, hidden: int = HIDDEN, embed: int = EMBED, variant: str = "v3"):
        super().__init__()
        assert variant in ("v1", "v3"), variant
        self.variant = variant
        E = [_embed(n, embed) for _ in range(4)]
        if variant == "v3":
            self.E1, self.E2, self.E3, self.E4 = (nn.Parameter(e) for e in E)
        else:
            self.K_free = nn.Parameter(torch.relu(E[0] @ E[1].T) - torch.relu(E[2] @ E[3].T))
        self.c = nn.Parameter(torch.softmax(torch.rand(n), dim=0))
        self.wt = nn.Linear(hidden, hidden)

    def K(self) -> torch.Tensor:
        if self.variant == "v1":
            return self.K_free
        return torch.relu(self.E1 @ self.E2.T) - torch.relu(self.E3 @ self.E4.T)

    def forward(self, t, h):                                       # h (B, n, hidden)
        return torch.relu(self.wt(torch.einsum("nm,bmc->bnc", self.K(), h) + self.c.reshape(1, -1, 1) * h))


class SGODE(nn.Module):
    def __init__(self, n: int, feat: int, variant: str, dt: float, method: str, step_size=None, rtol=1e-3, atol=1e-3,
                 hidden: int = HIDDEN, embed: int = EMBED, horizon: int = HORIZON):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(feat, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.func = SGODEFunc(n, hidden, embed, variant)
        self.dec = nn.Linear(hidden, feat)
        self.register_buffer("t", torch.arange(horizon + 1, dtype=torch.float32) * float(dt))
        self.method, self.rtol, self.atol = method, rtol, atol
        self.options = {"step_size": float(step_size)} if step_size is not None else None

    def forward(self, x0):                                         # x0 (B, n, F) -> (B, horizon, n, F)
        sol = odeint(self.func, self.enc(x0), self.t, method=self.method, rtol=self.rtol, atol=self.atol, options=self.options)
        return self.dec(sol[1:]).transpose(0, 1)

    def graph(self) -> np.ndarray:
        """W[j, i] = weight of j -> i = K[i, j]."""
        return self.func.K().detach().cpu().numpy().T.copy()


def window_index(traj_idx, T: int, horizon: int = HORIZON):
    """(trajectory, start) of every window of horizon + 1 snapshots that lies inside one trajectory."""
    s = np.repeat(np.asarray(traj_idx), T - horizon); t = np.tile(np.arange(T - horizon), len(traj_idx))
    return s, t


def take(X: torch.Tensor, s, t, horizon: int = HORIZON) -> torch.Tensor:
    """X (S, T, n, F) -> windows (B, horizon + 1, n, F)."""
    s = torch.as_tensor(s, device=X.device); t = torch.as_tensor(t, device=X.device)
    return X[s[:, None], t[:, None] + torch.arange(horizon + 1, device=X.device)[None]]


def sgode_scores(W: np.ndarray, d) -> dict:
    au, ap = auroc_auprc(W, d["A"]); sg = sign_accuracy(W, d["S"])
    return dict(auroc=au, auprc=ap, sign_acc=sg, sign_acc_flip=(max(sg, 1.0 - sg) if sg == sg else float("nan")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=["v1", "v3"], required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n_traj", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only; default = protocol()")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ds = os.path.splitext(os.path.basename(a.data))[0]
    out = a.out or os.path.join("experiments", "runs", "sgode", a.variant, ds, "seed%d" % a.seed); os.makedirs(out, exist_ok=True)
    d = load_dataset(a.data); X = np.asarray(d["data"], np.float64)
    epochs, method, step, trig = protocol(d["meta"].get("system")); epochs = a.epochs or epochs
    keep = select_trajectories(X.shape[0], a.n_traj, a.seed); X = X[keep]
    if trig:
        X = np.concatenate([np.cos(X), np.sin(X)], -1)
    tr, va, _ = trajectory_split(len(keep), a.seed)
    mu, sd = X[tr].mean((0, 1, 2)), X[tr].std((0, 1, 2)) + 1e-8
    dev = torch.device(a.device); Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    T, n, F = X.shape[1], X.shape[2], X.shape[3]
    model = SGODE(n, F, a.variant, float(d["meta"]["dt"]), method, step).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    s_tr, t_tr = window_index(tr, T); s_va, t_va = window_index(va, T)

    def val_loss():
        tot, pers, cnt = 0.0, 0.0, 0
        with torch.no_grad():
            for b in range(0, len(s_va), 1024):
                w = take(Xt, s_va[b:b + 1024], t_va[b:b + 1024])
                tot += float((model(w[:, 0]) - w[:, 1:]).abs().mean()) * len(w)
                pers += float((w[:, :1] - w[:, 1:]).abs().mean()) * len(w); cnt += len(w)
        return tot / cnt, pers / cnt

    rng = np.random.default_rng(a.seed); log = open(os.path.join(out, "train.log"), "w")
    best, best_ep, best_state, t0, pers = np.inf, -1, None, time.time(), float("nan")
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(s_tr)); tl = []
        model.train()
        for b in range(0, len(order), BATCH):
            sel = order[b:b + BATCH]; w = take(Xt, s_tr[sel], t_tr[sel])
            loss = (model(w[:, 0]) - w[:, 1:]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(float(loss))
        model.eval(); v, pers = val_loss()
        log.write("epoch %d train_l1 %.6f val_l1 %.6f %.0fs\n" % (ep, float(np.mean(tl)), v, time.time() - t0)); log.flush()
        if not np.isfinite(v):
            break
        if v < best:
            best, best_ep = v, ep; best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
    failed = best_state is None
    if not failed:
        model.load_state_dict(best_state)
    W = model.graph(); np.save(os.path.join(out, "W.npy"), W)
    sc = sgode_scores(W, d) if np.isfinite(W).all() else dict(auroc=float("nan"), auprc=float("nan"), sign_acc=float("nan"), sign_acc_flip=float("nan"))
    m = dict(method="sgode", variant=a.variant, **sc, failed=bool(failed or not np.isfinite(W).all()), best_epoch=best_ep, val_l1_best=best,
             val_l1_persistence=pers, epochs=epochs, solver=method, step_size=step, n_traj=len(keep), wall_sec=time.time() - t0,
             hidden=HIDDEN, embed=EMBED, lr=LR, weight_decay=WEIGHT_DECAY, batch=BATCH, horizon=HORIZON, args=vars(a))
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}, default=float))


if __name__ == "__main__":
    main()
