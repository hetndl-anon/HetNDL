"""AIDD / GGN-style baseline (Zhang et al., "A general deep learning framework for network reconstruction and dynamics
learning", Appl. Netw. Sci. 2019; Zhang et al., PRE 2022): ONE global adjacency distribution (a categorical
"edge / no edge" logit per ordered node pair), sampled with Gumbel-softmax each step and gating a shared dynamics
learner that predicts the next states; trained end to end on next-step prediction. This is the static-graph relative of
NRI (whose encoder re-infers the graph from every window): same decoder, same ELBO form (Gaussian NLL with the
data-adaptive variance + KL to a uniform prior), the graph parameters just live in a free logit table.
Edge score = P(edge) of the logit table at the epoch with the lowest validation NLL. Interface mirrors nri.py."""
import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import add_noise, phase_features, select_trajectories, trajectory_split
from ..gen.kuramoto import load_dataset
from ..metrics import auroc_auprc
from .nri import gaussian_nll, minmax, NRIDecoder, _pairs, windows


class StaticGumbelGraph(nn.Module):
    """A logit per ordered pair (sender j, receiver i), two types: 0 = no edge, 1 = edge."""
    def __init__(self, N: int, init: float = 0.0):
        super().__init__()
        self.N = N
        send, rec = _pairs(N)
        self.register_buffer("send", send); self.register_buffer("rec", rec)
        self.logits = nn.Parameter(torch.full((N * (N - 1), 2), init))

    def sample(self, tau: float, hard: bool):
        return F.gumbel_softmax(self.logits, tau=tau, hard=hard)

    def prob(self):
        return F.softmax(self.logits, -1)

    def edge_matrix(self):
        M = torch.zeros(self.N, self.N, device=self.logits.device, dtype=self.logits.dtype)
        M[self.send, self.rec] = self.prob()[:, 1]
        return M


def aidd(data: np.ndarray, seed: int = 1, epochs: int = 200, hidden: int = 64, win: int = 50, pred_steps: int = 10,
         lr: float = 5e-4, lr_graph: float = 1e-2, batch: int = 32, var: Optional[float] = None, tau: float = 0.5,
         n_traj: Optional[int] = None, noise_std: Optional[float] = None, noise_snr: Optional[float] = None,
         device: str = "cpu", log=None, feat: str = "raw", loss_norm: str = "mean", input_norm: str = "none") -> Tuple[np.ndarray, Dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    keep = select_trajectories(data.shape[0], n_traj, seed)
    obs = add_noise(data[keep], np.random.default_rng(seed + 12345), noise_snr, noise_std)
    obs = phase_features(obs, feat)
    tr, va, _ = trajectory_split(len(keep), seed)
    if input_norm == "minmax":
        obs = minmax(obs, tr)
    elif input_norm != "none":
        raise ValueError(input_norm)
    win = int(min(win, obs.shape[1]))
    Xtr, Xva = windows(obs, tr, win), windows(obs, va, win)
    N, d = obs.shape[2], obs.shape[3]
    if var is None:
        var = float(np.mean(np.diff(Xtr, axis=2) ** 2)) + 1e-12
    var = float(var)
    dev = torch.device(device)
    graph = StaticGumbelGraph(N).to(dev); dec = NRIDecoder(d, hidden, 2, N).to(dev)
    opt = torch.optim.Adam([dict(params=dec.parameters(), lr=lr), dict(params=graph.parameters(), lr=lr_graph)])
    Xtr_t, Xva_t = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    log_prior = math.log(0.5)

    def losses(xb, train: bool):
        z = graph.sample(tau, False) if train else graph.prob()
        z = z.unsqueeze(0).expand(xb.shape[0], -1, -1)
        pred = dec(xb, z, pred_steps)
        nll = gaussian_nll(pred, xb[:, :, 1:], var, N, loss_norm)
        prob = graph.prob()
        kl = (prob * (torch.log(prob + 1e-16) - log_prior)).sum() / N
        return nll, kl

    best = dict(val_nll=math.inf, epoch=0, P=None); hist = []
    for ep in range(1, epochs + 1):
        dec.train(); perm = torch.randperm(len(Xtr_t)); tl = 0.0
        for a in range(0, len(perm), batch):
            xb = Xtr_t[perm[a:a + batch]].to(dev)
            nll, kl = losses(xb, True); loss = nll + kl
            opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss) * xb.shape[0]
        dec.eval()
        with torch.no_grad():
            vn = 0.0
            for a in range(0, len(Xva_t), batch):
                xb = Xva_t[a:a + batch].to(dev); nll, kl = losses(xb, False); vn += float(nll) * xb.shape[0]
            vn /= max(len(Xva_t), 1)
            if vn < best["val_nll"]:
                best = dict(val_nll=vn, epoch=ep, P=graph.edge_matrix().cpu().numpy())
        hist.append((ep, tl / max(len(Xtr_t), 1), vn))
        if log is not None:
            log(ep, tl / max(len(Xtr_t), 1), vn)
    W = best["P"].astype(np.float64); np.fill_diagonal(W, 0.0)
    return W, dict(best_epoch=best["epoch"], val_nll_best=best["val_nll"], n_windows=(len(Xtr), len(Xva)), win=win,
                   history=hist, n_traj=len(keep), var=var, feat=feat, state_dim=int(obs.shape[3]), loss_norm=loss_norm,
                   input_norm=input_norm)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="AIDD / GGN-style static Gumbel-adjacency baseline")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--win", type=int, default=50)
    ap.add_argument("--pred_steps", type=int, default=10); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_graph", type=float, default=1e-2); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--var", type=float, default=None); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n_traj", type=int, default=None); ap.add_argument("--noise_std", type=float, default=None)
    ap.add_argument("--noise_snr", type=float, default=None); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--feat", choices=["raw", "trig"], default="raw", help="trig: states as (cos, sin) (phases)")
    ap.add_argument("--loss_norm", choices=["mean", "sum_time"], default="mean")
    ap.add_argument("--input_norm", choices=["none", "minmax"], default="none")
    a = ap.parse_args(argv)
    d = load_dataset(a.data)
    os.makedirs(a.out, exist_ok=True)
    lf = open(os.path.join(a.out, "log.csv"), "w"); lf.write("epoch,train_loss,val_nll\n")
    def log(ep, tl, vn):
        lf.write("%d,%.6g,%.6g\n" % (ep, tl, vn)); lf.flush()
    W, info = aidd(d["data"], a.seed, a.epochs, a.hidden, a.win, a.pred_steps, a.lr, a.lr_graph, a.batch, a.var, a.tau,
                   a.n_traj, a.noise_std, a.noise_snr, a.device, log, a.feat, a.loss_norm, a.input_norm)
    lf.close()
    auroc, auprc = auroc_auprc(W, d["A"])
    m = dict(method="aidd", auroc=auroc, auprc=auprc, sign_acc=float("nan"), best_epoch=info["best_epoch"],
             val_nll_best=info["val_nll_best"], n_windows=info["n_windows"], win=info["win"], n_traj=info["n_traj"],
             var=info["var"], noise_std=a.noise_std, noise_snr=a.noise_snr, args=vars(a))
    np.save(os.path.join(a.out, "W.npy"), W)
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}))


if __name__ == "__main__":
    main()
