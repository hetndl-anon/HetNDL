"""Neural Relational Inference baseline (Kipf, Fetaya, Wang, Welling, Zemel, ICML 2018), re-implemented from the paper:
an MLP encoder maps each window of trajectories to a categorical distribution over K edge types for every ordered node
pair (node -> edge -> node -> edge message passing), a Gumbel-softmax sample of the types gates an MLP decoder that
predicts the next states (multi-step, teacher forcing every `pred_steps`), and the ELBO is a Gaussian NLL plus the KL
to a uniform prior over types (output variance = mean squared one-step increment of the training data unless given,
so that the NLL is O(1) per element in any state units; the paper used a fixed 5e-5). Type 0 is "no interaction" (its decoder message is dropped, as in the paper).

Our graphs are static, so the edge score reported for pair (sender j, receiver i) is the posterior probability of the
"interaction" type averaged over all windows of the training trajectories; snapshot = epoch with the lowest validation
NLL (same truth-free rule as the protocol). Interface mirrors hetndl.baselines.arni: nri(data, ...) -> (W, info) with
W[j, i] = P(edge j -> i), and a CLI writing metrics.json / W.npy."""
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


def _mlp(n_in: int, n_hid: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_hid), nn.ELU(), nn.Linear(n_hid, n_out), nn.ELU(), nn.BatchNorm1d(n_out))


class _BN(nn.Module):
    """BatchNorm over the last dim of a (B, E, H) tensor."""
    def __init__(self, mlp):
        super().__init__(); self.mlp = mlp
    def forward(self, x):
        B, E, H = x.shape
        return self.mlp(x.reshape(B * E, H)).reshape(B, E, -1)


def _pairs(N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    send, rec = zip(*[(j, i) for j in range(N) for i in range(N) if i != j])
    return torch.tensor(send), torch.tensor(rec)


class NRIEncoder(nn.Module):
    """x: (B, N, T*d) -> edge-type logits (B, N(N-1), K); pair order = (sender j, receiver i), j != i."""
    def __init__(self, n_in: int, hidden: int, n_types: int, N: int):
        super().__init__()
        self.N = N
        send, rec = _pairs(N)
        self.register_buffer("send", send); self.register_buffer("rec", rec)
        self.mlp1 = _BN(_mlp(n_in, hidden, hidden))
        self.mlp2 = _BN(_mlp(2 * hidden, hidden, hidden))
        self.mlp3 = _BN(_mlp(hidden, hidden, hidden))
        self.mlp4 = _BN(_mlp(3 * hidden, hidden, hidden))   # [h_j, h_i] of the second pass + skip
        self.fc_out = nn.Linear(hidden, n_types)

    def node2edge(self, h):
        return torch.cat([h[:, self.send], h[:, self.rec]], -1)

    def edge2node(self, e):
        out = torch.zeros(e.shape[0], self.N, e.shape[2], device=e.device, dtype=e.dtype)
        out.index_add_(1, self.rec, e)
        return out / (self.N - 1)

    def forward(self, x):
        h = self.mlp1(x)
        e = self.mlp2(self.node2edge(h)); skip = e
        h = self.mlp3(self.edge2node(e))
        e = self.mlp4(torch.cat([self.node2edge(h), skip], -1))
        return self.fc_out(e)

    def edge_matrix(self, p_edge):
        """(B, E) -> (B, N, N) with M[b, j, i] = p for pair (sender j, receiver i); diagonal 0."""
        M = torch.zeros(p_edge.shape[0], self.N, self.N, device=p_edge.device, dtype=p_edge.dtype)
        M[:, self.send, self.rec] = p_edge
        return M


class NRIDecoder(nn.Module):
    """One-step MLP decoder gated by edge types; type 0 carries no message. x: (B, N, T, d), z: (B, E, K)."""
    def __init__(self, d: int, hidden: int, n_types: int, N: int):
        super().__init__()
        self.N, self.d, self.K = N, d, n_types
        send, rec = _pairs(N)
        self.register_buffer("send", send); self.register_buffer("rec", rec)
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
                                  for _ in range(n_types - 1)])
        self.out = nn.Sequential(nn.Linear(d + hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d))

    def step(self, x, z):
        pre = torch.cat([x[:, self.send], x[:, self.rec]], -1)                       # (B, E, 2d)
        agg = 0
        for k, m in enumerate(self.msg):
            agg = agg + m(pre) * z[..., k + 1:k + 2]
        node = torch.zeros(x.shape[0], self.N, agg.shape[-1], device=x.device, dtype=x.dtype)
        node.index_add_(1, self.rec, agg)
        node = node / (self.N - 1)
        return x + self.out(torch.cat([x, node], -1))

    def forward(self, x, z, pred_steps: int):
        B, N, T, d = x.shape
        preds = []
        cur = x[:, :, 0]
        for t in range(T - 1):
            if t % pred_steps == 0:
                cur = x[:, :, t]                                                      # teacher forcing
            cur = self.step(cur, z)
            preds.append(cur)
        return torch.stack(preds, 2)                                                  # (B, N, T-1, d)


def windows(data: np.ndarray, traj_idx: np.ndarray, win: int) -> np.ndarray:
    """(S, T, N, d) -> (n_windows, N, win, d): non-overlapping windows of `win` steps from the given trajectories."""
    out = []
    for s in traj_idx:
        T = data.shape[1]
        for a in range(0, T - win + 1, win):
            out.append(data[s, a:a + win].transpose(1, 0, 2))
    return np.stack(out).astype(np.float32)


def gaussian_nll(pred: torch.Tensor, target: torch.Tensor, var: float, N: int, norm: str = "mean") -> torch.Tensor:
    """Gaussian NLL of (batch, N, T, d) predictions. 'mean': per node, per predicted time step and summed over dims (the
    default of this re-implementation); 'sum_time': summed over time steps and dims and divided by batch x N, as in the
    reference code of the NRI paper (nll_gaussian), which weighs the reconstruction T times more against the KL term."""
    nll = ((pred - target) ** 2 / (2 * var)).sum((1, 2, 3)).mean() / N
    if norm == "mean":
        return nll / target.shape[2]
    if norm != "sum_time":
        raise ValueError(norm)
    return nll


def minmax(obs: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """Every input feature scaled to [-1, 1] with the minimum and maximum over the training trajectories (the data loader
    of the NRI paper)."""
    lo = obs[tr].reshape(-1, obs.shape[-1]).min(0); hi = obs[tr].reshape(-1, obs.shape[-1]).max(0)
    return (2 * (obs - lo) / np.maximum(hi - lo, 1e-12) - 1).astype(np.float32)


def nri(data: np.ndarray, seed: int = 1, epochs: int = 200, hidden: int = 64, win: int = 50, pred_steps: int = 10,
        lr: float = 5e-4, batch: int = 32, var: Optional[float] = None, tau: float = 0.5, n_types: int = 2,
        n_traj: Optional[int] = None, noise_std: Optional[float] = None, noise_snr: Optional[float] = None,
        device: str = "cpu", log=None, feat: str = "raw", loss_norm: str = "mean", input_norm: str = "none",
        node_feat: Optional[np.ndarray] = None, per_sample: bool = False) -> Tuple[np.ndarray, Dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    keep = select_trajectories(data.shape[0], n_traj, seed)
    obs = add_noise(data[keep], np.random.default_rng(seed + 12345), noise_snr, noise_std)
    obs = phase_features(obs, feat)
    if node_feat is not None:                 # input check only: a per-node input (the NRI paper's omega), uses the truth
        nf = np.asarray(node_feat, np.float32)
        nf = nf[None, None, :, None] if nf.ndim == 1 else nf[keep][:, None, :, None]       # (N,) or one row per trajectory
        obs = np.concatenate([obs, np.broadcast_to(nf, obs.shape[:3] + (1,))], -1)
    tr, va, te = trajectory_split(len(keep), seed)
    if input_norm == "minmax":
        obs = minmax(obs, tr)
    elif input_norm != "none":
        raise ValueError(input_norm)
    win = int(min(win, obs.shape[1]))
    Xtr, Xva = windows(obs, tr, win), windows(obs, va, win)
    N, d = obs.shape[2], obs.shape[3]
    if var is None:   # data-adaptive output variance: mean squared one-step increment of the training windows, so the
        var = float(np.mean(np.diff(Xtr, axis=2) ** 2)) + 1e-12   # NLL is O(1) per element whatever the state units
    var = float(var)
    dev = torch.device(device)
    enc = NRIEncoder(win * d, hidden, n_types, N).to(dev)
    dec = NRIDecoder(d, hidden, n_types, N).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    Xtr_t, Xva_t = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    log_prior = math.log(1.0 / n_types)

    def losses(xb, train: bool):
        logits = enc(xb.reshape(xb.shape[0], N, -1))
        z = F.gumbel_softmax(logits, tau=tau, hard=False) if train else F.softmax(logits, -1)
        pred = dec(xb, z, pred_steps)
        nll = gaussian_nll(pred, xb[:, :, 1:], var, N, loss_norm)
        prob = F.softmax(logits, -1)
        kl = (prob * (torch.log(prob + 1e-16) - log_prior)).sum((1, 2)).mean() / N
        return nll, kl, prob

    best = dict(val_nll=math.inf, epoch=0, P=None); hist = []
    for ep in range(1, epochs + 1):
        enc.train(); dec.train()
        perm = torch.randperm(len(Xtr_t))
        tl = 0.0
        for a in range(0, len(perm), batch):
            xb = Xtr_t[perm[a:a + batch]].to(dev)
            if xb.shape[0] < 2:
                continue
            nll, kl, _ = losses(xb, True)
            loss = nll + kl
            opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss) * xb.shape[0]
        enc.eval(); dec.eval()
        with torch.no_grad():
            vn, vk, ps = 0.0, 0.0, []
            for a in range(0, len(Xva_t), batch):
                xb = Xva_t[a:a + batch].to(dev); nll, kl, _ = losses(xb, False)
                vn += float(nll) * xb.shape[0]; vk += float(kl) * xb.shape[0]
            vn /= max(len(Xva_t), 1); vk /= max(len(Xva_t), 1)
            if vn < best["val_nll"]:
                for a in range(0, len(Xtr_t), batch):
                    xb = Xtr_t[a:a + batch].to(dev); _, _, prob = losses(xb, False); ps.append(prob[..., 1:].sum(-1))
                P = enc.edge_matrix(torch.cat(ps).mean(0, keepdim=True))[0].cpu().numpy()
                best = dict(val_nll=vn, epoch=ep, P=P, enc={k: v.detach().clone() for k, v in enc.state_dict().items()})
        hist.append((ep, tl / max(len(Xtr_t), 1), vn, vk))
        if log is not None:
            log(ep, tl / max(len(Xtr_t), 1), vn, vk)
    W = best["P"].astype(np.float64); np.fill_diagonal(W, 0.0)
    extra = {}
    if per_sample:                       # NRI-paper regime: one graph per trajectory; posterior of every test trajectory (best encoder)
        enc.load_state_dict(best["enc"]); enc.eval()
        Xte = windows(obs, te, win); assert len(Xte) == len(te), "per-sample scoring needs one window per trajectory"
        with torch.no_grad():
            ps = [F.softmax(enc(torch.from_numpy(Xte[a:a + batch]).to(dev).reshape(-1, N, Xte.shape[2] * d)), -1)[..., 1:].sum(-1)
                  for a in range(0, len(Xte), batch)]
            extra = dict(P_test=enc.edge_matrix(torch.cat(ps)).cpu().numpy(), test_index=keep[te])
    return W, dict(**extra, best_epoch=best["epoch"], val_nll_best=best["val_nll"], n_windows=(len(Xtr), len(Xva)), win=win,
                   history=hist, n_traj=len(keep), var=var, feat=feat, state_dim=int(obs.shape[3]), loss_norm=loss_norm,
                   input_norm=input_norm)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="NRI baseline (MLP encoder / decoder, K = 2 edge types)")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--win", type=int, default=50)
    ap.add_argument("--pred_steps", type=int, default=10); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--var", type=float, default=None, help="Gaussian output variance; default: mean squared increment")
    ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--n_types", type=int, default=2)
    ap.add_argument("--n_traj", type=int, default=None); ap.add_argument("--noise_std", type=float, default=None)
    ap.add_argument("--noise_snr", type=float, default=None); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--feat", choices=["raw", "trig", "nri", "nri_omega"], default="raw",
                    help="trig: states as (cos, sin) (phases); nri / nri_omega: input check on the NRI-paper Kuramoto only (nri_omega reads the true omega)")
    ap.add_argument("--loss_norm", choices=["mean", "sum_time"], default="mean")
    ap.add_argument("--input_norm", choices=["none", "minmax"], default="none")
    a = ap.parse_args(argv)
    d = load_dataset(a.data)
    os.makedirs(a.out, exist_ok=True)
    lf = open(os.path.join(a.out, "log.csv"), "w"); lf.write("epoch,train_loss,val_nll,val_kl\n")
    def log(ep, tl, vn, vk):
        lf.write("%d,%.6g,%.6g,%.6g\n" % (ep, tl, vn, vk)); lf.flush()
    per_sample = "A_all" in d                 # the NRI-paper regime, one graph per trajectory
    nf = (d["omega_all"] if per_sample else d["omega"]) if a.feat == "nri_omega" else None
    W, info = nri(d["data"], a.seed, a.epochs, a.hidden, a.win, a.pred_steps, a.lr, a.batch, a.var, a.tau, a.n_types,
                  a.n_traj, a.noise_std, a.noise_snr, a.device, log, a.feat, a.loss_norm, a.input_norm, nf, per_sample)
    lf.close()
    if per_sample:                            # pooled over test trajectories and ordered pairs, truth only here
        P, At = info["P_test"], d["A_all"][info["test_index"]]
        off = ~np.eye(P.shape[1], dtype=bool)
        from sklearn.metrics import auc, precision_recall_curve, roc_curve      # as hetndl.metrics.auroc_auprc, pooled
        y, sc = (At[:, off] > 0).astype(int).ravel(), P[:, off].astype(np.float64).ravel()
        fpr, tpr, _ = roc_curve(y, sc); pr, rc, _ = precision_recall_curve(y, sc)
        auroc, auprc = float(auc(fpr, tpr)), float(auc(rc, pr))
        extra = dict(per_sample_graph=True, n_test=int(len(P)), acc=float(((P[:, off] > 0.5) == (At[:, off] > 0)).mean()))
        np.save(os.path.join(a.out, "P_test.npy"), P)
    else:
        auroc, auprc = auroc_auprc(W, d["A"]); extra = {}
    m = dict(method="nri", auroc=auroc, auprc=auprc, sign_acc=float("nan"), best_epoch=info["best_epoch"], **extra,
             val_nll_best=info["val_nll_best"], n_windows=info["n_windows"], win=info["win"], n_traj=info["n_traj"], var=info["var"],
             state_dim=info["state_dim"],
             noise_std=a.noise_std, noise_snr=a.noise_snr, args=vars(a))
    np.save(os.path.join(a.out, "W.npy"), W)
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}))


if __name__ == "__main__":
    main()
