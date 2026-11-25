"""Linear Granger-causality baseline: multivariate vector autoregression with an L1 (lasso) penalty.
x_i(t+1) ~ sum_{l=1..L} sum_j B_{ij}^{(l)} x_j(t+1-l); the score of the directed edge j -> i is the summed |coef| over
lags and state dimensions; one lag by default (the same one-step formulation as every other method here; on the toy
system a second lag absorbs the deterministic drive and ruins the ranking although it lowers validation MSE). Classic model-free reference for fMRI (Smith et al. 2011) and noisy series. The lasso
strength is chosen on the held-out VALIDATION trajectories (same split as training), never on the truth.
Interface mirrors the other baselines: granger_lasso(data, ...) -> (W, info) with W[j, i], and a CLI writing
metrics.json / W.npy."""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Lasso

from ..data import add_noise, select_trajectories, smooth_series, trajectory_split
from ..gen.kuramoto import load_dataset
from ..metrics import auroc_auprc

ALPHA_GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def lagged_design(data: np.ndarray, traj_idx: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """(S, T, N, d) -> X (rows, lags*N*d) with the most recent lag first, Y (rows, N*d)."""
    X, Y = [], []
    for s in traj_idx:
        x = data[s].reshape(data.shape[1], -1)
        for t in range(lags, data.shape[1]):
            X.append(np.concatenate([x[t - l] for l in range(1, lags + 1)])); Y.append(x[t])
    return np.asarray(X, np.float64), np.asarray(Y, np.float64)


def granger_lasso(data: np.ndarray, seed: int = 1, lags: int = 1, alphas=ALPHA_GRID, n_traj: Optional[int] = None,
                  noise_std: Optional[float] = None, noise_snr: Optional[float] = None, max_rows: int = 50000,
                  smooth_window: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    keep = select_trajectories(data.shape[0], n_traj, seed)
    obs = add_noise(data[keep], np.random.default_rng(seed + 12345), noise_snr, noise_std)
    obs = smooth_series(obs, smooth_window)                          # the Savitzky-Golay filter of hetndl.data (train.py)
    tr, va, _ = trajectory_split(len(keep), seed)
    Xtr, Ytr = lagged_design(obs, tr, lags); Xva, Yva = lagged_design(obs, va, lags)
    if len(Xtr) > max_rows:
        sel = np.random.default_rng(seed).choice(len(Xtr), max_rows, replace=False); Xtr, Ytr = Xtr[sel], Ytr[sel]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    N, d = obs.shape[2], obs.shape[3]
    best = None
    for a in alphas:
        model = Lasso(alpha=a, fit_intercept=True, max_iter=20000, tol=1e-3).fit(Xtr, Ytr)
        mse = float(np.mean((model.predict(Xva) - Yva) ** 2))
        if best is None or mse < best[0]:
            best = (mse, a, model.coef_.copy())
    coef = best[2].reshape(N, d, lags, N, d)                          # [target i, dim, lag, source j, dim]
    W = np.abs(coef).sum((1, 2, 4)).T                                  # W[j, i]
    np.fill_diagonal(W, 0.0)
    return W, dict(alpha=best[1], val_mse=best[0], lags=lags, rows=len(Xtr), n_traj=len(keep))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Lasso-VAR Granger baseline")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--lags", type=int, default=1)
    ap.add_argument("--n_traj", type=int, default=None); ap.add_argument("--noise_std", type=float, default=None)
    ap.add_argument("--noise_snr", type=float, default=None); ap.add_argument("--max_rows", type=int, default=50000)
    ap.add_argument("--smooth_window", type=int, default=None)
    a = ap.parse_args(argv)
    d = load_dataset(a.data)
    W, info = granger_lasso(d["data"], a.seed, a.lags, n_traj=a.n_traj, noise_std=a.noise_std, noise_snr=a.noise_snr, max_rows=a.max_rows,
                       smooth_window=a.smooth_window)
    auroc, auprc = auroc_auprc(W, d["A"])
    m = dict(method="granger_lasso", auroc=auroc, auprc=auprc, sign_acc=float("nan"), **info, args=vars(a))
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "W.npy"), W)
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}))


if __name__ == "__main__":
    main()
