"""ARNI-style baseline (Casadiego et al., Nat. Commun. 8, 2192, 2017): model-free reconstruction of direct
interactions by expanding each node's derivative in self and pairwise basis functions and greedily adding the
incoming block that most reduces the least-squares residual.

dx_i/dt ~ Theta_self(x_i) a_i + sum_{j in sel_i} B(x_i, x_j) b_ij

Derivatives are central differences on the training trajectories (same split as hetndl.train).

mode "full" (default): one joint least-squares fit over the self block and all N - 1 pair blocks, with columns
standardised; an edge's score is the norm of its block's coefficients. Used as the reported baseline because it is
much stronger than the greedy search here (Kuramoto N = 100, sigma/K 0.15: AUROC 0.998 vs 0.876): with many weak
inputs (K / deg ~ 0.2) a single source's marginal gain does not separate true from false sources.
mode "greedy" (the published search): each node runs k_max greedy steps; edge scores rank selected sources by entry order and the remaining sources by their residual
gain in the last round (always below the selected ones). The sign of an edge is the sign of the coefficient on the
first pair-basis column, sin(x_j - x_i) for "fourier" and x_j (first coordinate) for "poly"; for an unselected
source it is the coefficient it would get if added to the final model (Frisch-Waugh-Lovell).
Full mode also returns W_jac: same magnitudes, sign of the Jacobian d(pair term)/d x_j at x_j = x_i, needed when the
interaction is multiplicative (GLV x_i x_j) and the first column carries no sign.
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from ..data import add_noise, select_trajectories, trajectory_split
from ..gen.kuramoto import load_dataset
from ..metrics import auroc_auprc, sign_accuracy, spearman_z


def self_basis(x: np.ndarray, basis: str) -> np.ndarray:
    """(rows, d) -> (rows, m_s). fourier: 1, sin x, cos x (d = 1). poly: all monomials up to total degree 3."""
    if basis == "fourier":
        return np.concatenate([np.ones_like(x[:, :1]), np.sin(x), np.cos(x)], 1)
    d = x.shape[1]
    cols = [np.ones(len(x))]
    for deg in (1, 2, 3):
        for idx in _combos(d, deg):
            cols.append(np.prod(x[:, idx], 1))
    return np.stack(cols, 1)


def _combos(d: int, deg: int) -> List[List[int]]:
    if deg == 0:
        return [[]]
    return [c + [k] for c in _combos(d, deg - 1) for k in range(d) if not c or k >= c[-1]]


def pair_basis(xi: np.ndarray, xj: np.ndarray, basis: str) -> np.ndarray:
    """(rows, d) x 2 -> (rows, m_p); column 0 carries the sign. fourier: sin(xj-xi), cos(xj-xi), sin xj, cos xj.
    poly, per coordinate k: xj_k, xj_k^2, xj_k^3, xi_k xj_k."""
    if basis == "fourier":
        dx = xj[:, 0] - xi[:, 0]
        return np.stack([np.sin(dx), np.cos(dx), np.sin(xj[:, 0]), np.cos(xj[:, 0])], 1)
    cols = []
    for k in range(xi.shape[1]):
        cols += [xj[:, k], xj[:, k] ** 2, xj[:, k] ** 3, xi[:, k] * xj[:, k]]
    return np.stack(cols, 1)


def derivatives(data: np.ndarray, dt: float, smooth: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """States and time derivatives, (S, T', N, d) each. smooth = 0: central differences (T' = T - 2).
    smooth = odd window w: Savitzky-Golay (cubic) smoothing and derivative, both ends trimmed (T' = T - w + 1),
    which is what a careful ARNI user would do on noisy series."""
    x = np.asarray(data, dtype=np.float64)
    if smooth and smooth > 1:
        from scipy.signal import savgol_filter
        if smooth % 2 == 0:
            raise ValueError("smooth must be an odd window length")
        h = smooth // 2
        X = savgol_filter(x, smooth, 3, axis=1)[:, h:-h]
        Y = savgol_filter(x, smooth, 3, deriv=1, delta=dt, axis=1)[:, h:-h]
        return X, Y
    return x[:, 1:-1], (x[:, 2:] - x[:, :-2]) / (2 * dt)


def _rows(data: np.ndarray, dt: float, traj: np.ndarray, max_rows: int, rng, smooth: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    x, y = derivatives(data[traj], dt, smooth)             # (S, T', N, d)
    X, Y = x.reshape(-1, *x.shape[2:]), y.reshape(-1, *y.shape[2:])
    if len(X) > max_rows:
        keep = rng.choice(len(X), max_rows, replace=False)
        X, Y = X[keep], Y[keep]
    return X, Y                                             # (rows, N, d) each


def _orth(M: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(M)
    return q[:, np.abs(np.diag(r)) > 1e-10 * max(1.0, np.abs(r).max())]


def _full_fit(S: np.ndarray, blocks: List[np.ndarray], yi: np.ndarray, ridge: float = 0.0):
    """Joint (ridge) least squares on standardised columns; returns beta in the original column scale,
    the column means and stds (prediction: B @ beta - mu @ beta)."""
    B = np.concatenate([S] + blocks, 1)
    mu, sd = B.mean(0), B.std(0)
    const = sd < 1e-12
    mu[const], sd[const] = 0.0, 1.0                         # keep the intercept column as is
    Bs = (B - mu) / sd
    if ridge > 0:
        pen = np.full(Bs.shape[1], float(ridge)); pen[const] = 0.0   # no penalty on the intercept
        beta = np.linalg.solve(Bs.T @ Bs + np.diag(pen), Bs.T @ yi)
    else:
        beta = np.linalg.lstsq(Bs, yi, rcond=None)[0]
    return beta / sd[:, None], mu, sd


def _full_node(S: np.ndarray, blocks: List[np.ndarray], yi: np.ndarray, ridge: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Joint fit; returns (signed block scores, self coefficients in the original scale)."""
    beta, mu, sd = _full_fit(S, blocks, yi, ridge)
    m_s, m_p = S.shape[1], blocks[0].shape[1]
    b = beta[m_s:].reshape(len(blocks), m_p, -1)
    score = np.sqrt(((b * sd[m_s:].reshape(len(blocks), m_p, 1)) ** 2).sum((1, 2))) * np.sign(b[:, 0, 0])
    self_c = beta[:m_s].copy()
    self_c[0] -= mu @ beta                                  # column 0 of the self basis is the constant
    return score, self_c, b


RIDGE_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _select_ridge(Xtr, Ytr, Xva, Yva, basis: str) -> float:
    """Global ridge strength by validation MSE of the joint fit, summed over nodes."""
    N = Xtr.shape[1]
    best, best_err = 0.0, np.inf
    for lam in RIDGE_GRID:
        err = 0.0
        for i in range(N):
            cand = [j for j in range(N) if j != i]
            S, Sv = self_basis(Xtr[:, i], basis), self_basis(Xva[:, i], basis)
            Btr = [pair_basis(Xtr[:, i], Xtr[:, j], basis) for j in cand]
            Bva = np.concatenate([Sv] + [pair_basis(Xva[:, i], Xva[:, j], basis) for j in cand], 1)
            beta, mu, _ = _full_fit(S, Btr, Ytr[:, i], lam)
            err += float(((Bva @ beta - mu @ beta - Yva[:, i]) ** 2).mean())
        if err < best_err:
            best, best_err = lam, err
    return best


def arni(data: np.ndarray, dt: float, basis: str = "poly", k_max: int = 10, seed: int = 1,
         max_rows: int = 8000, mode: str = "full", n_traj: Optional[int] = None, noise_std: Optional[float] = None,
         noise_snr: Optional[float] = None, ridge=0.0, smooth: int = 0) -> Tuple[np.ndarray, Dict]:
    if mode not in ("full", "greedy"):
        raise ValueError("unknown mode: %s" % mode)
    rng = np.random.default_rng(seed)
    # same trajectory budget, noise and split as hetndl.train
    keep = select_trajectories(data.shape[0], n_traj, seed)
    data = add_noise(data[keep], np.random.default_rng(seed + 12345), noise_snr, noise_std)
    train, val, _ = trajectory_split(data.shape[0], seed)
    X, Y = _rows(data, dt, train, max_rows, rng, smooth)
    _, N, d = X.shape
    if ridge == "auto":
        if mode != "full":
            raise ValueError("ridge='auto' needs mode='full'")
        Xv, Yv = _rows(data, dt, val, max_rows, rng, smooth)
        ridge = _select_ridge(X, Y, Xv, Yv, basis)
    ridge = float(ridge)
    W = np.zeros((N, N))
    W_jac = np.zeros((N, N))                                # same magnitudes, sign from the Jacobian (full mode)
    selected, self_coef = [], []
    for i in range(N):
        xi, yi = X[:, i], Y[:, i]                           # (rows, d)
        S = self_basis(xi, basis)
        cand = [j for j in range(N) if j != i]
        blocks = {j: pair_basis(xi, X[:, j], basis) for j in cand}
        if mode == "full":
            score, self_c, b = _full_node(S, [blocks[j] for j in cand], yi, ridge)
            W[cand, i] = score
            # Jacobian sign: row mean of d(pair term)/d x_j evaluated at x_j = x_i (influence near identical states;
            # the row mean at the observed x_j averages cos(theta_j - theta_i) to ~0 on Kuramoto). Same for every j.
            h = 1e-4 * (1.0 + np.abs(xi[:, 0]).mean())
            xp, xm = xi.copy(), xi.copy(); xp[:, 0] += h; xm[:, 0] -= h
            dphi = (pair_basis(xi, xp, basis) - pair_basis(xi, xm, basis)).mean(0) / (2 * h)
            for c_, j in enumerate(cand):
                W_jac[j, i] = abs(score[c_]) * np.sign(dphi @ b[c_, :, 0])
            selected.append([])
            self_coef.append(self_c.T.ravel())
            continue
        sel: List[int] = []
        gain = {}
        for step in range(min(k_max, len(cand)) + 1):
            Q = _orth(np.concatenate([S] + [blocks[j] for j in sel], 1))
            R = yi - Q @ (Q.T @ yi)
            rest = [j for j in cand if j not in sel]
            if not rest:
                break
            Bs = np.stack([blocks[j] for j in rest])        # (c, rows, m)
            Bt = Bs - Q[None] @ (Q.T[None] @ Bs)
            q, r = np.linalg.qr(Bt)
            qR = q.transpose(0, 2, 1) @ R[None]             # (c, m, d)
            g = (qR ** 2).sum((1, 2))
            if step == min(k_max, len(cand)):               # final round: rank and sign the remaining sources
                coef = np.linalg.pinv(r) @ qR               # FWL: coefficients of B_j in the joint fit
                top = g.max() if g.max() > 0 else 1.0
                for c, j in enumerate(rest):
                    W[j, i] = 0.99 * g[c] / top * np.sign(coef[c, 0, 0])
                break
            sel.append(rest[int(np.argmax(g))])
        full = np.concatenate([S] + [blocks[j] for j in sel], 1)
        beta = np.linalg.lstsq(full, yi, rcond=None)[0]      # (m_s + m_p k, d)
        m_s, m_p = S.shape[1], (blocks[cand[0]].shape[1] if cand else 0)
        for s, j in enumerate(sel):
            W[j, i] = (len(sel) + 1 - s) * np.sign(beta[m_s + s * m_p, 0])
        selected.append(sel)
        self_coef.append(beta[:m_s].T.ravel())
    return W, dict(selected=selected, self_coef=np.array(self_coef), basis=basis, mode=mode, rows=len(X), W_jac=W_jac,
                   ridge=ridge, ridge_grid=list(RIDGE_GRID), smooth=int(smooth))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="ARNI-style baseline")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis", choices=["auto", "poly", "fourier"], default="auto",
                    help="auto: fourier for Kuramoto phases, poly otherwise")
    ap.add_argument("--mode", choices=["full", "greedy"], default="full")
    ap.add_argument("--k_max", type=int, default=10, help="greedy mode only")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_rows", type=int, default=50000, help="cap on regression rows (the default keeps all rows of the benchmarks)")
    ap.add_argument("--cluster_k", type=int, default=0, help="0: number of distinct classes")
    ap.add_argument("--n_traj", type=int, default=None)
    ap.add_argument("--noise_std", type=float, default=None)
    ap.add_argument("--noise_snr", type=float, default=None)
    ap.add_argument("--ridge", default="0", help="ridge strength on standardised columns, or 'auto' (validation)")
    ap.add_argument("--smooth", type=int, default=0, help="odd Savitzky-Golay window for states and derivatives")
    a = ap.parse_args(argv)
    ridge = "auto" if a.ridge == "auto" else float(a.ridge)
    d = load_dataset(a.data)
    basis = a.basis if a.basis != "auto" else ("fourier" if d["meta"].get("system") == "kuramoto" else "poly")
    W, info = arni(d["data"], float(d["meta"]["dt"]), basis, a.k_max, a.seed, a.max_rows, a.mode,
                   n_traj=a.n_traj, noise_std=a.noise_std, noise_snr=a.noise_snr, ridge=ridge, smooth=a.smooth)
    auroc, auprc = auroc_auprc(W, d["A"])
    m = dict(basis=basis, mode=a.mode, k_max=a.k_max, n_traj=a.n_traj, noise_std=a.noise_std, noise_snr=a.noise_snr,
             ridge=info["ridge"], smooth=info["smooth"], rows=info["rows"], auroc=auroc, auprc=auprc,
             sign_acc=sign_accuracy(W, d["S"]) if (d["S"] < 0).any() else float("nan"),
             sign_acc_jac=sign_accuracy(info["W_jac"], d["S"]) if (d["S"] < 0).any() and a.mode == "full" else float("nan"),
             spearman_omega=float("nan"), cluster_ari=float("nan"), args=vars(a))
    omega, classes = d["omega"], d["classes"]
    if np.isfinite(omega).all() and np.std(omega) > 0:
        m["spearman_omega"] = spearman_z(info["self_coef"][:, :1], omega)   # constant self term ~ omega_i
    if int(classes.max()) >= 1:
        K = a.cluster_k or int(len(np.unique(classes)))
        C = info["self_coef"]
        C = (C - C.mean(0)) / (C.std(0) + 1e-12)
        m["cluster_ari"] = float(adjusted_rand_score(classes, KMeans(K, n_init=20, random_state=0).fit_predict(C)))
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "W.npy"), W)
    if a.mode == "full":
        np.save(os.path.join(a.out, "W_jac.npy"), info["W_jac"])
    np.save(os.path.join(a.out, "self_coef.npy"), info["self_coef"])
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}))


if __name__ == "__main__":
    main()
