"""Evaluation metrics. Ground truth enters only here."""
from typing import Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, auc, adjusted_rand_score
from sklearn.mixture import GaussianMixture


def offdiag(M: np.ndarray) -> np.ndarray:
    N = M.shape[0]
    return M[~np.eye(N, dtype=bool)]


def auroc_auprc(W: np.ndarray, A: np.ndarray) -> Tuple[float, float]:
    s = np.abs(offdiag(W)).astype(np.float64)
    y = (offdiag(A) > 0).astype(int)
    fpr, tpr, _ = roc_curve(y, s)
    p, r, _ = precision_recall_curve(y, s)
    return float(auc(fpr, tpr)), float(auc(r, p))


def sign_accuracy(W: np.ndarray, S: np.ndarray) -> float:
    E = S != 0
    if E.sum() == 0:
        return float("nan")
    return float((np.sign(W[E]) == np.sign(S[E])).mean())


def spurious_per_node(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """False positives per TARGET node when the top-|E| off-diagonal |W| entries are edges."""
    N = A.shape[0]
    true = A > 0
    np.fill_diagonal(true, False)
    E = int(true.sum())
    score = np.abs(W).astype(np.float64).copy()
    np.fill_diagonal(score, -np.inf)
    flat = np.argsort(score, axis=None)[::-1][:E]
    pred = np.zeros((N, N), dtype=bool)
    pred.flat[flat] = True
    fp = pred & ~true
    return fp.sum(axis=0)  # per target (column) node


def spearman_z(Z: np.ndarray, param: np.ndarray) -> float:
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        z1 = Z
    elif Z.shape[1] == 1:
        z1 = Z[:, 0]
    else:
        z1 = PCA(n_components=1).fit_transform(Z)[:, 0]
    rho = spearmanr(z1, param).correlation
    return float(abs(rho)) if np.isfinite(rho) else float("nan")


def cluster_ari(Z: np.ndarray, classes: np.ndarray, K: int) -> float:
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    gmm = GaussianMixture(n_components=K, n_init=5, random_state=0, covariance_type="full")
    pred = gmm.fit_predict(Z)
    return float(adjusted_rand_score(classes, pred))


@torch.no_grad()
def rollout_nrmse(ode, W: torch.Tensor, data: np.ndarray, traj_idx: np.ndarray, steps: int, device) -> float:
    """Autoregressive one-step rollout from t=0 of each trajectory in traj_idx."""
    steps = int(min(steps, data.shape[1] - 1))
    x = torch.from_numpy(data[traj_idx, 0]).to(device)
    preds = []
    for _ in range(steps):
        x = ode(x, W)
        preds.append(x)
    pred = torch.stack(preds, dim=1).cpu().numpy()           # (n, steps, N, d)
    true = data[traj_idx, 1:steps + 1]
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    std = float(np.std(true)) + 1e-12
    return rmse / std
