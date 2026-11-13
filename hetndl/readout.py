"""Truth-free read-out helpers. The number of mechanism classes is never the true K: select_k chooses K from the
features alone.
binarize_2means turns edge scores into a graph without knowing the edge density (for F1)."""
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def select_k(X: np.ndarray, k_max: int = 5, n_pc: int = 3, seed: int = 0, reg: float = 0.05,
             curves: bool = False) -> Tuple[int, np.ndarray, list]:
    """Number of groups in the rows of X by the Bayesian information criterion of a Gaussian mixture (K = 1..k_max),
    after standardising columns and projecting on at most n_pc principal components (F curves have tens of
    correlated grid values). reg = covariance floor in standardised units (0.05: 15 / 15 correct on synthetic blobs with
    K = 1-3, 15-30 points each; 1e-4 lets a component collapse onto a few points and over-splits). Returns (K, labels, BICs)."""
    X = np.asarray(X, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    # curves = rows are functions sampled on a common grid in common units: centre and scale by ONE global std instead
    # of standardising each grid point (which inflates near-constant regions of the curves). Fixed a priori by data type.
    Z = (X - X.mean(0)) / ((X - X.mean(0)).std() + 1e-12) if curves else (X - X.mean(0)) / (X.std(0) + 1e-12)
    if Z.shape[1] > n_pc:
        Z = PCA(n_pc, random_state=seed).fit_transform(Z)
    k_max = int(min(k_max, len(Z) - 1))
    bics, models = [], []
    for k in range(1, k_max + 1):
        g = GaussianMixture(k, covariance_type="diag", n_init=5, random_state=seed, reg_covar=reg).fit(Z)
        bics.append(float(g.bic(Z))); models.append(g)
    # parsimony: the smallest K whose BIC is within 10 of the minimum (Kass & Raftery: Delta BIC > 10 = very strong)
    K = int(min(k for k in range(1, k_max + 1) if bics[k - 1] <= min(bics) + 10.0))
    return K, models[K - 1].predict(Z), bics


def binarize_2means(W: np.ndarray) -> np.ndarray:
    """Edges = the upper of two KMeans groups of |W| (off-diagonal). Truth-free: no density or threshold given.
    If the off-diagonal |W| takes a single value (e.g. all zero) there is no upper group and no edge is returned."""
    W = np.abs(np.asarray(W, np.float64)); N = W.shape[0]
    off = ~np.eye(N, dtype=bool)
    s = W[off][:, None]
    if s.size < 2 or np.ptp(s) == 0:
        return np.zeros_like(W, dtype=bool)
    km = KMeans(2, n_init=10, random_state=0).fit(s)
    hi = int(np.argmax(km.cluster_centers_[:, 0]))
    B = np.zeros_like(W, dtype=bool)
    B[off] = km.labels_ == hi
    return B


def node_constant(vf, states: np.ndarray, n: int = 256, seed: int = 0) -> np.ndarray:
    """Truth-free node-parameter read-out ("node constant", paper Sec. 4.5): the learned self-field F(phi(x), z_i) of an
    additive model averaged over observed states x (a fixed random subset of at most n rows of `states`, shape
    (M, node_dim)). For a node constant that enters F additively (Kuramoto omega_i) this is that constant; unlike a
    projection of z it does not depend on how z is rotated or scaled. Returns (N, node_dim)."""
    import torch
    if not vf.additive:
        raise ValueError("node_constant needs an additive model (a separate self-field F)")
    x = np.asarray(states, dtype=np.float32)
    if len(x) > n:
        x = x[np.random.default_rng(seed).choice(len(x), n, replace=False)]
    with torch.no_grad():
        h = vf.phi(torch.tensor(x))
        if not vf.use_z:
            return np.repeat(vf.F(h).mean(0, keepdim=True).numpy(), vf.N, axis=0)
        Z = vf.z_eff()
        return np.stack([vf.F(torch.cat([h, Z[i].expand(len(h), -1)], -1)).mean(0).numpy() for i in range(vf.N)])
