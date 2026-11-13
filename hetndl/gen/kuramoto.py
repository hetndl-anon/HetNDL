"""Kuramoto generator with node heterogeneity (sigma_omega) and signed edges (neg_frac).

d theta_i/dt = omega_i + (K / k_i) * sum_j S[j, i] * sin(theta_j - theta_i)

k_i is the UNSIGNED in-degree sum_j A[j, i] (isolated nodes use k_i = 1).
"""
import argparse
import json
import os
from typing import Optional

import numpy as np
from scipy.integrate import odeint

from .graphs import make_graph


def signed_matrix(A: np.ndarray, neg_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Flip a fraction neg_frac of edges to -1 (per undirected edge when A is symmetric)."""
    S = A.copy().astype(np.float32)
    if neg_frac <= 0:
        return S
    sym = bool(np.allclose(A, A.T))
    if sym:
        ju, iu = np.triu_indices(A.shape[0], k=1)
        keep = A[ju, iu] > 0
        edges = list(zip(ju[keep], iu[keep]))
    else:
        edges = list(zip(*np.where(A > 0)))
    n_neg = int(round(neg_frac * len(edges)))
    for idx in rng.permutation(len(edges))[:n_neg]:
        j, i = edges[idx]
        S[j, i] = -1.0
        if sym:
            S[i, j] = -1.0
    return S


def _rhs(theta, t, omega, S, Kdeg):
    diff = np.sin(theta[None, :] - theta[:, None])  # diff[i, j] = sin(theta_j - theta_i)
    inter = (S.T * diff).sum(axis=1)                # sum_j S[j, i] * diff[i, j]
    return omega + Kdeg * inter


def order_parameter_tail(data: np.ndarray, frac: float = 0.2) -> float:
    """Mean Kuramoto order parameter |<exp(i theta)>| over the last `frac` of time steps."""
    Tn = data.shape[1]
    n_tail = max(1, int(round(frac * Tn)))
    th = data[:, -n_tail:, :, 0]
    r = np.abs(np.exp(1j * th).mean(axis=2))
    return float(r.mean())


def simulate_kuramoto(A: np.ndarray, K: float, sigma_omega: float, neg_frac: float, clip: bool,
                      samples: int, T: float, dt: float, seed: int,
                      theta0: Optional[np.ndarray] = None, S: Optional[np.ndarray] = None,
                      omega_uniform: Optional[tuple] = None, coupling_norm: str = "degree") -> dict:
    """omega_uniform = (lo, hi): omega ~ U[lo, hi) instead of 1 + sigma N(0, 1); coupling_norm = 'none': K multiplies
    the sum without dividing by the in-degree (the Kuramoto of the NRI paper: k_ij = 1, omega ~ U[1, 10))."""
    rng = np.random.default_rng(seed)
    A = A.astype(np.float32)
    N = A.shape[0]
    # omega is drawn first so that the same seed gives the same omega for every neg_frac
    omega = rng.uniform(omega_uniform[0], omega_uniform[1], N) if omega_uniform else 1.0 + sigma_omega * rng.standard_normal(N)
    if clip:
        omega = np.clip(omega, 0.1, 2.0)
    if S is None:
        S = signed_matrix(A, neg_frac, rng)
    S = S.astype(np.float32)
    k = A.sum(axis=0).astype(np.float64)
    k[k == 0] = 1.0
    if coupling_norm not in ("degree", "none"):
        raise ValueError(coupling_norm)
    Kdeg = K / k if coupling_norm == "degree" else np.full(N, float(K))
    tspan = np.arange(0, T, dt)
    out = np.empty((samples, len(tspan), N, 1), np.float32)
    for s in range(samples):
        th0 = np.asarray(theta0, dtype=np.float64) if theta0 is not None else rng.uniform(0, 2 * np.pi, N)
        sol = odeint(_rhs, th0, tspan, args=(omega, S, Kdeg))
        out[s, :, :, 0] = sol
    meta = dict(system="kuramoto", dt=float(dt), T=float(T), K=float(K), sigma_omega=float(sigma_omega),
                neg_frac=float(neg_frac), clip=bool(clip), seed=int(seed), N=int(N), samples=int(samples),
                n_edges=int((A > 0).sum()), r_tail=order_parameter_tail(out))
    if omega_uniform:
        meta["omega_uniform"] = [float(omega_uniform[0]), float(omega_uniform[1])]
    if coupling_norm != "degree":
        meta["coupling_norm"] = coupling_norm
    return dict(data=out, A=A, S=S, omega=omega.astype(np.float32),
                classes=np.full(N, -1, dtype=np.int64), meta=meta)


def simulate_kuramoto_nri_regime(n_sims: int, N: int = 5, p: float = 0.5, T: float = 5.0, dt: float = 0.1,
                                 seed: int = 1, omega_range=(1.0, 10.0), K: float = 1.0) -> dict:
    """The Kuramoto regime of the NRI paper (Kipf et al. 2018, App. B.3): EVERY simulation has its own random undirected
    graph (each pair connected with probability p, k_ij = K, no in-degree normalisation), its own omega ~ U[lo, hi) and
    its own initial phases ~ U[0, 2 pi); states every dt (odeint). Coupling written sin(theta_j - theta_i) as in every
    benchmark of this project. Returns data (S, T, N, 1), A_all (S, N, N), omega_all (S, N); A / omega hold simulation 0
    so that the usual loaders work; classes are unused."""
    rng = np.random.default_rng(seed)
    tspan = np.arange(0, T, dt)
    out = np.empty((n_sims, len(tspan), N, 1), np.float32)
    A_all = np.zeros((n_sims, N, N), np.float32); W_all = np.zeros((n_sims, N), np.float32)
    iu = np.triu_indices(N, 1)
    for s in range(n_sims):
        A = np.zeros((N, N), np.float32); e = rng.random(len(iu[0])) < p
        A[iu[0][e], iu[1][e]] = 1.0; A = np.maximum(A, A.T)
        omega = rng.uniform(omega_range[0], omega_range[1], N)
        th0 = rng.uniform(0, 2 * np.pi, N)
        out[s, :, :, 0] = odeint(_rhs, th0, tspan, args=(omega, A.astype(np.float64), np.full(N, float(K))))
        A_all[s], W_all[s] = A, omega
    meta = dict(system="kuramoto", regime="nri", dt=float(dt), T=float(T), K=float(K), p=float(p), seed=int(seed), N=int(N),
                samples=int(n_sims), omega_uniform=[float(omega_range[0]), float(omega_range[1])], coupling_norm="none")
    return dict(data=out, A=A_all[0], S=A_all[0], omega=W_all[0], classes=np.full(N, -1, dtype=np.int64), meta=meta,
                A_all=A_all, omega_all=W_all)


def save_dataset(path: str, d: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    std = ("data", "A", "S", "omega", "classes", "meta")
    extra = {k: v for k, v in d.items() if k not in std and isinstance(v, np.ndarray)}  # e.g. etypes, W_true
    np.savez_compressed(path, data=d["data"], A=d["A"], S=d["S"], omega=d["omega"],
                        classes=d["classes"], meta=json.dumps(d["meta"]), **extra)


def load_dataset(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in ("data", "A", "S", "omega", "classes")}
    for k in z.files:
        if k not in d and k != "meta":
            d[k] = z[k]
    d["meta"] = json.loads(str(z["meta"]))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kuramoto dataset generator")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--graph", type=str, default="ER", choices=["ER", "BA"])
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--K", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--neg_frac", type=float, default=0.0)
    ap.add_argument("--clip", action="store_true", help="clip omega to [0.1, 2.0] as in the NOGI paper")
    ap.add_argument("--no_clip", action="store_true", help="explicit no-op for readability")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--graph_seed", type=int, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--omega_uniform", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--coupling_norm", choices=["degree", "none"], default="degree")
    a = ap.parse_args(argv)
    gseed = a.seed if a.graph_seed is None else a.graph_seed
    A = make_graph(a.N, a.graph, seed=gseed, p=a.p, m=a.m)
    d = simulate_kuramoto(A, a.K, a.sigma, a.neg_frac, a.clip, a.samples, a.T, a.dt, a.seed,
                          omega_uniform=tuple(a.omega_uniform) if a.omega_uniform else None, coupling_norm=a.coupling_norm)
    save_dataset(a.out, d)
    print(json.dumps(d["meta"]))


if __name__ == "__main__":
    main()
