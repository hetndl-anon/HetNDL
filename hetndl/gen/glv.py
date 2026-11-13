"""Generalised Lotka-Volterra with signed, degree-normalised interactions (second signed benchmark).

dx_i/dt = x_i ( r_i - x_i + (c / k_i) sum_j S[j, i] x_j ),   S[j, i] in {+1, -1} on the edges j -> i

r_i ~ U(0.5, 1.5) is the node parameter (saved as `omega`), k_i the in-degree (>= 1), c < 1 keeps the interaction
matrix strictly diagonally dominant so trajectories stay positive and bounded. The coupling x_i x_j is multiplicative,
NOT diffusive (it does not vanish at x_j = x_i): with the diffusive form G(x_i, x_j) - G(x_i, x_i) the leftover
-x_i^2 sum_j W_ij is node-specific and has to be absorbed by F(x_i, z_i). W_true[j, i] = c S[j, i] / k_i.
Random initial abundances U(0.05, 2.5) and short windows keep most samples in the transient."""
import argparse
import json
from typing import Optional

import numpy as np
from scipy.integrate import odeint

from .graphs import make_graph
from .kuramoto import save_dataset, signed_matrix


def _rhs(x, t, r, Wt):
    return x * (r - x + Wt.T @ x)          # (Wt.T @ x)_i = sum_j Wt[j, i] x_j


def simulate_glv(A: np.ndarray, c: float, neg_frac: float, samples: int, T: float, dt: float, seed: int,
                 x0: Optional[np.ndarray] = None, x0_range=(0.05, 2.5)) -> dict:
    rng = np.random.default_rng(seed)
    A = (A > 0).astype(np.float32); N = A.shape[0]
    r = rng.uniform(0.5, 1.5, N)
    S = signed_matrix(A, neg_frac, rng).astype(np.float32)
    k = np.maximum(A.sum(axis=0), 1.0).astype(np.float64)
    Wt = float(c) * S.astype(np.float64) / k[None, :]
    tspan = np.arange(0, T, dt)
    out = np.empty((samples, len(tspan), N, 1), np.float32)
    for s in range(samples):
        xi = np.asarray(x0, dtype=np.float64) if x0 is not None else rng.uniform(*x0_range, N)
        out[s, :, :, 0] = odeint(_rhs, xi, tspan, args=(r, Wt))
    assert np.isfinite(out).all() and out.min() > 0 and out.max() < 10, "GLV trajectories left the positive bounded region"
    meta = dict(system="glv", dt=float(dt), T=float(T), c=float(c), neg_frac=float(neg_frac), seed=int(seed), N=int(N),
                samples=int(samples), n_edges=int(A.sum()), x_tail_mean=float(out[:, -1].mean()))
    return dict(data=out, A=A, S=S, omega=r.astype(np.float32), classes=np.full(N, -1, dtype=np.int64), meta=meta,
                W_true=Wt.astype(np.float32))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Signed generalised Lotka-Volterra dataset generator")
    ap.add_argument("--N", type=int, default=30); ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--c", type=float, default=0.5); ap.add_argument("--neg_frac", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=40); ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--graph_seed", type=int, default=None); ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--graph", type=str, default="ER", choices=["ER", "BA"],
                    help="ER: directed G(N, p) (the benchmark); BA: undirected Barabasi-Albert with m edges per new node")
    ap.add_argument("--m", type=int, default=2)
    a = ap.parse_args(argv)
    gs = a.seed if a.graph_seed is None else a.graph_seed
    A = make_graph(a.N, "ER", seed=gs, p=a.p, directed=True) if a.graph == "ER" else make_graph(a.N, "BA", seed=gs, m=a.m)
    d = simulate_glv(A, a.c, a.neg_frac, a.samples, a.T, a.dt, a.seed)
    if a.graph == "BA":
        d["meta"].update(graph="BA", m=int(a.m))
    save_dataset(a.out, d)
    print(json.dumps(d["meta"]))


if __name__ == "__main__":
    main()
