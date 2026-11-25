"""Spring network on one fixed undirected graph: the "Spring" system of AIDD (Zhang et al., Phys. Rev. E 106, 034315,
2022, Table II), i.e. the springs of NRI (Kipf et al., ICML 2018) without NRI's reflecting box. All nodes share the
same dynamics; the supplement uses it to check the NRI and AIDD re-implementations in AIDD's own setting.

Graph: undirected Erdos-Renyi on N nodes, each pair an edge with probability p (AIDD: 20 % for small networks, to
avoid isolated nodes); drawn with numpy.random.default_rng(seed) and redrawn with seed + 1000 r (r = 1, 2, ...) until
no node is isolated (checked on the graph only).
Dynamics: dr_i/dt = v_i, dv_i/dt = -k sum_j A_ij (r_i - r_j), k = 0.1, r, v in R^2. Initial states
r(0) ~ N(0, 0.5^2) per coordinate, v(0) of random direction and norm 0.5 (NRI's springs), drawn with
numpy.random.default_rng(10 seed + 2). Exact solution (matrix exponential) every dt time units; node state
(r_x, r_y, v_x, v_y).

Usage: python -m hetndl.gen.springs --N 10 --p 0.2 --samples 1000 --steps 50 --dt 0.1 --seed 1 \
           --out experiments/data/spring10_seed1.npz"""
import argparse

import numpy as np
from scipy.linalg import expm

from .kuramoto import save_dataset


def graph(seed: int, N: int = 10, p: float = 0.2):
    """(A, r): symmetric 0/1 adjacency without isolated nodes and the number of redraws r it took."""
    r = 0
    while True:
        rng = np.random.default_rng(seed + 1000 * r)
        iu = np.triu_indices(N, 1); e = rng.random(len(iu[0])) < p
        A = np.zeros((N, N)); A[iu[0][e], iu[1][e]] = 1.0; A = np.maximum(A, A.T)
        if A.sum(0).min() > 0:
            return A, r
        r += 1


def spring(A: np.ndarray, rng, samples: int = 1000, steps: int = 50, kk: float = 0.1, dt: float = 0.1) -> np.ndarray:
    """(samples, steps, N, 4) trajectories of the spring network with adjacency A, sampled every dt."""
    N = A.shape[0]
    L = np.diag(A.sum(0)) - A
    M = np.block([[np.zeros((N, N)), np.eye(N)], [-kk * L, np.zeros((N, N))]]); Phi = expm(M * dt)
    r0 = rng.normal(0, 0.5, (samples, 2, N)); v = rng.normal(size=(samples, 2, N))
    v0 = v / np.linalg.norm(v, axis=1, keepdims=True) * 0.5
    y = np.concatenate([r0, v0], 2)                                                   # (samples, 2 coords, 2N)
    out = np.zeros((samples, steps, N, 4))
    for t in range(steps):
        out[:, t, :, 0:2] = y[:, :, :N].transpose(0, 2, 1); out[:, t, :, 2:4] = y[:, :, N:].transpose(0, 2, 1)
        y = y @ Phi.T
    return out


def simulate_springs(N: int, p: float, samples: int, steps: int, dt: float, k: float, seed: int) -> dict:
    A, r = graph(seed, N, p)
    X = spring(A, np.random.default_rng(10 * seed + 2), samples, steps, k, dt)
    meta = dict(system="spring", dt=dt, T=steps * dt, N=N, samples=samples, seed=seed, n_edges=int(A.sum() // 2),
                graph_redraws=r, edge_prob=p, source="hetndl.gen.springs (AIDD, PRE 2022, Table II)")
    return dict(data=X.astype(np.float64), A=A, S=A.copy(), omega=np.zeros(N), classes=np.zeros(N, int), meta=meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Spring network dataset generator (one fixed undirected graph)")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--p", type=float, default=0.2, help="edge probability of the undirected graph")
    ap.add_argument("--samples", type=int, default=1000, help="number of trajectories")
    ap.add_argument("--steps", type=int, default=50, help="stored states per trajectory")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--k", type=float, default=0.1, help="spring constant")
    ap.add_argument("--seed", type=int, default=1, help="graph seed; the initial states use 10 seed + 2")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args(argv)
    d = simulate_springs(a.N, a.p, a.samples, a.steps, a.dt, a.k, a.seed)
    save_dataset(a.out, d)
    print("saved", a.out, d["data"].shape, "edges", d["meta"]["n_edges"], "redraws", d["meta"]["graph_redraws"])


if __name__ == "__main__":
    main()
