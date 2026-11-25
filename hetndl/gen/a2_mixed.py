"""MixOsc (internal name A2): nodes that are FitzHugh-Nagumo or Van der Pol oscillators.

class 0, FitzHugh-Nagumo:  du = u - u^3/3 - v + I,   dv = eps (u + a - b v)
class 1, Van der Pol:      du = v,                    dv = mu (1 - u^2) v - u
coupling on the first coordinate only: kappa * sum_j A[j, i] (u_j - u_i)
I = 0.5, eps = 0.08, a = 0.7, b = 0.8, mu = 1.
"""
import argparse
import json
from typing import Optional

import numpy as np
from scipy.integrate import odeint

from .graphs import make_graph
from .kuramoto import save_dataset, signed_matrix

I_EXT, EPS, A_FHN, B_FHN, MU = 0.5, 0.08, 0.7, 0.8, 1.0
MECHANISM_TEXT = {0: "FHN: (u - u^3/3 - v + 0.5, 0.08 (u + 0.7 - 0.8 v))",
                  1: "VdP: (v, (1 - u^2) v - u)"}


def sample_classes2(N: int, seed: int) -> np.ndarray:
    """Near-balanced: N // 2 nodes of class 0, the rest class 1, in random order."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.r_[np.zeros(N // 2), np.ones(N - N // 2)]).astype(np.int64)


def self_field(u: np.ndarray, v: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """(N, 2) self-term of every node."""
    fhn = np.stack([u - u ** 3 / 3 - v + I_EXT, EPS * (u + A_FHN - B_FHN * v)], -1)
    vdp = np.stack([v, MU * (1 - u ** 2) * v - u], -1)
    return np.where((np.asarray(classes) == 0)[:, None], fhn, vdp)


def _rhs(y, t, classes, A, kappa):                   # A: signed adjacency S when edges can be inhibitory
    x = y.reshape(-1, 2)
    u = x[:, 0]
    f = self_field(u, x[:, 1], classes)
    f[:, 0] += kappa * (A.T * (u[None, :] - u[:, None])).sum(axis=1)  # sum_j A[j, i] (u_j - u_i)
    return f.ravel()


def simulate_a2(A: np.ndarray, classes: np.ndarray, kappa: float, samples: int, T: float, dt: float,
                seed: int, x0: Optional[np.ndarray] = None, x0_range: float = 2.5, neg_frac: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    A = A.astype(np.float32)
    # neg_frac > 0: that fraction of the edges is inhibitory, coupling kappa * S[j, i] (u_j - u_i) with S = -1 on them
    # (as in the MixRelax generator). signed_matrix draws nothing when neg_frac <= 0: the default output is unchanged.
    S = signed_matrix(A, neg_frac, rng).astype(np.float32)
    classes = np.asarray(classes, dtype=np.int64)
    N = A.shape[0]
    tspan = np.arange(0, T, dt)
    out = np.empty((samples, len(tspan), N, 2), np.float32)
    for s in range(samples):
        xi = np.asarray(x0, dtype=np.float64) if x0 is not None else rng.uniform(-x0_range, x0_range, (N, 2))
        sol = odeint(_rhs, xi.ravel(), tspan, args=(classes, S.astype(np.float64), float(kappa)))
        out[s] = sol.reshape(len(tspan), N, 2)
    assert np.isfinite(out).all() and np.abs(out).max() < 10, "A2 trajectories diverged"
    meta = dict(system="a2", dt=float(dt), T=float(T), kappa=float(kappa), seed=int(seed), N=int(N),
                samples=int(samples), n_edges=int((A > 0).sum()),
                class_counts=[int((classes == k).sum()) for k in range(2)],
                mechanisms={str(k): v for k, v in MECHANISM_TEXT.items()})
    d = dict(data=out, A=A, S=S, omega=np.full(N, np.nan, dtype=np.float32), classes=classes, meta=meta)
    if neg_frac > 0:
        meta.update(neg_frac=float(neg_frac), n_neg=int((S < 0).sum())); d["W_true"] = (float(kappa) * S).astype(np.float32)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="A2 mixed FHN / Van der Pol dataset generator")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--kappa", type=float, default=0.3)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--graph_seed", type=int, default=None)
    ap.add_argument("--neg_frac", type=float, default=0.0, help="fraction of edges made inhibitory (S[j, i] = -1)")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args(argv)
    gseed = a.seed if a.graph_seed is None else a.graph_seed
    A = make_graph(a.N, "ER", seed=gseed, p=a.p, directed=a.directed)
    classes = sample_classes2(a.N, seed=a.seed)
    d = simulate_a2(A, classes, a.kappa, a.samples, a.T, a.dt, a.seed, neg_frac=a.neg_frac)
    save_dataset(a.out, d)
    print(json.dumps(d["meta"]))


if __name__ == "__main__":
    main()
