"""MixRelax (internal name A1): nodes with one of three scalar relaxation mechanisms.

f_0(x) = -x
f_1(x) = -x - 0.5 x^3
f_2(x) = -1.2 x + 0.6 tanh(x)
dx_i/dt = f_{c_i}(x_i) + kappa * sum_j S[j, i] (tanh(x_j) - tanh(x_i))

S = A (all edges excitatory) by default; --neg_frac flips that fraction of the edges to S[j, i] = -1 (inhibitory),
as in the signed Kuramoto and GLV generators. W_true[j, i] = kappa * S[j, i].

--coef_spread d (default 0; one check of the supplement) gives every node its own coefficients of its class law,
f_0 = -a x, f_1 = -a x - b x^3, f_2 = -a x + b tanh(x), with (a, b) = the class values of BASE_COEF times independent
factors U[1 - d, 1 + d], drawn from an RNG stream separate from the simulation, so graph, classes and initial states
are those of the d = 0 data set. The coefficients are stored as "coef" (N, 2). d = 0 leaves the output unchanged.
"""
import argparse
import json
from typing import Optional

import numpy as np
from scipy.integrate import odeint

from .graphs import make_graph
from .kuramoto import save_dataset, signed_matrix

MECHANISM_TEXT = {0: "-x", 1: "-x - 0.5*x^3", 2: "-1.2*x + 0.6*tanh(x)"}


def sample_classes(N: int, seed: int, n_classes: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        c = rng.integers(0, n_classes, N)
        if len(np.unique(c)) == n_classes:
            return c.astype(np.int64)


BASE_COEF = np.array([[1.0, 0.0], [1.0, 0.5], [1.2, 0.6]])   # (a, b) per class, see the module docstring


def node_coefficients(classes: np.ndarray, spread: float, seed: int) -> np.ndarray:
    """(N, 2) coefficients (a, b) of each node's law: class values times U[1 - spread, 1 + spread] factors, from an RNG
    stream of its own (the simulation's draws are unchanged)."""
    base = BASE_COEF[np.asarray(classes, dtype=np.int64)]
    if spread <= 0:
        return base.copy()
    rng = np.random.default_rng([int(seed), 20261113])
    return base * rng.uniform(1.0 - spread, 1.0 + spread, base.shape)


def self_term(x: np.ndarray, classes: np.ndarray, coef: Optional[np.ndarray] = None) -> np.ndarray:
    if coef is not None:                         # per-element coefficients, shape x.shape + (2,)
        a, b = coef[..., 0], coef[..., 1]
        return np.where(classes == 0, -a * x, np.where(classes == 1, -a * x - b * x ** 3, -a * x + b * np.tanh(x)))
    f0 = -x
    f1 = -x - 0.5 * x ** 3
    f2 = -1.2 * x + 0.6 * np.tanh(x)
    return np.where(classes == 0, f0, np.where(classes == 1, f1, f2))


def _rhs(x, t, classes, S, kappa, coef=None):
    th = np.tanh(x)
    diff = th[None, :] - th[:, None]            # diff[i, j] = tanh(x_j) - tanh(x_i)
    inter = kappa * (S.T * diff).sum(axis=1)    # sum_j S[j, i] * diff[i, j]
    return self_term(x, classes, coef) + inter


def simulate_a1(A: np.ndarray, classes: np.ndarray, kappa: float, samples: int, T: float, dt: float,
                seed: int, x0: Optional[np.ndarray] = None, x0_range: float = 3.0, neg_frac: float = 0.0,
                coef_spread: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    A = A.astype(np.float32)
    # signed_matrix draws nothing from rng when neg_frac <= 0, so the default output is unchanged
    S = signed_matrix(A, neg_frac, rng).astype(np.float32)
    classes = np.asarray(classes, dtype=np.int64)
    N = A.shape[0]
    coef = node_coefficients(classes, coef_spread, seed) if coef_spread > 0 else None
    tspan = np.arange(0, T, dt)
    out = np.empty((samples, len(tspan), N, 1), np.float32)
    for s in range(samples):
        xi = np.asarray(x0, dtype=np.float64) if x0 is not None else rng.uniform(-x0_range, x0_range, N)
        sol = odeint(_rhs, xi, tspan, args=(classes, S.astype(np.float64), float(kappa), coef))
        out[s, :, :, 0] = sol
    assert np.isfinite(out).all() and np.abs(out).max() < 10, "A1 trajectories left the dissipative region"
    meta = dict(system="a1", dt=float(dt), T=float(T), kappa=float(kappa), seed=int(seed), N=int(N),
                samples=int(samples), n_edges=int((A > 0).sum()),
                neg_frac=float(neg_frac), n_neg=int((S < 0).sum()),
                class_counts=[int((classes == k).sum()) for k in range(3)],
                mechanisms={str(k): v for k, v in MECHANISM_TEXT.items()})
    d = dict(data=out, A=A, S=S, omega=np.full(N, np.nan, dtype=np.float32),
             classes=classes, meta=meta, W_true=(float(kappa) * S).astype(np.float32))
    if coef is not None:                         # only then, so that the default output stays byte-identical
        meta["coef_spread"] = float(coef_spread)
        d["coef"] = coef.astype(np.float32)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="A1 mixed-mechanism dataset generator")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--graph", choices=["ER", "BA"], default="ER", help="BA (m) is undirected, ignores --p / --directed")
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=18)
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--graph_seed", type=int, default=None)
    ap.add_argument("--neg_frac", type=float, default=0.0, help="fraction of edges made inhibitory (S[j, i] = -1)")
    ap.add_argument("--coef_spread", type=float, default=0.0,
                    help="node coefficients = class values x U[1 - d, 1 + d] (0 = identical within a class)")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args(argv)
    gseed = a.seed if a.graph_seed is None else a.graph_seed
    A = make_graph(a.N, a.graph, seed=gseed, p=a.p, m=a.m, directed=a.directed)
    classes = sample_classes(a.N, seed=a.seed)
    d = simulate_a1(A, classes, a.kappa, a.samples, a.T, a.dt, a.seed, neg_frac=a.neg_frac, coef_spread=a.coef_spread)
    d["meta"].update(graph=a.graph, m=int(a.m) if a.graph == "BA" else None)
    save_dataset(a.out, d)
    print(json.dumps(d["meta"]))


if __name__ == "__main__":
    main()
