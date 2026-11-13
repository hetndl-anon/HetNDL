"""A3: A2's FHN / Van der Pol node mixture with TWO edge coupling types, chosen by the source node's class.

source class 0 (FHN):  kappa * (u_j - u_i)              linear, gap-junction-like
source class 1 (VdP):  kappa * tanh(2 (u_j - u_i))      saturating
Both vanish at u_j = u_i (diffusive), so the endpoint-type coupling G(x_i, x_j, z_i, z_j) can represent them.
"""
import argparse
import json
from typing import Optional

import numpy as np
from scipy.integrate import odeint

from .a2_mixed import sample_classes2, self_field
from .graphs import make_graph
from .kuramoto import save_dataset, signed_matrix

COUPLING_TEXT = {0: "kappa (u_j - u_i)", 1: "kappa tanh(2 (u_j - u_i))"}


def edge_types_from_classes(A: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """(N, N) int: type of edge j -> i = class of the source j; -1 where there is no edge."""
    E = np.full(A.shape, -1, dtype=np.int64)
    src, dst = np.nonzero(A > 0)
    E[src, dst] = np.asarray(classes)[src]
    return E


LAWS = {"lin": lambda du: du, "tanh": lambda du: np.tanh(2.0 * du), "sin": np.sin}
LAW_TEXT = {"lin": "u_j - u_i", "tanh": "tanh(2 (u_j - u_i))", "sin": "sin(u_j - u_i)"}


def edge_coupling(du: np.ndarray, etype, laws: str = "lin,tanh") -> np.ndarray:
    """Coupling as a function of u_j - u_i; etype is a scalar or an array broadcastable to du. laws = the law of
    source class 0 and of source class 1 (default: linear and saturating, the MixOsc-E benchmark). A linear law on
    an inhibitory edge makes these oscillators unstable, so the signed data sets use two bounded laws."""
    l0, l1 = laws.split(",")
    return np.where(np.asarray(etype) == 0, LAWS[l0](du), LAWS[l1](du))


def _rhs(y, t, classes, A, E, kappa, laws="lin,tanh"):   # A: signed adjacency S when edges can be inhibitory
    x = y.reshape(-1, 2)
    u = x[:, 0]
    f = self_field(u, x[:, 1], classes)
    du = u[None, :] - u[:, None]                      # du[i, j] = u_j - u_i
    c = edge_coupling(du, E.T, laws)                   # E.T[i, j] = type of edge j -> i
    f[:, 0] += kappa * (A.T * c).sum(axis=1)
    return f.ravel()


def simulate_a3(A: np.ndarray, classes: np.ndarray, kappa: float, samples: int, T: float, dt: float,
                seed: int, x0: Optional[np.ndarray] = None, x0_range: float = 2.5, neg_frac: float = 0.0,
                laws: str = "lin,tanh", x_max: float = 10.0) -> dict:
    rng = np.random.default_rng(seed)
    A = A.astype(np.float32)
    # neg_frac > 0: that fraction of the edges is inhibitory (S[j, i] = -1), drawn independently of the node classes
    # and hence of the coupling type of the edge. signed_matrix draws nothing when neg_frac <= 0 (default unchanged).
    S = signed_matrix(A, neg_frac, rng).astype(np.float32)
    classes = np.asarray(classes, dtype=np.int64)
    E = edge_types_from_classes(A, classes)
    N = A.shape[0]
    tspan = np.arange(0, T, dt)
    out = np.empty((samples, len(tspan), N, 2), np.float32)
    for s in range(samples):
        xi = np.asarray(x0, dtype=np.float64) if x0 is not None else rng.uniform(-x0_range, x0_range, (N, 2))
        sol = odeint(_rhs, xi.ravel(), tspan, args=(classes, S.astype(np.float64), E, float(kappa), laws))
        out[s] = sol.reshape(len(tspan), N, 2)
    assert np.isfinite(out).all() and np.abs(out).max() < x_max, "A3 trajectories diverged"
    meta = dict(system="a3", dt=float(dt), T=float(T), kappa=float(kappa), seed=int(seed), N=int(N),
                samples=int(samples), n_edges=int((A > 0).sum()),
                class_counts=[int((classes == k).sum()) for k in range(2)],
                edge_type_counts=[int((E == k).sum()) for k in range(2)],
                couplings={str(k): v for k, v in COUPLING_TEXT.items()})
    d = dict(data=out, A=A, S=S, omega=np.full(N, np.nan, dtype=np.float32), classes=classes, etypes=E, meta=meta)
    if neg_frac > 0:
        meta.update(neg_frac=float(neg_frac), n_neg=int((S < 0).sum())); d["W_true"] = (float(kappa) * S).astype(np.float32)
    if laws != "lin,tanh":
        meta.update(laws=laws, couplings={str(k): "kappa * " + LAW_TEXT[l] for k, l in enumerate(laws.split(","))})
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="A3: FHN / VdP nodes with source-class-dependent coupling")
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
    ap.add_argument("--laws", default="lin,tanh", help="coupling law of source class 0 and 1: lin | tanh | sin")
    ap.add_argument("--x_max", type=float, default=10.0, help="trajectories beyond this magnitude count as diverged")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args(argv)
    gseed = a.seed if a.graph_seed is None else a.graph_seed
    A = make_graph(a.N, "ER", seed=gseed, p=a.p, directed=a.directed)
    classes = sample_classes2(a.N, seed=a.seed)
    d = simulate_a3(A, classes, a.kappa, a.samples, a.T, a.dt, a.seed, neg_frac=a.neg_frac, laws=a.laws, x_max=a.x_max)
    save_dataset(a.out, d)
    print(json.dumps(d["meta"]))


if __name__ == "__main__":
    main()
