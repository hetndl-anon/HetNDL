"""Power grid on a REAL topology (MATPOWER case files, e.g. IEEE 118-bus) with heterogeneous node dynamics:
the structure-preserving swing model with every node carrying a phase and a frequency,

    d theta_i / dt = omega_i
    d omega_i / dt = (P_i - D omega_i) / M_i + sum_j (kappa K_ij / M_i) sin(theta_j - theta_i)

generator buses: inertia M = m_gen (class 1); load buses: M = m_load << m_gen (class 0, "motor loads").
P_i = net injected power in p.u. (balanced so that the synchronous state is at omega = 0), K_ij = line
susceptance 1 / x with parallel lines summed. In the additive form the receiver scale 1 / M_i sits in the true
edge weight W[j, i] = kappa K_ij / M_i, so the true W is asymmetric although the line graph is undirected.
The case file is parsed as DATA (text matrices), never executed."""
import argparse
import json
import re
from typing import Dict, Optional

import numpy as np
from scipy.integrate import odeint

from .kuramoto import save_dataset


def _matrix(text: str, name: str) -> np.ndarray:
    m = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % re.escape(name), text, re.S)
    if m is None:
        raise ValueError("no mpc.%s in case file" % name)
    rows = []
    for line in m.group(1).splitlines():
        line = line.split("%")[0].strip().rstrip(";")
        if line:
            rows.append([float(v) for v in line.replace(",", " ").split()])
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows], dtype=np.float64)


def parse_matpower(path: str) -> Dict:
    text = open(path).read()
    base = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)\s*;", text)
    return dict(baseMVA=float(base.group(1)) if base else 100.0, bus=_matrix(text, "bus"),
                gen=_matrix(text, "gen"), branch=_matrix(text, "branch"))


def build_grid(case: Dict, balance: bool = True) -> Dict:
    """K (N, N) summed susceptances of in-service lines, P (N,) net injection in p.u. (balanced over loads),
    is_gen (N,) buses with a generator entry, bus_ids in file order."""
    bus, gen, br, base = case["bus"], case["gen"], case["branch"], case["baseMVA"]
    ids = [int(b[0]) for b in bus]
    idx = {b: i for i, b in enumerate(ids)}
    N = len(ids)
    K = np.zeros((N, N))
    for b in br:
        if b.shape[0] > 10 and b[10] == 0:  # out of service
            continue
        i, j = idx[int(b[0])], idx[int(b[1])]
        if i != j and b[3] != 0:
            K[i, j] += 1.0 / abs(b[3]); K[j, i] += 1.0 / abs(b[3])
    Pg = np.zeros(N); is_gen = np.zeros(N, bool)
    for g in gen:
        if g.shape[0] > 7 and g[7] == 0:
            continue
        i = idx[int(g[0])]; Pg[i] += g[1]; is_gen[i] = True
    Pd = bus[:, 2]
    P = (Pg - Pd) / base
    if balance and abs(P.sum()) > 1e-12:
        w = Pd / Pd.sum() if Pd.sum() > 0 else np.full(N, 1.0 / N)
        P = P - P.sum() * w
    return dict(K=K, P=P, is_gen=is_gen, bus_ids=ids)


def _rhs(y, t, P, M, D, KK):
    x = y.reshape(-1, 2)
    th, om = x[:, 0], x[:, 1]
    coup = (KK * np.sin(th[None, :] - th[:, None])).sum(1)     # sum_j K_ij sin(theta_j - theta_i)
    return np.stack([om, (P - D * om + coup) / M], -1).ravel()


def equilibrium(g: Dict, kappa: float, M: np.ndarray, damping: float, T: float = 60.0) -> np.ndarray:
    """Synchronous state found by relaxing from theta = omega = 0; raises when the grid does not settle."""
    N = len(g["P"])
    sol = odeint(_rhs, np.zeros(2 * N), np.linspace(0, T, 601), args=(g["P"], M, damping, kappa * g["K"]))
    x = sol[-1].reshape(N, 2)
    if not np.isfinite(x).all() or np.abs(x[:, 1]).max() > 1e-2:
        raise RuntimeError("grid does not synchronise at kappa=%g (max |omega| %.3g)" % (kappa, np.abs(x[:, 1]).max()))
    x[:, 0] -= x[:, 0].mean()
    return x


def simulate_grid(g: Dict, kappa: float, m_gen: float, m_load: float, damping: float, samples: int, T: float,
                  dt: float, seed: int, x0: Optional[np.ndarray] = None, theta_pert: float = 0.3,
                  omega_pert: float = 0.3) -> Dict:
    rng = np.random.default_rng(seed)
    N = len(g["P"])
    M = np.where(g["is_gen"], m_gen, m_load).astype(np.float64)
    KK = kappa * g["K"]
    tspan = np.arange(0, T, dt)
    out = np.empty((samples, len(tspan), N, 2), np.float32)
    eq = None if x0 is not None else equilibrium(g, kappa, M, damping)
    for s in range(samples):
        if x0 is not None:
            xi = np.asarray(x0, dtype=np.float64)
        else:  # perturb the synchronous state: trajectories are the transients back to it
            xi = eq + np.stack([rng.uniform(-theta_pert, theta_pert, N), rng.normal(0, omega_pert, N)], -1)
        sol = odeint(_rhs, xi.ravel(), tspan, args=(g["P"], M, damping, KK))
        out[s] = sol.reshape(len(tspan), N, 2)
    assert np.isfinite(out).all(), "grid trajectories diverged"
    A = (g["K"] > 0).astype(np.float32)
    W_true = (KK / M[None, :]).astype(np.float32)                 # W[j, i] = kappa K_ij / M_i
    meta = dict(system="powergrid", dt=float(dt), T=float(T), kappa=float(kappa), m_gen=m_gen, m_load=m_load,
                damping=damping, seed=int(seed), N=int(N), samples=int(samples), n_edges=int(A.sum()),
                n_gen=int(g["is_gen"].sum()), bus_ids=[int(b) for b in g["bus_ids"]],
                note="classes: 1 generator (M=m_gen), 0 load (M=m_load); omega field = net power P_i (p.u.)")
    return dict(data=out, A=A, S=A.copy(), omega=g["P"].astype(np.float32), classes=g["is_gen"].astype(np.int64),
                W_true=W_true, meta=meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description="swing dynamics on a MATPOWER topology")
    ap.add_argument("--case", required=True, help="MATPOWER .m file (parsed as text)")
    # IEEE 118 defaults: kappa 0.3 keeps line angles up to 1.2 rad (sin regime), m_load 0.2
    # makes loads 5x faster than generators, |omega| p99 ~ 1, trajectories are 10 s transients to the sync state.
    ap.add_argument("--kappa", type=float, default=0.3)
    ap.add_argument("--m_gen", type=float, default=1.0)
    ap.add_argument("--m_load", type=float, default=0.2)
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    g = build_grid(parse_matpower(a.case))
    d = simulate_grid(g, a.kappa, a.m_gen, a.m_load, a.damping, a.samples, a.T, a.dt, a.seed)
    save_dataset(a.out, d)
    print(json.dumps({k: v for k, v in d["meta"].items() if k != "bus_ids"}))


if __name__ == "__main__":
    main()
