"""Random graph generators.

Convention (shared by the whole package): A[j, i] = 1 means an edge j -> i,
i.e. matrices are stored [source, target]; the diagonal is always zero.
"""
from typing import Optional

import networkx as nx
import numpy as np


def make_graph(N: int, kind: str, seed: int, p: Optional[float] = None, m: int = 1,
               directed: bool = False) -> np.ndarray:
    if p is None:
        p = 0.1 if N <= 10 else 0.04
    if kind == "ER":
        G = nx.erdos_renyi_graph(N, p, seed=seed, directed=directed)
    elif kind == "BA":
        G = nx.barabasi_albert_graph(N, m, seed=seed)
    else:
        raise ValueError("unknown graph kind: %s" % kind)
    A = nx.to_numpy_array(G, nodelist=range(N), dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float32)
