"""Signed variant of the mixed-relaxation generator (neg_frac): half of the edges inhibitory, default output unchanged."""
import os

import numpy as np
import pytest

from hetndl.gen.a1_mixed import main, sample_classes, simulate_a1
from hetndl.gen.graphs import make_graph
from hetndl.gen.kuramoto import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# generator arguments of a1v2_N30_seed1 (experiments/reproduce/datasets.json)
A1V2_SEED1 = ["--N", "30", "--p", "0.1", "--directed", "--samples", "40", "--T", "4", "--dt", "0.02", "--seed", "1",
              "--kappa", "0.5"]


def test_default_unchanged(tmp_path):
    ref_path = os.path.join(ROOT, "experiments", "data", "a1v2_N30_seed1.npz")
    if not os.path.exists(ref_path):
        pytest.skip("reference dataset a1v2_N30_seed1.npz not present")
    out = str(tmp_path / "a1v2_N30_seed1.npz")
    main(A1V2_SEED1 + ["--out", out])
    new, ref = load_dataset(out), load_dataset(ref_path)
    for k in ("data", "A", "classes"):
        assert new[k].dtype == ref[k].dtype and np.array_equal(new[k], ref[k]), k
    # without inhibitory edges the sign matrix is the adjacency and the true weights are kappa on every edge
    assert np.array_equal(new["S"], ref["A"])
    assert np.array_equal(new["W_true"], np.float32(0.5) * ref["A"])
    assert new["meta"]["neg_frac"] == 0.0


def test_half_negative():
    A = make_graph(30, "ER", seed=1, p=0.1, directed=True)
    classes = sample_classes(30, seed=1)
    d = simulate_a1(A, classes, kappa=0.5, samples=40, T=4.0, dt=0.02, seed=1, neg_frac=0.5)
    S, n_edges = d["S"], int((A > 0).sum())
    assert int((S < 0).sum()) == int(round(0.5 * n_edges))
    assert (S[A == 0] == 0).all()
    assert np.array_equal(np.abs(S), d["A"])
    assert set(np.unique(S).tolist()) == {-1.0, 0.0, 1.0}
    assert np.allclose(d["W_true"], 0.5 * S)
    assert np.isfinite(d["data"]).all() and np.abs(d["data"]).max() < 10
    assert d["meta"]["neg_frac"] == 0.5


def test_rhs_uses_sign():
    A = np.zeros((2, 2), np.float32)
    A[0, 1] = 1.0  # 0 -> 1, made inhibitory by neg_frac = 1
    classes = np.array([1, 0])  # node 0 cubic, node 1 linear
    d = simulate_a1(A, classes, kappa=0.5, samples=1, T=0.002, dt=0.001, seed=0,
                    x0=np.array([2.0, 1.0]), neg_frac=1.0)
    assert d["S"][0, 1] == -1.0 and d["S"][1, 0] == 0.0
    rate = (d["data"][0, 1, :, 0] - d["data"][0, 0, :, 0]) / 0.001
    expected0 = -2.0 - 0.5 * 8.0                                   # f_1(2), no incoming edge
    expected1 = -1.0 - 0.5 * (np.tanh(2.0) - np.tanh(1.0))         # f_0(1) - kappa (tanh x_j - tanh x_i)
    assert abs(rate[0] - expected0) < 0.05
    assert abs(rate[1] - expected1) < 0.05
    # the sign of the coupling term is what separates this from the excitatory case (difference 0.2 > tolerance)
    assert abs(rate[1] - (-1.0 + 0.5 * (np.tanh(2.0) - np.tanh(1.0)))) > 0.1
