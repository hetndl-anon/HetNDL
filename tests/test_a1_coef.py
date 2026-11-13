"""Within-law coefficient spread of the mixed-relaxation generator (--coef_spread, a check of the supplement): every node
keeps its class law, with its own coefficients drawn around the class values; graph, classes and initial states are
those of the spread-0 data set, and the default output is unchanged."""
import numpy as np

from hetndl.gen.a1_mixed import BASE_COEF, node_coefficients, sample_classes, self_term, simulate_a1
from hetndl.gen.graphs import make_graph


def _setup():
    A = make_graph(30, "ER", seed=1, p=0.1, directed=True)
    return A, sample_classes(30, seed=1)


def test_self_term_with_base_coefficients_is_unchanged():
    x = np.linspace(-3, 3, 50)
    for k in range(3):
        cl = np.full(50, k)
        coef = np.broadcast_to(BASE_COEF[k], (50, 2))
        assert np.allclose(self_term(x, cl, coef), self_term(x, cl))


def test_coefficients_within_spread_and_reproducible():
    _, classes = _setup()
    c = node_coefficients(classes, 0.2, seed=1)
    base = BASE_COEF[classes]
    assert c.shape == (30, 2)
    nz = base != 0
    r = c[nz] / base[nz]
    assert (r >= 0.8).all() and (r <= 1.2).all() and r.std() > 0.05
    assert np.array_equal(c, node_coefficients(classes, 0.2, seed=1))
    assert not np.array_equal(c, node_coefficients(classes, 0.2, seed=2))
    assert np.array_equal(node_coefficients(classes, 0.0, seed=1), base)


def test_spread_keeps_graph_classes_and_initial_states():
    A, classes = _setup()
    d0 = simulate_a1(A, classes, kappa=0.5, samples=4, T=0.4, dt=0.02, seed=1)
    d1 = simulate_a1(A, classes, kappa=0.5, samples=4, T=0.4, dt=0.02, seed=1, coef_spread=0.2)
    assert np.array_equal(d0["data"][:, 0], d1["data"][:, 0])          # same initial states
    assert np.array_equal(d0["A"], d1["A"]) and np.array_equal(d0["classes"], d1["classes"])
    assert not np.allclose(d0["data"][:, 1:], d1["data"][:, 1:])        # different laws
    assert d1["meta"]["coef_spread"] == 0.2 and np.allclose(d1["coef"], node_coefficients(classes, 0.2, seed=1))
    assert "coef" not in d0 and "coef_spread" not in d0["meta"]         # default output unchanged


def test_rhs_uses_node_coefficients():
    A = np.zeros((1, 1), np.float32)
    classes = np.array([2])
    d = simulate_a1(A, classes, kappa=0.5, samples=1, T=0.002, dt=0.001, seed=0, x0=np.array([1.5]), coef_spread=0.2)
    a, b = d["coef"][0]
    rate = (d["data"][0, 1, 0, 0] - d["data"][0, 0, 0, 0]) / 0.001
    assert abs(rate - (-a * 1.5 + b * np.tanh(1.5))) < 0.01
