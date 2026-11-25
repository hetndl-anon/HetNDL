import numpy as np

from hetndl.gen.a2_mixed import simulate_a2, self_field, sample_classes2
from hetndl.gen.graphs import make_graph


def test_a2_classes_shape_and_bounded():
    A = make_graph(30, "ER", seed=1, p=0.1, directed=True)
    classes = sample_classes2(30, seed=1)
    d = simulate_a2(A, classes, kappa=0.3, samples=6, T=10.0, dt=0.05, seed=1)
    assert d["data"].shape == (6, 200, 30, 2)
    assert set(np.unique(d["classes"]).tolist()) == {0, 1}
    assert np.isfinite(d["data"]).all() and np.abs(d["data"]).max() < 10
    assert d["meta"]["system"] == "a2" and d["meta"]["dt"] == 0.05
    assert np.isnan(d["omega"]).all()


def test_a2_self_fields_are_fhn_and_vdp():
    u, v = np.array([1.5, -0.5]), np.array([0.3, 2.0])
    f = self_field(u, v, np.array([0, 1]))          # node 0 FHN, node 1 Van der Pol
    fhn = (1.5 - 1.5 ** 3 / 3 - 0.3 + 0.5, 0.08 * (1.5 + 0.7 - 0.8 * 0.3))
    vdp = (2.0, 1.0 * (1 - 0.25) * 2.0 + 0.5)
    assert np.allclose(f[0], fhn) and np.allclose(f[1], vdp)


def test_a2_coupling_is_diffusive_on_u_only():
    A = np.zeros((2, 2), np.float32)
    A[0, 1] = 1.0  # 0 -> 1
    classes = np.array([1, 0])
    x0 = np.array([[2.0, 0.5], [1.0, -1.0]])
    d = simulate_a2(A, classes, kappa=0.3, samples=1, T=0.002, dt=0.001, seed=0, x0=x0)
    rate = (d["data"][0, 1] - d["data"][0, 0]) / 0.001       # (N, 2)
    f = self_field(x0[:, 0], x0[:, 1], classes)
    assert np.allclose(rate[0], f[0], atol=0.05)                                  # no incoming edge
    assert np.allclose(rate[1], f[1] + np.array([0.3 * (2.0 - 1.0), 0.0]), atol=0.05)


def test_sample_classes2_balanced_and_covering():
    for seed in range(5):
        c = sample_classes2(30, seed=seed)
        assert set(np.unique(c).tolist()) == {0, 1} and 10 <= (c == 1).sum() <= 20
