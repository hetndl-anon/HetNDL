import numpy as np

from hetndl.gen.a3_mixed import simulate_a3, edge_coupling, edge_types_from_classes
from hetndl.gen.a2_mixed import sample_classes2, self_field
from hetndl.gen.graphs import make_graph
from hetndl.gen.kuramoto import save_dataset, load_dataset


def test_edge_types_follow_source_class():
    A = np.zeros((3, 3), np.float32)
    A[0, 1] = A[1, 2] = A[2, 0] = 1.0
    E = edge_types_from_classes(A, np.array([0, 1, 1]))
    assert E[0, 1] == 0 and E[1, 2] == 1 and E[2, 0] == 1
    assert E[0, 0] == -1 and E[1, 0] == -1            # non-edges carry -1


def test_edge_coupling_shapes():
    du = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    lin, sat = edge_coupling(du, 0), edge_coupling(du, 1)
    assert np.allclose(lin, du)
    assert np.allclose(sat, np.tanh(2 * du)) and sat[2] == 0


def test_a3_rate_matches_mixed_couplings():
    A = np.zeros((3, 3), np.float32)
    A[0, 2] = A[1, 2] = 1.0                           # 0 -> 2 (source FHN), 1 -> 2 (source VdP)
    classes = np.array([0, 1, 0])
    x0 = np.array([[2.0, 0.0], [-1.0, 0.0], [0.5, 0.2]])
    d = simulate_a3(A, classes, kappa=0.3, samples=1, T=0.002, dt=0.001, seed=0, x0=x0)
    rate = (d["data"][0, 1] - d["data"][0, 0]) / 0.001
    f = self_field(x0[:, 0], x0[:, 1], classes)
    expect_u = f[2, 0] + 0.3 * ((2.0 - 0.5) + np.tanh(2 * (-1.0 - 0.5)))
    assert np.allclose(rate[2], [expect_u, f[2, 1]], atol=0.05)
    assert d["meta"]["system"] == "a3" and d["etypes"].shape == (3, 3)


def test_a3_bounded_and_roundtrip(tmp_path):
    A = make_graph(30, "ER", seed=1, p=0.1, directed=True)
    c = sample_classes2(30, 1)
    d = simulate_a3(A, c, kappa=0.3, samples=4, T=10.0, dt=0.05, seed=1)
    assert d["data"].shape == (4, 200, 30, 2) and np.abs(d["data"]).max() < 10
    p = str(tmp_path / "a3.npz")
    save_dataset(p, d)
    back = load_dataset(p)
    assert np.array_equal(back["etypes"], d["etypes"]) and back["meta"]["system"] == "a3"
