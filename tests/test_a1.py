import numpy as np

from hetndl.gen.a1_mixed import simulate_a1, sample_classes
from hetndl.gen.graphs import make_graph


def test_a1_classes_and_bounded():
    A = make_graph(30, "ER", seed=1, p=0.1, directed=True)
    classes = sample_classes(30, seed=1)
    d = simulate_a1(A, classes, kappa=0.5, samples=18, T=10.0, dt=0.05, seed=1)
    assert d["data"].shape == (18, 200, 30, 1)
    assert set(np.unique(d["classes"]).tolist()) == {0, 1, 2}
    assert np.isfinite(d["data"]).all() and np.abs(d["data"]).max() < 10
    assert d["meta"]["system"] == "a1" and d["meta"]["dt"] == 0.05
    assert np.isnan(d["omega"]).all()  # no scalar node parameter in A1


def test_a1_mechanisms_two_nodes():
    A = np.zeros((2, 2), np.float32)
    A[0, 1] = 1.0  # 0 -> 1
    classes = np.array([1, 0])  # node 0 cubic, node 1 linear
    d = simulate_a1(A, classes, kappa=0.5, samples=1, T=0.002, dt=0.001, seed=0,
                    x0=np.array([2.0, 1.0]))
    rate = (d["data"][0, 1, :, 0] - d["data"][0, 0, :, 0]) / 0.001
    expected0 = -2.0 - 0.5 * 8.0                                   # f_1(2), no incoming edge
    expected1 = -1.0 + 0.5 * (np.tanh(2.0) - np.tanh(1.0))         # f_0(1) + coupling from node 0
    assert abs(rate[0] - expected0) < 0.05
    assert abs(rate[1] - expected1) < 0.05


def test_sample_classes_covers_all():
    for seed in range(5):
        c = sample_classes(6, seed=seed)
        assert c.shape == (6,) and set(np.unique(c).tolist()) == {0, 1, 2}


def test_a1_cli_ba_graph(tmp_path):
    # scale-free variant for the graph-type robustness check: BA(m) is undirected -> symmetric A
    import numpy as np
    from hetndl.gen.a1_mixed import main
    from hetndl.gen.kuramoto import load_dataset
    out = str(tmp_path / "a1_ba.npz")
    main(["--N", "20", "--graph", "BA", "--m", "2", "--samples", "3", "--T", "1", "--dt", "0.05", "--seed", "4", "--out", out])
    d = load_dataset(out)
    A = d["A"]
    assert (A == A.T).all() and A.sum() == 2 * 2 * (20 - 2) and np.diag(A).sum() == 0
    assert d["meta"]["graph"] == "BA" and d["meta"]["m"] == 2
