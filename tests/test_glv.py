import numpy as np

from hetndl.gen.glv import simulate_glv
from hetndl.gen.graphs import make_graph
from hetndl.gen.kuramoto import save_dataset, load_dataset


def test_glv_shapes_positive_bounded_and_signed(tmp_path):
    A = make_graph(30, "ER", seed=1, p=0.1, directed=True)
    d = simulate_glv(A, c=0.5, neg_frac=0.5, samples=6, T=5.0, dt=0.05, seed=1)
    x = d["data"]
    assert x.shape == (6, 100, 30, 1) and np.isfinite(x).all()
    assert x.min() > 0 and x.max() < 5                      # positive, bounded (diagonally dominant interactions)
    S, A2 = d["S"], d["A"]
    assert ((S != 0) == (A2 > 0)).all() and set(np.unique(S[S != 0]).tolist()) == {-1.0, 1.0}
    assert abs((S < 0).sum() / (S != 0).sum() - 0.5) < 0.05
    assert d["omega"].shape == (30,) and 0.5 <= d["omega"].min() and d["omega"].max() <= 1.5   # r_i
    assert (d["classes"] == -1).all() and d["meta"]["system"] == "glv"
    k = np.maximum(A2.sum(0), 1)
    assert np.allclose(d["W_true"], 0.5 * S / k[None, :])   # W_true[j, i] = c s_ji / k_i
    out = str(tmp_path / "g.npz"); save_dataset(out, d)
    assert np.allclose(load_dataset(out)["W_true"], d["W_true"])


def test_glv_rate_two_species():
    # 0 -> 1 inhibitory edge; node 0 has no input: dx0 = x0 (r0 - x0); dx1 = x1 (r1 - x1 - c x0)
    A = np.zeros((2, 2), np.float32); A[0, 1] = 1.0
    x0 = np.array([0.8, 0.6])
    d = simulate_glv(A, c=0.5, neg_frac=1.0, samples=1, T=0.002, dt=0.001, seed=0, x0=x0)
    r = d["omega"].astype(np.float64)
    rate = (d["data"][0, 1, :, 0] - d["data"][0, 0, :, 0]) / 0.001
    expect = np.array([0.8 * (r[0] - 0.8), 0.6 * (r[1] - 0.6 - 0.5 * 0.8)])
    assert np.allclose(rate, expect, atol=0.01)


def test_glv_cli_default_graph_is_the_directed_er_graph_of_the_benchmark(tmp_path):
    from hetndl.gen.glv import main
    out = str(tmp_path / "er.npz")
    main(["--N", "12", "--samples", "2", "--T", "0.5", "--seed", "4", "--out", out])
    assert np.array_equal(load_dataset(out)["A"], make_graph(12, "ER", seed=4, p=0.1, directed=True))


def test_glv_cli_ba_graph(tmp_path):
    # the same GLV-+- dynamics on a Barabasi-Albert graph (m = 2, undirected as in the other BA sets)
    from hetndl.gen.glv import main
    out = str(tmp_path / "ba.npz")
    main(["--N", "30", "--graph", "BA", "--m", "2", "--samples", "2", "--T", "1.0", "--seed", "2", "--out", out])
    d = load_dataset(out)
    assert np.array_equal(d["A"], make_graph(30, "BA", seed=2, m=2))
    assert np.array_equal(d["A"], d["A"].T) and d["A"].sum() == 2 * 2 * (30 - 2)
    assert d["meta"]["graph"] == "BA" and d["meta"]["m"] == 2
