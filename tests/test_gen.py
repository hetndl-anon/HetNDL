import numpy as np

from hetndl.gen.graphs import make_graph
from hetndl.gen.kuramoto import simulate_kuramoto, save_dataset, load_dataset


def test_make_graph_shapes_and_convention():
    A = make_graph(10, "ER", seed=0)
    assert A.shape == (10, 10) and A.dtype == np.float32
    assert np.allclose(np.diag(A), 0)
    assert np.allclose(A, A.T)  # undirected by default
    B = make_graph(20, "ER", seed=0, p=0.3, directed=True)
    assert np.allclose(np.diag(B), 0) and not np.allclose(B, B.T)
    C = make_graph(50, "BA", seed=0)
    assert C.sum() == 2 * 49  # m = 1 tree, symmetric


def test_kuramoto_direction_two_nodes():
    A = np.zeros((2, 2), np.float32)
    A[0, 1] = 1.0  # 0 -> 1
    d = simulate_kuramoto(A, K=1.0, sigma_omega=0.0, neg_frac=0.0, clip=False,
                          samples=1, T=0.02, dt=0.01, seed=0, theta0=np.array([0.0, 1.0]))
    rate = (d["data"][0, 1, :, 0] - d["data"][0, 0, :, 0]) / 0.01
    assert abs(rate[0] - 1.0) < 1e-2                        # source unaffected
    assert abs(rate[1] - (1.0 + np.sin(0.0 - 1.0))) < 2e-2  # target coupled


def test_kuramoto_unsigned_degree_normalisation():
    A = np.zeros((3, 3), np.float32)
    A[0, 2] = 1.0
    A[1, 2] = 1.0
    S = A.copy()
    S[1, 2] = -1.0  # signed sum of in-weights is 0; unsigned degree is 2
    d = simulate_kuramoto(A, K=2.0, sigma_omega=0.0, neg_frac=0.0, clip=False, samples=1,
                          T=0.02, dt=0.01, seed=0, theta0=np.array([0.5, 0.0, 1.0]), S=S)
    rate = (d["data"][0, 1, :, 0] - d["data"][0, 0, :, 0]) / 0.01
    expected = 1.0 + (2.0 / 2.0) * (np.sin(0.5 - 1.0) - np.sin(0.0 - 1.0))
    assert np.isfinite(rate).all()
    assert abs(rate[2] - expected) < 2e-2


def test_kuramoto_neg_frac():
    A = make_graph(60, "ER", seed=3, p=0.2)
    d = simulate_kuramoto(A, 2.0, 0.3, 0.5, False, 1, 0.1, 0.01, 0)
    S = d["S"]
    E = A > 0
    assert np.allclose(np.abs(S), A) and np.allclose(S, S.T)
    assert 0.4 <= (S[E] < 0).mean() <= 0.6


def test_kuramoto_sigma_zero_and_clip():
    A = make_graph(10, "ER", seed=0)
    d = simulate_kuramoto(A, 2.0, 0.0, 0.0, False, 2, 0.1, 0.01, 0)
    assert np.allclose(d["omega"], 1.0) and d["data"].shape == (2, 10, 10, 1)
    e = simulate_kuramoto(A, 2.0, 5.0, 0.0, True, 1, 0.05, 0.01, 0)
    assert e["omega"].min() >= 0.1 and e["omega"].max() <= 2.0
    f = simulate_kuramoto(A, 2.0, 5.0, 0.0, False, 1, 0.05, 0.01, 0)
    assert f["omega"].min() < 0.1 or f["omega"].max() > 2.0


def test_dataset_roundtrip(tmp_path):
    A = make_graph(10, "BA", seed=0)
    d = simulate_kuramoto(A, 2.0, 0.3, 0.2, True, 1, 0.1, 0.01, 0)
    p = str(tmp_path / "x.npz")
    save_dataset(p, d)
    e = load_dataset(p)
    assert np.allclose(e["data"], d["data"]) and np.allclose(e["omega"], d["omega"])
    assert e["meta"]["K"] == 2.0 and e["meta"]["system"] == "kuramoto" and e["meta"]["dt"] == 0.01
    assert "r_tail" in e["meta"]


def test_kuramoto_omega_independent_of_neg_frac():
    A = make_graph(30, "ER", seed=2, p=0.2)
    d0 = simulate_kuramoto(A, 2.0, 0.3, 0.0, False, 1, 0.05, 0.01, 7)
    d1 = simulate_kuramoto(A, 2.0, 0.3, 0.5, False, 1, 0.05, 0.01, 7)
    assert np.allclose(d0["omega"], d1["omega"])


def test_kuramoto_cli_defaults_reproduce_the_benchmark_generator(tmp_path):
    from hetndl.gen.kuramoto import main, load_dataset, simulate_kuramoto
    from hetndl.gen.graphs import make_graph
    out = str(tmp_path / "k.npz")
    main(["--N", "12", "--samples", "2", "--T", "0.5", "--seed", "3", "--out", out])
    ref = simulate_kuramoto(make_graph(12, "ER", seed=3), 2.0, 0.3, 0.0, False, 2, 0.5, 0.01, 3)
    d = load_dataset(out)
    assert np.array_equal(d["data"], ref["data"]) and np.array_equal(d["omega"], ref["omega"])


def test_kuramoto_nri_setting_uniform_omega_and_unnormalised_coupling(tmp_path):
    # the 5-oscillator Kuramoto of the NRI paper (omega ~ U[1, 10), k_ij = 1 without dividing by the
    # in-degree, undirected edges with probability 0.5)
    from hetndl.gen.kuramoto import main, load_dataset
    out = str(tmp_path / "k5.npz")
    main(["--N", "5", "--p", "0.5", "--K", "1", "--omega_uniform", "1", "10", "--coupling_norm", "none",
          "--samples", "3", "--T", "0.02", "--dt", "0.01", "--seed", "1", "--out", out])
    d = load_dataset(out); th = d["data"][:, :, :, 0].astype(np.float64); w = d["omega"].astype(np.float64)
    assert ((w >= 1) & (w < 10)).all() and d["meta"]["coupling_norm"] == "none" and d["meta"]["omega_uniform"] == [1.0, 10.0]
    S = d["S"].astype(np.float64)
    for s in range(3):
        x = th[s, 0]; rhs = w + (S * np.sin(x[None, :] - x[:, None]).T).sum(0)   # sum_j S[j, i] sin(x_j - x_i)
        assert np.allclose((th[s, 1] - th[s, 0]) / 0.01, rhs, atol=0.1)


def test_kuramoto_nri_regime_draws_one_graph_and_omega_per_trajectory(tmp_path):
    # the NRI paper's regime (Kipf et al. 2018, App. B.3): every simulation has its own random undirected graph
    # (edge probability 0.5, k_ij = 1), its own omega ~ U[1, 10) and initial phases
    from hetndl.gen.kuramoto import simulate_kuramoto_nri_regime
    d = simulate_kuramoto_nri_regime(n_sims=4, N=5, p=0.5, T=0.003, dt=0.001, seed=3)
    assert d["data"].shape == (4, 3, 5, 1) and d["A_all"].shape == (4, 5, 5) and d["omega_all"].shape == (4, 5)
    for s in range(4):
        A = d["A_all"][s]
        assert np.array_equal(A, A.T) and not np.diag(A).any()
    assert not all(np.array_equal(d["A_all"][0], d["A_all"][s]) for s in range(1, 4))
    w = d["omega_all"]; assert (w >= 1).all() and (w < 10).all()
    th = d["data"][0, :, :, 0].astype(np.float64); A = d["A_all"][0]
    rhs = w[0] + (A * np.sin(th[0][None, :] - th[0][:, None]).T).sum(0)
    assert np.allclose((th[1] - th[0]) / 0.001, rhs, atol=0.05)
    assert d["meta"]["regime"] == "nri"


def test_subsample_keeps_every_kth_state_and_scales_dt(tmp_path):
    from hetndl.gen.kuramoto import main, load_dataset
    from hetndl.gen.subsample import main as sub
    src, out = str(tmp_path / "k.npz"), str(tmp_path / "k_sub4.npz")
    main(["--N", "6", "--samples", "2", "--T", "0.2", "--dt", "0.01", "--seed", "2", "--out", src])
    sub(["--src", src, "--k", "4", "--out", out])
    a, b = load_dataset(src), load_dataset(out)
    assert np.array_equal(b["data"], a["data"][:, ::4]) and b["data"].flags["C_CONTIGUOUS"]
    for key in ("A", "S", "omega", "classes"):
        assert np.array_equal(a[key], b[key])
    assert b["meta"]["dt"] == 0.04 and b["meta"]["subsample"] == 4 and b["meta"]["source"] == "k"
    assert {k: v for k, v in b["meta"].items() if k not in ("dt", "subsample", "source")} == \
           {k: v for k, v in a["meta"].items() if k != "dt"}


def test_springs_graph_has_no_isolated_node_and_energy_is_conserved(tmp_path):
    # exact solution of dr/dt = v, dv/dt = -k L r: the energy |v|^2 / 2 + k r^T L r / 2 is constant
    from hetndl.gen.springs import graph, main
    from hetndl.gen.kuramoto import load_dataset
    for seed in range(1, 6):
        A, r = graph(seed, 10, 0.2)
        assert np.array_equal(A, A.T) and not np.diag(A).any() and A.sum(0).min() > 0
    out = str(tmp_path / "s.npz")
    main(["--N", "6", "--p", "0.4", "--samples", "3", "--steps", "20", "--dt", "0.1", "--seed", "2", "--out", out])
    d = load_dataset(out); X = d["data"]; A = d["A"]; L = np.diag(A.sum(0)) - A
    assert X.shape == (3, 20, 6, 4) and d["meta"]["system"] == "spring" and np.array_equal(d["S"], A)
    assert np.allclose(np.linalg.norm(X[:, 0, :, 2:4], axis=-1), 0.5)                # |v_i(0)| = 0.5 at every node
    r, v = X[..., 0:2], X[..., 2:4]
    E = 0.5 * (v ** 2).sum((-1, -2)) + 0.5 * 0.1 * np.einsum("stid,ij,stjd->st", r, L, r)
    assert np.allclose(E, E[:, :1], rtol=1e-8)
