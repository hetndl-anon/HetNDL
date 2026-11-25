import json

import numpy as np

from hetndl.baselines.arni import arni, pair_basis, self_basis
from hetndl.gen.a2_mixed import sample_classes2, simulate_a2
from hetndl.gen.graphs import make_graph
from hetndl.gen.kuramoto import save_dataset, simulate_kuramoto
from hetndl.metrics import auroc_auprc, sign_accuracy


def _kura(neg):
    A = make_graph(12, "ER", seed=3, p=0.25, directed=True)
    return simulate_kuramoto(A, 2.0, 0.3, neg, False, 6, 5.0, 0.01, seed=3)


def test_basis_shapes():
    x = np.random.default_rng(0).normal(size=(50, 2))
    assert self_basis(x, "poly").shape == (50, 10)          # 1 + 2 + 3 + 4 monomials of (u, v) up to degree 3
    assert pair_basis(x, x[::-1], "poly").shape == (50, 8)   # per coordinate: x_j, x_j^2, x_j^3, x_i x_j
    t = x[:, :1]
    assert self_basis(t, "fourier").shape == (50, 3)
    pb = pair_basis(t, t + 0.3, "fourier")
    assert pb.shape == (50, 4) and np.allclose(pb[:, 0], np.sin(0.3))  # first column is the sign-carrying term


def test_arni_full_recovers_signed_kuramoto():
    # joint block least squares over all sources, ranked by the standardised block coefficient norm
    d = _kura(0.4)
    W, info = arni(d["data"], d["meta"]["dt"], basis="fourier", mode="full")
    auroc, _ = auroc_auprc(W, d["A"])
    assert auroc > 0.95
    assert sign_accuracy(W, d["S"]) > 0.9
    assert W.shape == d["A"].shape and np.all(np.diag(W) == 0)
    assert info["self_coef"].shape == (12, 3)


def test_selected_edges_outrank_unselected():
    d = _kura(0.0)
    W, info = arni(d["data"], d["meta"]["dt"], basis="fourier", k_max=2, mode="greedy")
    for i in range(12):
        sel = info["selected"][i]
        assert len(sel) == 2
        rest = [j for j in range(12) if j != i and j not in sel]
        assert np.abs(W[sel, i]).min() > np.abs(W[rest, i]).max()


def test_arni_cli_on_a2(tmp_path):
    from hetndl.baselines.arni import main
    A = make_graph(16, "ER", seed=1, p=0.15, directed=True)
    c = sample_classes2(16, 1)
    path = str(tmp_path / "a2.npz")
    save_dataset(path, simulate_a2(A, c, 0.3, 12, 10, 0.05, 1))
    out = str(tmp_path / "arni")
    main(["--data", path, "--out", out, "--k_max", "5", "--cluster_k", "2"])
    m = json.load(open(out + "/metrics.json"))
    assert m["basis"] == "poly" and m["mode"] == "full" and m["auroc"] > 0.9 and 0.0 <= m["cluster_ari"] <= 1.0
    assert np.load(out + "/W.npy").shape == (16, 16)


def test_full_self_constant_is_in_original_scale():
    # Kuramoto: the constant self term is omega_i, so it must survive the column standardisation
    d = _kura(0.0)
    _, info = arni(d["data"], d["meta"]["dt"], basis="fourier", mode="full")
    assert np.abs(info["self_coef"][:, 0] - d["omega"]).max() < 0.1


def test_arni_cli_n_traj_and_noise(tmp_path):
    from hetndl.baselines.arni import main
    d = _kura(0.0)
    path = str(tmp_path / "k.npz"); save_dataset(path, d)
    out = str(tmp_path / "arni")
    main(["--data", path, "--out", out, "--mode", "full", "--n_traj", "3", "--noise_std", "0.05"])
    m = json.load(open(out + "/metrics.json"))
    assert m["n_traj"] == 3 and m["noise_std"] == 0.05 and 0.0 <= m["auroc"] <= 1.0


def test_ridge_shrinks_and_auto_selects():
    from hetndl.baselines.arni import arni, _full_fit, self_basis, pair_basis
    d = _kura(0.0)
    # ridge shrinks the standardised coefficient vector (its objective), not necessarily the block scores
    x = d["data"][0, 1:-1, :, 0]; y = (d["data"][0, 2:, 0, 0] - d["data"][0, :-2, 0, 0]) / 0.02
    S = self_basis(x[:, :1], "fourier"); blocks = [pair_basis(x[:, :1], x[:, j:j + 1], "fourier") for j in range(1, 12)]
    b0, _, sd0 = _full_fit(S, blocks, y[:, None], 0.0)
    b1, _, sd1 = _full_fit(S, blocks, y[:, None], 10.0)
    assert np.linalg.norm((b1 * sd1[:, None])[1:]) < np.linalg.norm((b0 * sd0[:, None])[1:])
    Wa, ia = arni(d["data"], d["meta"]["dt"], basis="fourier", mode="full", ridge="auto")
    assert ia["ridge"] in ia["ridge_grid"] and auroc_auprc(Wa, d["A"])[0] > 0.95


def test_smoothing_reduces_derivative_noise():
    from hetndl.baselines.arni import derivatives
    t = np.arange(0, 5, 0.01)
    clean = np.sin(t)[None, :, None, None]                            # (S=1, T, N=1, d=1)
    noisy = clean + np.random.default_rng(0).normal(0, 0.05, clean.shape)
    X0, Y0 = derivatives(noisy, 0.01, smooth=0)
    X1, Y1 = derivatives(noisy, 0.01, smooth=15)
    true = np.cos(t)
    e0 = np.abs(Y0[0, :, 0, 0] - true[1:-1]).mean()
    e1 = np.abs(Y1[0, :, 0, 0] - true[7:-7]).mean()
    assert e1 < e0 / 3 and X1.shape[1] == Y1.shape[1] == len(t) - 14


def test_arni_jacobian_sign_on_multiplicative_coupling():
    # GLV: the interaction enters through x_i x_j, so the sign of the x_j column says nothing; the Jacobian sign
    # (row-mean of d pair-term / d x_j) recovers it. On Kuramoto both rules agree.
    from hetndl.gen.glv import simulate_glv
    A = make_graph(10, "ER", seed=2, p=0.3, directed=True)
    d = simulate_glv(A, c=0.5, neg_frac=0.5, samples=30, T=3.0, dt=0.03, seed=2)
    W, info = arni(d["data"], d["meta"]["dt"], basis="poly", mode="full", ridge=1e-3)
    Wj = info["W_jac"]
    assert Wj.shape == W.shape and np.allclose(np.abs(Wj), np.abs(W))
    assert sign_accuracy(Wj, d["S"]) > 0.9
    k = _kura(0.4)
    W2, info2 = arni(k["data"], k["meta"]["dt"], basis="fourier", mode="full")
    assert sign_accuracy(info2["W_jac"], k["S"]) > 0.9 and sign_accuracy(W2, k["S"]) > 0.9
