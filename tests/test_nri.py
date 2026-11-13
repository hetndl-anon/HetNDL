import json

import numpy as np
import torch

from hetndl.baselines.nri import NRIEncoder, NRIDecoder, nri, windows


def _linear_system(N=8, S=24, T=60, dt=0.1, seed=0):
    # x_{t+1} = x_t + dt * (A^T (x - x_i) + drive): a plain diffusive linear network, directed ER graph
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) < 0.3).astype(float); np.fill_diagonal(A, 0)
    drive = rng.normal(0, 1, N)
    data = np.zeros((S, T, N, 1), np.float32)
    for s in range(S):
        x = rng.normal(0, 1, N)
        for t in range(T):
            data[s, t, :, 0] = x
            coup = np.array([sum(A[j, i] * (x[j] - x[i]) for j in range(N)) for i in range(N)])
            x = x + dt * (coup + 0.3 * drive * np.sin(0.5 * t * dt) - 0.1 * x)
    return data, A


def test_encoder_decoder_shapes():
    N, T, d, K, B = 5, 12, 2, 2, 3
    enc = NRIEncoder(T * d, hidden=16, n_types=K, N=N)
    dec = NRIDecoder(d, hidden=16, n_types=K, N=N)
    x = torch.randn(B, N, T, d)
    logits = enc(x.reshape(B, N, T * d))
    assert logits.shape == (B, N * (N - 1), K)
    z = torch.softmax(logits, -1)
    pred = dec(x, z, pred_steps=4)
    assert pred.shape == (B, N, T - 1, d) and torch.isfinite(pred).all()
    P = enc.edge_matrix(z[..., 1])
    assert P.shape == (B, N, N) and torch.all(torch.diagonal(P, dim1=1, dim2=2) == 0)


def test_windows_cover_trajectories():
    data = np.arange(2 * 10 * 3 * 1, dtype=np.float32).reshape(2, 10, 3, 1)
    w = windows(data, np.array([0, 1]), win=4)
    assert w.shape == (4, 3, 4, 1)                     # 2 trajectories x 2 full windows of 4 steps (stride = win)
    assert np.array_equal(w[0, :, :, 0], data[0, 0:4, :, 0].T)


def test_nri_recovers_linear_network(tmp_path):
    data, A = _linear_system()
    W, info = nri(data, seed=0, epochs=40, hidden=32, win=20, pred_steps=5, lr=5e-3, device="cpu")
    from hetndl.metrics import auroc_auprc
    au, ap = auroc_auprc(W, A)
    assert W.shape == A.shape and np.all(np.diag(W) == 0) and 0 <= W.min() and W.max() <= 1
    assert au > 0.8, au
    assert info["best_epoch"] >= 1 and np.isfinite(info["val_nll_best"])


def test_phase_features_raw_is_identity_and_trig_gives_cos_and_sin():
    from hetndl.data import phase_features
    x = np.random.default_rng(0).uniform(-20, 20, (3, 7, 4, 1)).astype(np.float32)
    assert phase_features(x, "raw") is x
    t = phase_features(x, "trig")
    assert t.shape == (3, 7, 4, 2) and np.allclose(t[..., 0], np.cos(x[..., 0])) and np.allclose(t[..., 1], np.sin(x[..., 0]))


def test_nri_trig_features_double_the_state_dimension():
    # phases may enter NRI as (cos, sin), the rule HetNDL, NOGI and MTGNN use for phases
    data, A = _linear_system(N=5, S=6, T=30)
    W, info = nri(data, seed=0, epochs=1, hidden=8, win=10, pred_steps=5, device="cpu", feat="trig")
    assert W.shape == (5, 5) and info["state_dim"] == 2 and info["feat"] == "trig"
    W, info = nri(data, seed=0, epochs=1, hidden=8, win=10, pred_steps=5, device="cpu")
    assert info["state_dim"] == 1 and info["feat"] == "raw"


def test_phase_features_nri_is_the_increment_and_the_sine():
    # the Kuramoto inputs of the NRI paper without omega (d phi as a forward difference, sin phi)
    from hetndl.data import phase_features
    x = np.cumsum(np.random.default_rng(1).normal(0, 0.3, (2, 6, 3, 1)), axis=1).astype(np.float32)
    f = phase_features(x, "nri")
    assert f.shape == (2, 5, 3, 2)
    assert np.allclose(f[..., 0], np.diff(x[..., 0], axis=1)) and np.allclose(f[..., 1], np.sin(x[:, :-1, :, 0]))


def test_gaussian_nll_sum_time_is_the_mean_times_the_horizon():
    from hetndl.baselines.nri import gaussian_nll
    pred, tgt = torch.randn(4, 3, 9, 2), torch.randn(4, 3, 9, 2)
    m = gaussian_nll(pred, tgt, 0.5, 3, "mean"); s = gaussian_nll(pred, tgt, 0.5, 3, "sum_time")
    assert torch.allclose(s, m * 9)


def test_minmax_scales_the_training_trajectories_to_unit_range():
    from hetndl.baselines.nri import minmax
    x = np.random.default_rng(2).normal(3, 5, (5, 7, 2, 2)).astype(np.float32)
    y = minmax(x, np.array([0, 2, 4]))
    assert np.allclose(y[[0, 2, 4]].reshape(-1, 2).min(0), -1) and np.allclose(y[[0, 2, 4]].reshape(-1, 2).max(0), 1)


def test_nri_cli_nri_omega_appends_the_node_frequency(tmp_path):
    from hetndl.baselines.nri import main
    from hetndl.gen.kuramoto import main as gen
    p = str(tmp_path / "k.npz"); gen(["--N", "4", "--p", "0.5", "--samples", "6", "--T", "1.0", "--dt", "0.1", "--out", p])
    out = str(tmp_path / "o")
    main(["--data", p, "--out", out, "--epochs", "1", "--win", "8", "--pred_steps", "4", "--hidden", "8", "--device", "cpu",
          "--feat", "nri_omega", "--loss_norm", "sum_time", "--input_norm", "minmax", "--var", "5e-5"])
    m = json.load(open(out + "/metrics.json"))
    assert m["state_dim"] == 3 and m["args"]["loss_norm"] == "sum_time" and m["args"]["input_norm"] == "minmax"


def test_nri_cli_scores_per_trajectory_graphs_in_the_nri_regime(tmp_path):
    # one graph per trajectory (the NRI-paper regime) -> the posterior of each test trajectory is scored against its own graph
    from hetndl.baselines.nri import main
    from hetndl.gen.kuramoto import simulate_kuramoto_nri_regime, save_dataset
    p = str(tmp_path / "r.npz"); save_dataset(p, simulate_kuramoto_nri_regime(n_sims=20, N=4, T=1.0, dt=0.1, seed=1))
    out = str(tmp_path / "o")
    main(["--data", p, "--out", out, "--epochs", "1", "--win", "50", "--pred_steps", "4", "--hidden", "8", "--device", "cpu",
          "--feat", "nri_omega", "--loss_norm", "sum_time", "--input_norm", "minmax", "--var", "5e-5", "--batch", "4"])
    m = json.load(open(out + "/metrics.json"))
    assert m["per_sample_graph"] is True and m["n_test"] == 4 and 0 <= m["acc"] <= 1 and 0 <= m["auroc"] <= 1
    assert m["state_dim"] == 3
