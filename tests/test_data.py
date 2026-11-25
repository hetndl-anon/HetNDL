import numpy as np
import torch

from hetndl.data import add_noise_snr, trajectory_split, make_pairs, build_loaders


def test_trajectory_split_disjoint_cover():
    tr, va, te = trajectory_split(20, seed=0)
    s = set(tr) | set(va) | set(te)
    assert s == set(range(20))
    assert not (set(tr) & set(va)) and not (set(va) & set(te)) and not (set(tr) & set(te))
    assert len(va) >= 1 and len(te) >= 1
    tr2, _, _ = trajectory_split(20, seed=0)
    assert np.array_equal(tr, tr2)


def test_trajectory_split_small():
    tr, va, te = trajectory_split(3, seed=1)
    assert len(tr) == 1 and len(va) == 1 and len(te) == 1


def test_pairs_shift():
    data = np.arange(3 * 5 * 2 * 1, dtype=np.float32).reshape(3, 5, 2, 1)
    x, y = make_pairs(data, np.array([0, 2]))
    assert x.shape == (8, 2, 1) and y.shape == (8, 2, 1)
    assert torch.allclose(x[1], y[0])
    assert torch.allclose(x[4], torch.tensor(data[2, 0]))


def test_noise_hits_input_and_target():
    data = np.ones((2, 6, 3, 1), np.float32)
    rng = np.random.default_rng(0)
    noisy = add_noise_snr(data, 20.0, rng)
    assert noisy.shape == data.shape and not np.allclose(noisy, data)
    assert np.allclose(add_noise_snr(data, None, rng), data)
    x, y = make_pairs(noisy, np.array([0]))
    assert not torch.allclose(y, torch.ones_like(y))


def test_build_loaders():
    data = np.random.default_rng(0).standard_normal((10, 7, 4, 1)).astype(np.float32)
    tr, va, te, te_idx = build_loaders(data, seed=0, batch=8, snr_db=None, num_workers=0)
    xb, yb = next(iter(tr))
    assert xb.shape[1:] == (4, 1) and yb.shape == xb.shape
    assert len(te_idx) == 2
    assert len(tr.dataset) == 6 * 6 and len(va.dataset) == 2 * 6


def test_build_loaders_pairs_split():
    data = np.random.default_rng(0).standard_normal((10, 7, 4, 1)).astype(np.float32)
    tr, va, te, te_idx = build_loaders(data, seed=0, batch=8, snr_db=None, num_workers=0, split="pairs")
    total = 10 * 6
    assert len(tr.dataset) == int(0.6 * total) and len(va.dataset) == int(0.2 * total)
    assert len(te.dataset) == total - int(0.6 * total) - int(0.2 * total)
    assert len(te_idx) == 10


def test_noise_std_adds_absolute_gaussian_noise():
    import numpy as np
    from hetndl.data import add_noise
    data = np.zeros((2, 50, 3, 1), np.float32)
    rng = np.random.default_rng(0)
    out = add_noise(data, rng, snr_db=None, noise_std=0.1)
    assert out.shape == data.shape and abs(out.std() - 0.1) < 0.02
    assert np.array_equal(add_noise(data, rng, snr_db=None, noise_std=None), data)


def test_n_traj_limits_the_trajectories_before_the_split():
    import numpy as np
    from hetndl.data import build_loaders, select_trajectories
    data = np.random.default_rng(0).normal(size=(20, 6, 4, 1)).astype(np.float32)
    keep = select_trajectories(20, 8, seed=1)
    assert len(keep) == 8 and len(set(keep.tolist())) == 8
    assert np.array_equal(keep, select_trajectories(20, 8, seed=1))            # deterministic
    tr, va, te, te_idx = build_loaders(data, 1, 16, None, n_traj=8)
    n_pairs = len(tr.dataset) + len(va.dataset) + len(te.dataset)
    assert n_pairs == 8 * 5                                                   # 8 trajectories x (T - 1) pairs
    assert set(te_idx.tolist()) <= set(keep.tolist())                         # test ids index the ORIGINAL array
    assert select_trajectories(20, None, seed=1).tolist() == list(range(20))


def test_smooth_series_denoises_along_time_only():
    from hetndl.data import smooth_series
    t = np.linspace(0, 4 * np.pi, 400, dtype=np.float32)
    clean = np.stack([np.sin(t), np.cos(0.5 * t)], -1)[None, :, :, None]        # (1, T, 2 nodes, 1)
    noisy = clean + np.random.default_rng(0).normal(0, 0.1, clean.shape).astype(np.float32)
    out = smooth_series(noisy, 21)
    assert out.shape == noisy.shape and out.dtype == np.float32
    assert np.sqrt(np.mean((out - clean) ** 2)) < 0.4 * np.sqrt(np.mean((noisy - clean) ** 2))
    assert np.array_equal(smooth_series(noisy, None), noisy) and np.array_equal(smooth_series(noisy, 0), noisy)
    # an even window is widened to the next odd one; nodes do not bleed into each other
    assert np.array_equal(smooth_series(noisy, 20), smooth_series(noisy, 21))
    one = smooth_series(noisy[:, :, :1], 21)
    assert np.allclose(one[:, :, 0], out[:, :, 0])


def test_build_loaders_smooths_after_noise():
    from hetndl.data import build_loaders, smooth_series, add_noise
    data = np.cumsum(np.random.default_rng(0).normal(size=(6, 60, 3, 1)), 1).astype(np.float32) * 0.01
    tr, va, te, _ = build_loaders(data, 1, 1000, None, noise_std=0.1, smooth_window=11)
    obs = add_noise(data, np.random.default_rng(1 + 12345), None, 0.1)
    x_ref, _ = make_pairs(smooth_series(obs, 11), trajectory_split(6, 1)[1])
    x_va = va.dataset.tensors[0]
    assert torch.allclose(x_va, x_ref)
