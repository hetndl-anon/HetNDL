"""Input files of the runner for the original NRI code (the code itself is external and not needed here)."""
import os

import numpy as np

import experiments.launch.nri_original_run as nor
from hetndl.data import trajectory_split
from hetndl.gen.kuramoto import load_dataset, main as kuramoto


def test_nri_feats_are_the_inputs_of_the_nri_paper():
    rng = np.random.default_rng(0); th = rng.normal(size=(2, 6, 3)); w = rng.uniform(1, 10, (2, 3))
    F = nor.nri_feats(th, w, 0.1)                                  # (S, N, T - 1, 3)
    assert F.shape == (2, 3, 5, 3) and F.dtype == np.float32
    assert np.allclose(F[..., 0], (np.diff(th, axis=1) / 0.1).transpose(0, 2, 1), atol=1e-4)
    assert np.allclose(F[..., 1], np.sin(th[:, :-1]).transpose(0, 2, 1), atol=1e-6)
    assert np.allclose(F[..., 2], np.broadcast_to(w[:, :, None], (2, 3, 5)))


def test_bench_inputs_follow_our_split_and_hold_no_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(nor, "OFF", str(tmp_path)); os.makedirs(tmp_path / "data")
    src = str(tmp_path / "k.npz")
    kuramoto(["--N", "4", "--samples", "10", "--T", "1.0", "--dt", "0.01", "--seed", "1", "--out", src])   # 100 states
    suffix, (N, dims) = nor.prepare_bench(src, 1, None)
    assert suffix == "_tab2_k_s1" and (N, dims) == (4, 2)                    # Kuramoto: (cos, sin) of the phases
    d = load_dataset(src); tr, va, te = trajectory_split(10, 1)
    f = np.load(str(tmp_path / "data" / ("feat_train%s.npy" % suffix)))
    e = np.load(str(tmp_path / "data" / ("edges_train%s.npy" % suffix)))
    assert f.shape == (2 * len(tr), 4, 49, 2) and e.shape == (2 * len(tr), 4, 4)   # two windows of 49 states each
    assert np.allclose(f[0, :, :, 0], np.cos(d["data"][tr[0], :49, :, 0]).T, atol=1e-6)
    assert np.allclose(f[1, :, :, 1], np.sin(d["data"][tr[0], 49:98, :, 0]).T, atol=1e-6)
    assert not e.any()                                                       # all-zero edge labels


def test_train_arguments_and_batch_fallback_parse():
    a = nor.parse(["--mode", "bench", "--data", "x.npz", "--seed", "2", "--epochs", "600", "--batch", "32", "--out", "o"])
    assert (a.mode, a.seed, a.batch, a.n_traj, a.device) == ("bench", 2, 32, None, "cuda:0")
    args = nor.train_args("_tab2_x_s2", 30, 1, 600, 16, 2, "o")
    assert args[args.index("--batch-size") + 1] == "16" and "--skip-first" in args and "--no-cuda" not in args
    assert args[args.index("--encoder") + 1] == "cnn" and args[args.index("--timesteps") + 1] == "49"
