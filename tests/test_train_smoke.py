import json

import numpy as np

from hetndl.gen.graphs import make_graph
from hetndl.gen.kuramoto import simulate_kuramoto, save_dataset
from hetndl.train import main


def _dataset(tmp_path, bad=False):
    A = make_graph(10, "ER", seed=0, p=0.3)
    d = simulate_kuramoto(A, 2.0, 0.3, 0.2, False, 4, 0.5, 0.01, 0)  # 4 traj x 50 steps
    if bad:
        d["data"][:, 5, 0, 0] = np.inf  # every trajectory carries a non-finite value
    p = str(tmp_path / "k.npz")
    save_dataset(p, d)
    return p


def test_train_smoke_writes_metrics(tmp_path):
    p = _dataset(tmp_path)
    out = str(tmp_path / "run")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "2", "--device", "cpu",
              "--eval_every", "1", "--batch", "16", "--num_workers", "0"])
    mj = json.load(open(out + "/metrics.json"))
    assert m["failed"] is False and mj["failed"] is False
    for k in ("best", "last"):
        for key in ("auroc", "auprc", "sign_acc", "spearman_z", "rollout_nrmse", "spurious_per_node"):
            assert key in mj[k], key
    assert mj["sec_per_epoch"] > 0 and mj["epochs_run"] == 2 and mj["best_epoch"] in (1, 2)
    assert "val_mse_best" in mj and "test_mse" in mj and mj["args"]["model"] == "add_z_signed"
    assert (tmp_path / "run" / "W_best.npy").exists()
    assert (tmp_path / "run" / "Z_best.npy").exists()
    assert (tmp_path / "run" / "W_last.npy").exists()
    assert (tmp_path / "run" / "log.csv").exists()
    assert np.load(str(tmp_path / "run" / "W_best.npy")).shape == (10, 10)


def test_train_marks_failed_on_nonfinite(tmp_path):
    p = _dataset(tmp_path, bad=True)
    out = str(tmp_path / "run2")
    m = main(["--data", p, "--model", "orig", "--out", out, "--epochs", "2", "--device", "cpu",
              "--batch", "16", "--num_workers", "0"])
    mj = json.load(open(out + "/metrics.json"))
    assert m["failed"] is True and mj["failed"] is True and "reason" in mj


def test_train_selection_records_penalised_objective(tmp_path):
    p = _dataset(tmp_path)
    out = str(tmp_path / "run3")
    m = main(["--data", p, "--model", "add", "--out", out, "--epochs", "2", "--device", "cpu",
              "--batch", "16", "--num_workers", "0", "--lambda_l1", "1e-3"])
    assert m["selection"] == "val_mse"  # the default
    assert m["val_obj_best"] >= m["val_mse_best"] - 1e-12
    out2 = str(tmp_path / "run4")
    m2 = main(["--data", p, "--model", "add", "--out", out2, "--epochs", "2", "--device", "cpu",
               "--batch", "16", "--num_workers", "0", "--lambda_l1", "1e-3", "--select", "val_obj"])
    assert m2["selection"] == "val_obj"
    assert m2["val_obj_best"] >= m2["val_mse_best"] - 1e-12
    assert m2["best"]["epoch"] == m2["by_obj"]["epoch"]


def test_train_snapshots_and_selection(tmp_path):
    import csv
    import os
    p = _dataset(tmp_path)
    out = str(tmp_path / "run5")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "3", "--device", "cpu",
              "--eval_every", "1", "--batch", "16", "--num_workers", "0"])
    mj = json.load(open(out + "/metrics.json"))
    for k in ("by_mse", "by_obj", "last", "best"):
        assert "auroc" in mj[k] and "val_mse" in mj[k] and "epoch" in mj[k], k
    assert mj["selection"] == "val_mse"
    log = list(csv.DictReader(open(out + "/log.csv")))
    assert abs(mj["by_mse"]["val_mse"] - min(float(r["val_mse"]) for r in log)) < 1e-12
    assert np.allclose(np.load(out + "/W_best.npy"), np.load(out + "/W_by_mse.npy"))
    for name in ("by_mse", "by_obj", "last"):
        assert os.path.exists(os.path.join(out, "state_%s.pt" % name))
    assert mj["best"]["gauge"] in (-1, 1)
    assert m["failed"] is False


def test_evaluate_is_fault_tolerant(tmp_path, monkeypatch):
    import os
    import hetndl.train as T
    from hetndl.gen.a1_mixed import simulate_a1, sample_classes

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(T, "cluster_ari", boom)
    A = make_graph(6, "ER", seed=0, p=0.4, directed=True)
    d = simulate_a1(A, sample_classes(6, seed=0), 0.5, 4, 0.5, 0.05, 0)
    p = str(tmp_path / "a1.npz")
    save_dataset(p, d)
    out = str(tmp_path / "run6")
    m = main(["--data", p, "--model", "add_z", "--d_z", "2", "--out", out, "--epochs", "1", "--device", "cpu",
              "--batch", "8", "--num_workers", "0", "--cluster_k", "3"])
    assert m["failed"] is False
    assert np.isnan(m["best"]["cluster_ari"]) and np.isfinite(m["best"]["auroc"])
    assert any("cluster_ari" in e for e in m["eval_errors"])
    assert os.path.exists(os.path.join(out, "W_last.npy"))


def test_train_prox_tau_sparsifies(tmp_path):
    p = _dataset(tmp_path)
    out = str(tmp_path / "run_prox")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "2", "--device", "cpu",
              "--batch", "16", "--num_workers", "0", "--prox_tau", "1.0"])
    assert m["failed"] is False and m["args"]["prox_tau"] == 1.0
    W = np.load(out + "/W_last.npy")
    assert (W == 0).mean() > 0.9  # threshold 1.0 per step zeroes U(0,1)-initialised weights


def test_train_defaults_lambda_z_zero_and_saves_standardized_z(tmp_path):
    p = _dataset(tmp_path)
    out = str(tmp_path / "run")
    main(["--data", p, "--model", "add_z", "--out", out, "--epochs", "1", "--device", "cpu", "--eval_every", "1",
          "--batch", "16", "--num_workers", "0", "--d_z", "2", "--z_norm", "standardize"])
    mj = json.load(open(out + "/metrics.json"))
    assert mj["args"]["lambda_z"] == 0.0 and mj["args"]["z_norm"] == "standardize"
    Z = np.load(out + "/Z_last.npy")  # the z that F actually sees
    assert np.allclose(Z.mean(0), 0, atol=1e-5) and np.allclose(Z.std(0), 1, atol=1e-3)


def test_train_accepts_weight_decay_and_smoothing(tmp_path):
    p = _dataset(tmp_path)
    out = str(tmp_path / "run5")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "1", "--device", "cpu",
              "--eval_every", "1", "--batch", "16", "--num_workers", "0", "--weight_decay", "1e-3",
              "--noise_std", "0.05", "--smooth_window", "7"])
    assert m["failed"] is False
    assert m["args"]["weight_decay"] == 1e-3 and m["args"]["smooth_window"] == 7


def test_train_sign_warmup_runs_and_is_recorded(tmp_path):
    p = _dataset(tmp_path)
    out = str(tmp_path / "run6")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "2", "--device", "cpu",
              "--eval_every", "1", "--batch", "16", "--num_workers", "0", "--sign_warmup", "1"])
    assert m["failed"] is False and m["args"]["sign_warmup"] == 1


def test_release_sign_resets_adam_state_and_scales_lr():
    # at the end of the sign warm-up the clamped W entries carry Adam moments that push them
    # negative; release_sign() can clear W's Adam state and scale W's learning rate for the signed phase
    import torch
    from hetndl.model import GraphParam
    from hetndl.train import release_sign
    g = GraphParam(4, "signed"); opt = torch.optim.Adam(g.parameters(), lr=1e-2)
    g.W().sum().backward(); opt.step()
    assert len(opt.state) > 0
    release_sign(opt, reset=True, lr=1e-3)
    assert len(opt.state) == 0 and all(pg["lr"] == 1e-3 for pg in opt.param_groups)
    opt2 = torch.optim.Adam(g.parameters(), lr=1e-2); g.W().sum().backward(); opt2.step()
    release_sign(opt2, reset=False, lr=None)
    assert len(opt2.state) > 0 and opt2.param_groups[0]["lr"] == 1e-2       # defaults change nothing


def test_train_logs_w_stats_and_accepts_release_options(tmp_path):
    import csv
    p = _dataset(tmp_path)
    out = str(tmp_path / "run7")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "3", "--device", "cpu",
              "--eval_every", "1", "--batch", "16", "--num_workers", "0", "--sign_warmup", "1",
              "--release_reset", "1", "--release_lr_scale", "0.1", "--log_w_stats", "1"])
    assert m["failed"] is False and m["args"]["release_reset"] == 1 and m["args"]["release_lr_scale"] == 0.1
    rows = list(csv.DictReader(open(out + "/log.csv")))
    assert "w_neg_frac" in rows[0] and "w_l1" in rows[0] and float(rows[0]["w_neg_frac"]) == 0.0   # epoch 1: clamped


def test_graphparam_support_mask_zeroes_and_freezes_nonsupport():
    # prune-and-refit: a fixed edge support; W outside it is exactly 0 and receives no gradient
    import torch
    from hetndl.model import GraphParam
    g = GraphParam(4, "signed", mask_diag=False)
    B = torch.zeros(4, 4, dtype=torch.bool); B[0, 1] = B[2, 3] = True; B[1, 1] = True   # diagonal never kept
    g.set_support(B)
    W = g.W()
    assert W[0, 1] != 0 and W[2, 3] != 0 and int((W != 0).sum()) == 2
    W.sum().backward()
    assert int((g.raw.grad != 0).sum()) == 2


def test_train_init_from_with_2means_support(tmp_path):
    p = _dataset(tmp_path)
    src = str(tmp_path / "src")
    main(["--data", p, "--model", "add_z_signed", "--out", src, "--epochs", "2", "--device", "cpu",
          "--eval_every", "1", "--batch", "16", "--num_workers", "0"])
    out = str(tmp_path / "refit")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "1", "--device", "cpu",
              "--eval_every", "1", "--batch", "16", "--num_workers", "0", "--lr_w", "0", "--lr_f", "0",
              "--init_from", src, "--w_support", "2means"])
    from hetndl.readout import binarize_2means
    W0 = np.load(src + "/W_best.npy"); B = binarize_2means(W0)
    assert m["failed"] is False and m["args"]["init_from"] == src and m["support_size"] == int(B.sum())
    assert np.allclose(np.load(out + "/W_last.npy"), W0 * B, atol=1e-6)      # lr 0: source W restricted to support


def test_two_means_support_matches_kmeans_binarisation():
    # adaptive sign release: an exact 1-D 2-means split of |W| (cheap enough to run every epoch)
    from hetndl.readout import binarize_2means
    from hetndl.train import support_2means
    rng = np.random.default_rng(0)
    off = ~np.eye(12, dtype=bool)

    def sse(W, B):
        v, b = np.abs(W[off]), B[off]
        return sum(((v[m] - v[m].mean()) ** 2).sum() for m in (b, ~b))

    for _ in range(5):   # any |W|: the exact split is never worse than Lloyd's (KMeans can stop at a local optimum)
        W = np.abs(rng.normal(size=(12, 12))) * (rng.random((12, 12)) < 0.3) + 0.01 * rng.random((12, 12))
        assert sse(W, support_2means(W)) <= sse(W, binarize_2means(W)) + 1e-12
    for _ in range(5):   # a trained-like bimodal |W|: the same edge set
        A = rng.random((12, 12)) < 0.3
        W = A * (1 + 0.2 * rng.random((12, 12))) + 0.02 * rng.random((12, 12))
        assert (support_2means(W) == binarize_2means(W)).all()


def test_support_jaccard_release_rule():
    from hetndl.train import support_stable
    a = np.zeros((4, 4), bool); a[0, 1] = a[1, 2] = a[2, 3] = True
    b = a.copy(); b[3, 0] = True                                           # Jaccard 3/4
    assert support_stable(a, a, 0.95) and not support_stable(a, b, 0.95) and support_stable(a, b, 0.7)
    assert not support_stable(a, None, 0.95)


def test_train_release_on_stable_support_is_recorded(tmp_path):
    import csv
    p = _dataset(tmp_path)
    out = str(tmp_path / "run8")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "4", "--device", "cpu",
              "--eval_every", "1", "--batch", "16", "--num_workers", "0", "--sign_warmup", "4",
              "--release_stable", "1", "--release_jaccard", "0.0", "--log_w_stats", "1"])
    # Jaccard threshold 0: the support counts as stable at the first comparison (epoch 2), so the sign is
    # released from epoch 3 on, before the --sign_warmup cap of 4
    assert m["failed"] is False and m["release_epoch"] == 3
