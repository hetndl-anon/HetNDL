"""SGODE baseline (hetndl/baselines/sgode.py): graph orientation, the v3 parameterisation, windows, protocol rules and a
CPU smoke run."""
import json
import os

import numpy as np
import pytest
import torch

from hetndl.baselines.sgode import HORIZON, SGODE, SGODEFunc, main, protocol, sgode_scores, take, window_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_graph_orientation_is_sender_row_receiver_column():
    """W[j, i] must be the weight with which node j enters the derivative of node i."""
    torch.manual_seed(0)
    f = SGODEFunc(4, hidden=3, variant="v1")
    with torch.no_grad():
        f.K_free.zero_(); f.K_free[2, 1] = 1.0; f.c.zero_()         # only node 1 -> node 2
        f.wt.weight.copy_(torch.eye(3)); f.wt.bias.zero_()
    h = torch.zeros(1, 4, 3); h[0, 1] = 1.0                        # only node 1 is active
    dh = f(0.0, h)[0]
    assert dh[2].abs().sum() > 0 and dh[[0, 1, 3]].abs().sum() == 0
    m = SGODE(4, 1, "v1", dt=0.1, method="rk4", hidden=3)
    m.func = f
    W = m.graph()
    assert W[1, 2] == 1.0 and np.count_nonzero(W) == 1


def test_v3_matrix_is_difference_of_two_relu_products_and_is_signed():
    torch.manual_seed(0)
    f = SGODEFunc(6, variant="v3")
    with torch.no_grad():
        f.E3.mul_(5.0)
    K = f.K()
    ref = torch.relu(f.E1 @ f.E2.T) - torch.relu(f.E3 @ f.E4.T)
    assert torch.equal(K, ref) and (K < 0).any()


def test_both_variants_start_from_the_same_distribution():
    torch.manual_seed(3); a = SGODEFunc(8, variant="v1").K().detach()
    torch.manual_seed(3); b = SGODEFunc(8, variant="v3").K().detach()
    assert torch.allclose(a, b)


def test_windows_stay_inside_one_trajectory():
    X = torch.arange(3 * 20, dtype=torch.float32).reshape(3, 20, 1, 1)   # value = 20 * trajectory + time
    s, t = window_index([0, 2], 20)
    assert len(s) == 2 * (20 - HORIZON) and set(s.tolist()) == {0, 2}
    w = take(X, s, t)
    assert w.shape == (len(s), HORIZON + 1, 1, 1)
    first, last = w[:, 0, 0, 0], w[:, -1, 0, 0]
    assert torch.equal(last - first, torch.full_like(first, float(HORIZON)))       # consecutive snapshots
    assert torch.equal(torch.div(first, 20, rounding_mode="floor"), torch.div(last, 20, rounding_mode="floor"))


def test_protocol_matches_the_main_runs():
    assert protocol("kuramoto") == (600, "rk4", None, True)
    assert protocol("a1") == protocol("a2") == protocol("a3") == (1000, "dopri5", None, False)
    assert protocol("glv") == protocol("powergrid") == (1000, "rk4", None, False)
    assert protocol("netsim") == (200, "rk4", 0.25, False)
    with pytest.raises(ValueError):
        protocol("unknown")


def test_scores_use_absolute_weights_and_report_the_sign_gauge():
    A = np.zeros((3, 3)); A[0, 1] = A[1, 2] = 1; S = A.copy(); S[1, 2] = -1
    W = np.zeros((3, 3)); W[0, 1] = -2.0; W[1, 2] = 3.0            # perfect ranking, both signs flipped
    sc = sgode_scores(W, dict(A=A, S=S))
    assert sc["auroc"] == 1.0 and sc["sign_acc"] == 0.0 and sc["sign_acc_flip"] == 1.0


@pytest.mark.parametrize("variant", ["v1", "v3"])
def test_cpu_smoke_run_writes_metrics(tmp_path, variant):
    data = os.path.join(ROOT, "experiments", "data", "netsim_sim1.npz")
    if not os.path.exists(data):
        pytest.skip("netsim_sim1.npz not present")
    out = str(tmp_path / variant)
    main(["--data", data, "--seed", "1", "--variant", variant, "--device", "cpu", "--epochs", "2", "--n_traj", "5", "--out", out])
    m = json.load(open(os.path.join(out, "metrics.json")))
    assert m["method"] == "sgode" and m["variant"] == variant and not m["failed"] and m["epochs"] == 2
    assert m["solver"] == "rk4" and m["step_size"] == 0.25 and 0.0 <= m["auroc"] <= 1.0
    assert np.isfinite(m["val_l1_best"]) and np.isfinite(m["val_l1_persistence"])
    W = np.load(os.path.join(out, "W.npy")); assert W.shape == (5, 5)
