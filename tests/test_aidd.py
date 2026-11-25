import numpy as np
import torch

from hetndl.baselines.aidd import StaticGumbelGraph, aidd
from hetndl.metrics import auroc_auprc
from tests.test_nri import _linear_system


def test_static_graph_sampling_and_matrix():
    g = StaticGumbelGraph(N=5)
    z = g.sample(tau=0.5, hard=False)
    assert z.shape == (5 * 4, 2) and torch.allclose(z.sum(-1), torch.ones(20))
    P = g.edge_matrix()
    assert P.shape == (5, 5) and torch.all(torch.diagonal(P) == 0) and (P >= 0).all() and (P <= 1).all()


def test_aidd_recovers_linear_network():
    data, A = _linear_system()
    W, info = aidd(data, seed=0, epochs=60, hidden=32, win=20, pred_steps=5, lr=5e-3, lr_graph=5e-2, device="cpu")
    assert W.shape == A.shape and np.all(np.diag(W) == 0) and 0 <= W.min() and W.max() <= 1
    au, ap = auroc_auprc(W, A)
    assert au > 0.85, au
    assert info["best_epoch"] >= 1


def test_aidd_trig_features_double_the_state_dimension():
    data, A = _linear_system(N=5, S=6, T=30)
    W, info = aidd(data, seed=0, epochs=1, hidden=8, win=10, pred_steps=5, device="cpu", feat="trig")
    assert W.shape == (5, 5) and info["state_dim"] == 2 and info["feat"] == "trig"
    W, info = aidd(data, seed=0, epochs=1, hidden=8, win=10, pred_steps=5, device="cpu")
    assert info["state_dim"] == 1 and info["feat"] == "raw"


def test_aidd_accepts_the_nri_loss_options():
    data, A = _linear_system(N=5, S=6, T=30)
    W, info = aidd(data, seed=0, epochs=1, hidden=8, win=10, pred_steps=5, device="cpu", feat="trig",
                   loss_norm="sum_time", input_norm="minmax", var=5e-5)
    assert W.shape == (5, 5) and info["loss_norm"] == "sum_time"
