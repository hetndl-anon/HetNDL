"""Switches behind the additional checks of the supplement: a fixed scale for the shared coupling (--gauge_scale ref), an
edge normalisation that is constant within an epoch (--edge_scale epoch), a training seed that leaves the data split alone
(--init_seed), the sign-free NOGI-style variant (orig_signed) and inhibitory edges in the MixOsc / MixOsc-E generators."""
import json

import numpy as np
import pytest
import torch

from hetndl.gen.a2_mixed import sample_classes2, simulate_a2
from hetndl.gen.a3_mixed import simulate_a3
from hetndl.gen.graphs import make_graph
from hetndl.gen.kuramoto import save_dataset, simulate_kuramoto
from hetndl.model import VARIANTS, build
from hetndl.train import main


def _build(variant="add_z_signed", N=5, edge_emb=0, edge_scale="batch", coupling="diffusive"):
    torch.manual_seed(0)
    gp, ode = build(variant, N, 1, 8, 2, 0.1, 1e-3, 1e-3, "rk4", True, coupling=coupling, edge_emb=edge_emb,
                    edge_scale=edge_scale)
    return gp.double(), ode.double()


def _kura(tmp_path):
    A = make_graph(10, "ER", seed=0, p=0.3)
    d = simulate_kuramoto(A, 2.0, 0.3, 0.2, False, 4, 0.5, 0.01, 0)
    p = str(tmp_path / "k.npz"); save_dataset(p, d)
    return p


ARGS = ["--epochs", "2", "--device", "cpu", "--eval_every", "1", "--batch", "16", "--num_workers", "0", "--coupling", "diffusive",
        "--feat", "trig", "--method", "rk4"]


def test_orig_signed_is_the_shared_function_model_without_the_sign_constraint():
    assert VARIANTS["orig_signed"] == dict(VARIANTS["orig"], mode="signed")
    gp, ode = build("orig_signed", 4, 1, 8, 2, 0.1, 1e-3, 1e-3, "rk4", True)
    with torch.no_grad():
        gp.raw.fill_(-1.0)
    gp.project()
    assert (gp.W()[~torch.eye(4, dtype=torch.bool)] < 0).all() and not ode.vf.additive and ode.vf.Z is None


def test_scale_projection_keeps_the_vector_field_and_sets_unit_rms():
    gp, ode = _build(); vf = ode.vf
    x = torch.randn(6, 5, 1, dtype=torch.float64); ref = torch.randn(4, 5, 1, dtype=torch.float64)
    before = vf(x, gp.W()).detach().clone(); l1 = float(gp.l1())
    r = float(vf.coupling_rms(ref))
    assert r > 0 and abs(r - 1.0) > 1e-3
    with torch.no_grad():
        gp.raw.mul_(r)
    vf.rescale_coupling(r)
    assert torch.allclose(vf(x, gp.W()), before, atol=1e-10, rtol=0)        # the same model
    assert abs(float(vf.coupling_rms(ref)) - 1.0) < 1e-9                    # in the gauge "coupling RMS = 1"
    assert abs(float(gp.l1()) - r * l1) < 1e-9 * max(1.0, l1)              # where the penalty has a definite value


def test_scale_projection_is_refused_where_it_is_not_defined():
    for kw in (dict(variant="orig", coupling="free"), dict(edge_emb=2)):
        gp, ode = _build(**kw)
        with pytest.raises(ValueError):
            ode.vf.coupling_rms(torch.randn(2, 5, 1, dtype=torch.float64))


def test_fixed_edge_scale_removes_the_dependence_on_the_other_samples_of_a_batch():
    x = torch.randn(2, 5, 1, dtype=torch.float64); y = x.clone(); y[1] = 10 * torch.randn(5, 1, dtype=torch.float64)
    for mode, same in (("batch", False), ("epoch", True)):
        gp, ode = _build(edge_emb=2, edge_scale=mode); vf = ode.vf.train()
        if mode == "epoch":
            vf.refresh_edge_scale(x)
        a = vf(x, gp.W())[0].detach().clone(); b = vf(y, gp.W())[0].detach()
        assert torch.allclose(a, b, atol=1e-12) == same, mode


def test_refresh_edge_scale_stores_the_rms_per_edge_and_is_used_in_training_and_evaluation():
    gp, ode = _build(edge_emb=2, edge_scale="epoch"); vf = ode.vf
    x = torch.randn(8, 5, 1, dtype=torch.float64)
    vf.refresh_edge_scale(x)
    g = vf.pair_coupling(x)
    assert torch.allclose(vf.edge_scale, g.pow(2).mean(dim=(0, 3)).add(1e-12).sqrt())
    before = vf.edge_scale.clone()
    out_train = vf.train()(x, gp.W()); out_eval = vf.eval()(x, gp.W())
    assert torch.equal(vf.edge_scale, before) and torch.allclose(out_train, out_eval)


def test_training_runs_with_the_fixed_scale_and_records_it(tmp_path):
    p = _kura(tmp_path); out = str(tmp_path / "g")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--gauge_scale", "ref"] + ARGS)
    mj = json.load(open(out + "/metrics.json"))
    assert not m["failed"] and mj["args"]["gauge_scale"] == "ref" and np.isfinite(mj["val_mse_best"])
    assert abs(mj["coupling_rms_last"] - 1.0) < 1e-3            # the scale is fixed at the end of training


def test_gauge_scale_needs_a_shared_additive_coupling(tmp_path):
    p = _kura(tmp_path)
    with pytest.raises(ValueError):
        main(["--data", p, "--model", "orig", "--out", str(tmp_path / "o"), "--gauge_scale", "ref", "--epochs", "1",
              "--device", "cpu", "--batch", "16", "--num_workers", "0", "--method", "rk4"])


def test_training_runs_with_the_epoch_edge_scale(tmp_path):
    p = _kura(tmp_path); out = str(tmp_path / "e")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--edge_emb", "2", "--edge_scale", "epoch"] + ARGS)
    assert not m["failed"] and json.load(open(out + "/metrics.json"))["args"]["edge_scale"] == "epoch"


def test_default_switches_change_nothing_and_init_seed_changes_only_the_training(tmp_path):
    p = _kura(tmp_path)
    run = lambda name, extra: main(["--data", p, "--model", "add_z_signed", "--out", str(tmp_path / name), "--seed", "1"] + ARGS + extra)
    a = run("a", []); b = run("b", ["--gauge_scale", "none", "--edge_scale", "batch"]); c = run("c", ["--init_seed", "1"])
    d = run("d", ["--init_seed", "7"])
    W = lambda n: np.load(str(tmp_path / n / "W_last.npy"))
    assert np.array_equal(W("a"), W("b")) and np.array_equal(W("a"), W("c"))     # defaults and init_seed = seed: identical
    assert not np.array_equal(W("a"), W("d"))                                    # another training seed: another run
    assert a["test_idx"] == d["test_idx"]                                        # on the same held-out trajectories


def test_mixosc_generators_take_inhibitory_edges_and_keep_the_default_output():
    A = make_graph(12, "ER", seed=3, p=0.3, directed=True); cl = sample_classes2(12, seed=3)
    for sim in (simulate_a2, simulate_a3):
        d0 = sim(A, cl, 0.3, 2, 2.0, 0.05, 3); d1 = sim(A, cl, 0.3, 2, 2.0, 0.05, 3, neg_frac=0.0)
        assert np.array_equal(d0["data"], d1["data"]) and set(d0) == set(d1) and "W_true" not in d0     # default unchanged
        ds = sim(A, cl, 0.3, 2, 2.0, 0.05, 3, neg_frac=0.5)
        S = ds["S"]; n = int((A > 0).sum())
        assert np.array_equal(S != 0, A > 0) and int((S < 0).sum()) == round(0.5 * n) == ds["meta"]["n_neg"]
        assert np.array_equal(ds["W_true"], np.float32(0.3) * S) and np.array_equal(ds["classes"], d0["classes"])
        assert not np.array_equal(ds["data"], d0["data"]) and np.abs(ds["data"]).max() < 10
        if "etypes" in d0:
            assert np.array_equal(ds["etypes"], d0["etypes"])            # the coupling type of an edge does not depend on its sign


def test_mixosc_e_generator_can_use_two_bounded_laws():
    from hetndl.gen.a3_mixed import edge_coupling
    du = np.linspace(-3, 3, 7)
    assert np.allclose(edge_coupling(du, 0), du) and np.allclose(edge_coupling(du, 1), np.tanh(2 * du))          # default laws
    assert np.allclose(edge_coupling(du, 0, "sin,tanh"), np.sin(du)) and np.allclose(edge_coupling(du, 1, "tanh,tanh"), np.tanh(2 * du))
    A = make_graph(12, "ER", seed=3, p=0.3, directed=True); cl = sample_classes2(12, seed=3)
    d0 = simulate_a3(A, cl, 0.3, 2, 2.0, 0.05, 3); d1 = simulate_a3(A, cl, 0.3, 2, 2.0, 0.05, 3, laws="lin,tanh")
    assert np.array_equal(d0["data"], d1["data"]) and "laws" not in d0["meta"]
    d2 = simulate_a3(A, cl, 0.3, 2, 2.0, 0.05, 3, laws="sin,tanh", neg_frac=0.5, x_max=20.0)
    assert d2["meta"]["laws"] == "sin,tanh" and not np.array_equal(d2["data"], d0["data"]) and np.array_equal(d2["etypes"], d0["etypes"])
