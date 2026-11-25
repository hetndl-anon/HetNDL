"""Design option B for edge-function heterogeneity (edge embeddings, paper Sec. 4.7):
coupling c_ji = W_ji * Ghat(x_i, x_j; b_i, a_j) with an outgoing embedding a_j and an incoming embedding b_i that are
separate from the self-dynamics latent z, and a per-edge unit-RMS normalisation of the coupling function, so that the
(W, G) scale gauge is fixed and |W| stays comparable between edges."""
import pytest
import torch

from hetndl.model import VectorField, build


def _vf(N=4, d=1, d_e=2, coupling="diffusive"):
    torch.manual_seed(0)
    return VectorField(d, 16, True, True, 2, N, coupling=coupling, edge_emb=d_e)


def test_edge_emb_needs_additive_and_excludes_edge_z():
    with pytest.raises(ValueError):
        VectorField(1, 16, False, True, 2, 4, edge_emb=2)
    with pytest.raises(ValueError):
        VectorField(1, 16, True, True, 2, 4, edge_z=True, edge_emb=2)


def test_source_embedding_changes_coupling_but_not_self_term():
    vf = _vf(); vf.eval()
    x = torch.randn(3, 4, 1)
    W = torch.zeros(4, 4); W[0, 1] = 1.0                          # 0 -> 1 only
    out1 = vf(x, W).detach().clone()
    with torch.no_grad():
        vf.Ea[0] += 1.0                                            # source 0's outgoing embedding
    out2 = vf(x, W).detach()
    assert not torch.allclose(out1[:, 1], out2[:, 1])             # the target of 0 -> 1 changes
    assert torch.allclose(out1[:, [0, 2, 3]], out2[:, [0, 2, 3]])  # nobody else (F does not see a, b)


def test_diffusive_edge_coupling_vanishes_at_coincident_states():
    vf = _vf(); vf.eval()
    x = torch.full((2, 4, 1), 0.3)
    W = torch.rand(4, 4)
    assert torch.allclose(vf(x, W), vf(x, torch.zeros(4, 4)), atol=1e-6)


def test_per_edge_normalisation_fixes_the_scale_gauge_in_training_mode():
    vf = _vf(); vf.train()
    x = torch.randn(8, 4, 1); W = torch.rand(4, 4)
    out1 = vf(x, W).detach().clone()
    with torch.no_grad():                                          # rescale the coupling network's output layer
        last = [m for m in vf.G.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight *= 3.0; last.bias *= 3.0
    out2 = vf(x, W).detach()
    assert torch.allclose(out1, out2, atol=1e-5)


def test_eval_uses_running_scale_so_outputs_do_not_depend_on_the_batch():
    vf = _vf(); vf.train()
    for _ in range(50):
        vf(torch.randn(16, 4, 1), torch.rand(4, 4))                # build the running scale
    vf.eval()
    x = torch.randn(6, 4, 1); W = torch.rand(4, 4)
    full = vf(x, W)
    part = vf(x[:2], W)
    assert torch.allclose(full[:2], part, atol=1e-6)


def test_build_and_train_accept_edge_emb(tmp_path):
    gp, ode = build("add_z_signed", 4, 1, 16, 2, 0.1, 1e-3, 1e-3, "rk4", True, coupling="diffusive", edge_emb=2)
    assert ode.vf.edge_emb == 2 and ode.vf.Ea.shape == (4, 2) and ode.vf.Eb.shape == (4, 2)
    from tests.test_train_smoke import _dataset
    from hetndl.train import main
    p = _dataset(tmp_path); out = str(tmp_path / "run_ee")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "2", "--device", "cpu", "--eval_every", "1",
              "--batch", "16", "--num_workers", "0", "--coupling", "diffusive", "--edge_emb", "2"])
    assert m["failed"] is False and m["args"]["edge_emb"] == 2


@pytest.mark.parametrize("kind", ["shared", "edge_z", "edge_emb"])
def test_edge_coupling_matches_the_forward_contribution_of_one_edge(kind):
    # read-outs must evaluate a single edge's coupling with the model's own code path (one code path)
    torch.manual_seed(1)
    kw = dict(edge_z=True) if kind == "edge_z" else (dict(edge_emb=2) if kind == "edge_emb" else {})
    vf = VectorField(1, 16, True, True, 2, 4, coupling="diffusive", **kw)
    if kind == "edge_emb":
        vf.train()
        for _ in range(20):
            vf(torch.randn(16, 4, 1), torch.rand(4, 4))
    vf.eval()
    x = torch.randn(5, 4, 1)
    W = torch.zeros(4, 4); W[2, 1] = 0.7                       # 2 -> 1 only
    contrib = (vf(x, W) - vf(x, torch.zeros(4, 4)))[:, 1]     # (5, 1)
    c = vf.edge_coupling(x[:, 1], x[:, 2], i=1, j=2)          # (5, 1)
    assert torch.allclose(contrib, 0.7 * c, atol=1e-5)


# --- sign gauge for option B: G depends on (b_i, a_j), so the (W, G) -> (-W, -G) symmetry holds per edge.
# The read-out orients every edge with the shared-G functional (mean dG/dx_j at x_j = x_i over observed states),
# evaluated with that edge's own embeddings; gauge_sign then returns a +-1 matrix in W's [j, i] indexing.
class _SignBySource(torch.nn.Module):
    """G(x_i, x_j; b_i, a_j) = sign(a_j[-1]) * x_j for scalar raw states: dG/dx_j = sign(a_j[-1])."""
    def forward(self, inp):
        return inp[..., 1:2] * torch.sign(inp[..., -1:])


def test_gauge_sign_with_edge_emb_orients_each_edge_by_its_own_embeddings():
    from hetndl.model import gauge_sign
    vf = _vf(N=4); vf.eval(); vf.G = _SignBySource()
    with torch.no_grad():
        vf.Ea[:, -1] = torch.tensor([1.0, -1.0, 2.0, -0.5])        # sources 1 and 3 carry a flipped coupling
    g = torch.as_tensor(gauge_sign(vf, torch.randn(8, 4, 1)), dtype=torch.float32)
    expect = torch.sign(vf.Ea[:, -1].detach()).view(4, 1).expand(4, 4)   # row j = source j
    assert g.shape == (4, 4) and torch.equal(g, expect)


def test_gauge_sign_with_edge_emb_flips_with_G():
    from hetndl.model import gauge_sign
    vf = _vf(N=5); vf.eval(); x = torch.randn(8, 5, 1)
    g1 = torch.as_tensor(gauge_sign(vf, x))
    with torch.no_grad():
        last = [m for m in vf.G.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.neg_(); last.bias.neg_()
    g2 = torch.as_tensor(gauge_sign(vf, x))
    assert g1.shape == (5, 5) and torch.equal(g2, -g1)


def test_train_records_the_gauge_without_error_for_edge_emb(tmp_path):
    from tests.test_train_smoke import _dataset
    from hetndl.train import main
    p = _dataset(tmp_path); out = str(tmp_path / "run_ee_gauge")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "2", "--device", "cpu", "--eval_every", "1",
              "--batch", "16", "--num_workers", "0", "--coupling", "diffusive", "--edge_emb", "2"])
    assert not any("gauge" in e for e in m.get("eval_errors", [])), m.get("eval_errors")


def test_oriented_sign_accuracy_is_invariant_to_the_edge_sign_symmetry(tmp_path):
    # rescoring of saved option-B runs (trained before the per-edge gauge existed): reload the snapshot, orient each
    # edge on training states, score against the true signs. (W, G) -> (-W, -G) leaves the model unchanged, so the
    # score must not change (without the gauge it would become 1 - score).
    import numpy as np
    from tests.test_train_smoke import _dataset
    from hetndl.train import main
    from hetndl.edge_readout import oriented_sign_accuracy
    from hetndl.snapshot import array_path, state_path
    p = _dataset(tmp_path); out = str(tmp_path / "run_ee_rescore")
    m = main(["--data", p, "--model", "add_z_signed", "--out", out, "--epochs", "3", "--device", "cpu", "--eval_every", "1",
              "--batch", "16", "--num_workers", "0", "--coupling", "diffusive", "--edge_emb", "2"])
    acc1 = oriented_sign_accuracy(out)
    sp = state_path(out, m); st = torch.load(sp, map_location="cpu")
    last = max((k.rsplit(".", 1)[0] for k in st["vf"] if k.startswith("vf.G.") and k.endswith(".weight")),
               key=lambda k: int(k.split(".")[2]))
    st["vf"][last + ".weight"] = -st["vf"][last + ".weight"]; st["vf"][last + ".bias"] = -st["vf"][last + ".bias"]
    st["gp"]["raw"] = -st["gp"]["raw"]; torch.save(st, sp)
    ap = array_path(out, "W", m); np.save(ap, -np.load(ap))
    acc2 = oriented_sign_accuracy(out)
    assert 0.0 <= acc1 <= 1.0 and acc1 == acc2
