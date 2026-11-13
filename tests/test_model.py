import torch

from hetndl.model import GraphParam, VectorField, GraphNeuralODE, VARIANTS


def _vf(additive, use_z, N=4, d=1, d_z=1):
    torch.manual_seed(0)
    return VectorField(d, 16, additive, use_z, d_z, N)


def test_diag_mask_blocks_gradient():
    gp = GraphParam(4, "nonneg")
    vf = _vf(True, False)
    x = torch.randn(2, 4, 1)
    vf(x, gp.W()).sum().backward()
    assert torch.allclose(torch.diagonal(gp.raw.grad), torch.zeros(4))
    gp2 = GraphParam(4, "nonneg", mask_diag=False)
    vf2 = _vf(True, False)
    vf2(x, gp2.W()).sum().backward()
    assert torch.diagonal(gp2.raw.grad).abs().sum() > 0


def test_additive_decomposition():
    vf = _vf(True, True)
    x = torch.randn(3, 4, 1)
    W = torch.rand(4, 4)
    base = vf(x, torch.zeros(4, 4))
    full = vf(x, W)
    xi = x.unsqueeze(2).expand(3, 4, 4, 1)
    xj = x.unsqueeze(1).expand(3, 4, 4, 1)
    G = vf.G(torch.cat([xi, xj], -1))  # (B, i, j, d)
    manual = (G * W.T.unsqueeze(0).unsqueeze(-1)).sum(2)
    assert torch.allclose(full - base, manual, atol=1e-5)
    # F alone reproduces the W = 0 output
    assert torch.allclose(base, vf.F(torch.cat([x, vf.Z.unsqueeze(0).expand(3, 4, 1)], -1)), atol=1e-6)


def test_direction_in_model_additive():
    vf = _vf(True, False, N=2)
    W = torch.zeros(2, 2)
    W[0, 1] = 1.0  # 0 -> 1
    x = torch.randn(1, 2, 1)
    J = torch.autograd.functional.jacobian(lambda v: vf(v, W), x)[0, :, 0, 0, :, 0]  # (i, j)
    assert J[0, 1].abs() < 1e-7 and J[1, 0].abs() > 1e-7


def test_direction_in_model_nonadditive():
    vf = _vf(False, False, N=2)
    W = torch.zeros(2, 2)
    W[0, 1] = 1.0
    x = torch.randn(1, 2, 1)
    J = torch.autograd.functional.jacobian(lambda v: vf(v, W), x)[0, :, 0, 0, :, 0]
    assert J[0, 1].abs() < 1e-7 and J[1, 0].abs() > 1e-7


def test_z_only_in_F():
    vf = _vf(True, True)
    x = torch.randn(2, 4, 1)
    W = torch.rand(4, 4)
    base1 = vf(x, torch.zeros(4, 4))
    inter1 = vf(x, W) - base1
    with torch.no_grad():
        vf.Z.add_(1.0)
    base2 = vf(x, torch.zeros(4, 4))
    inter2 = vf(x, W) - base2
    assert torch.allclose(inter1, inter2, atol=1e-6)
    assert not torch.allclose(base1, base2)


def test_nonneg_projection_and_signed_free():
    gp = GraphParam(5, "nonneg")
    gp.raw.data -= 0.7
    gp.project()
    assert (gp.raw >= 0).all()
    gs = GraphParam(5, "signed")
    gs.raw.data -= 0.7
    gs.project()
    assert (gs.raw < 0).any()


def test_ode_step_shapes_and_time():
    vf = _vf(False, False)
    gp = GraphParam(4, "nonneg")
    ode = GraphNeuralODE(vf, t_end=0.01, rtol=1e-3, atol=1e-3)
    x0 = torch.randn(2, 4, 1)
    out = ode(x0, gp.W())
    assert out.shape == (2, 4, 1)
    # one short step moves the state by about t_end * vf(x0)
    assert torch.allclose(out - x0, 0.01 * vf(x0, gp.W()), atol=1e-3)


def test_variants_table():
    assert set(VARIANTS) == {"orig", "orig_z", "add", "add_z", "add_signed", "add_z_signed", "add_pm", "add_z_pm", "orig_signed"}
    assert VARIANTS["orig"] == {"additive": False, "use_z": False, "mode": "nonneg"}
    assert VARIANTS["add_z_signed"] == {"additive": True, "use_z": True, "mode": "signed"}


def test_adaptive_solver_has_step_cap():
    vf = _vf(True, False)
    ode = GraphNeuralODE(vf, t_end=0.01, rtol=1e-3, atol=1e-3, method="dopri5")
    assert ode.options.get("max_num_steps") == 2000
    ode2 = GraphNeuralODE(vf, t_end=0.01, rtol=1e-3, atol=1e-3, method="rk4")
    assert "max_num_steps" not in (ode2.options or {})
    assert ode2(torch.randn(2, 4, 1), GraphParam(4, "nonneg").W()).shape == (2, 4, 1)


def test_gauge_sign_flips_with_G():
    from hetndl.model import gauge_sign
    vf = _vf(True, True)
    x = torch.randn(64, 4, 1)
    s1 = gauge_sign(vf, x)
    with torch.no_grad():
        vf.G[-1].weight.neg_()
        vf.G[-1].bias.neg_()
    s2 = gauge_sign(vf, x)
    assert s1 in (-1, 1) and s2 == -s1
    assert gauge_sign(_vf(False, False), x) == 1  # non-additive variants carry no gauge


def test_trig_features_make_field_periodic():
    from hetndl.model import gauge_sign
    torch.manual_seed(0)
    for additive, use_z in ((True, True), (False, False)):
        vf = VectorField(1, 16, additive, use_z, 1, 4, feat="trig")
        W = torch.rand(4, 4)
        x = torch.randn(3, 4, 1) * 5
        out1 = vf(x, W)
        out2 = vf(x + 2 * 3.141592653589793, W)
        assert out1.shape == (3, 4, 1)
        assert torch.allclose(out1, out2, atol=1e-5)
        assert gauge_sign(vf, x) in (-1, 1)
    raw = VectorField(1, 16, True, False, 1, 4, feat="raw")
    x = torch.randn(3, 4, 1) * 5
    assert not torch.allclose(raw(x, torch.rand(4, 4)), raw(x + 2 * 3.141592653589793, torch.rand(4, 4)), atol=1e-3)


def test_diffusive_coupling_vanishes_at_coincident_states():
    import torch
    from hetndl.model import VectorField
    torch.manual_seed(0)
    N = 5
    vf = VectorField(1, 8, additive=True, use_z=True, d_z=1, N=N, feat="trig", coupling="diffusive")
    x = torch.full((2, N, 1), 0.7)
    W1, W2 = torch.rand(N, N), 3 * torch.rand(N, N)
    out1, out2 = vf(x, W1), vf(x, W2)
    assert torch.allclose(out1, out2, atol=1e-6)  # coupling term is zero when x_j = x_i
    fin = torch.cat([vf.phi(x), vf._z(2)], dim=-1)
    assert torch.allclose(out1, vf.F(fin), atol=1e-6)


def test_diffusive_coupling_matches_explicit_difference():
    import torch
    from hetndl.model import VectorField
    torch.manual_seed(1)
    N = 4
    vf = VectorField(1, 8, additive=True, use_z=False, d_z=1, N=N, feat="raw", coupling="diffusive")
    x, W = torch.randn(3, N, 1), torch.rand(N, N)
    h = vf.phi(x)
    ref = vf.F(h).clone()
    for i in range(N):
        gii = vf.G(torch.cat([h[:, i], h[:, i]], -1))
        for j in range(N):
            ref[:, i] += W[j, i] * (vf.G(torch.cat([h[:, i], h[:, j]], -1)) - gii)
    assert torch.allclose(vf(x, W), ref, atol=1e-5)


def test_coupling_rejects_unknown_and_nonadditive():
    import pytest
    from hetndl.model import VectorField
    with pytest.raises(ValueError):
        VectorField(1, 8, additive=True, use_z=False, d_z=1, N=3, coupling="bogus")
    with pytest.raises(ValueError):
        VectorField(1, 8, additive=False, use_z=False, d_z=1, N=3, coupling="diffusive")


def test_pm_mode_is_difference_of_nonnegatives():
    import torch
    from hetndl.model import GraphParam, VARIANTS
    torch.manual_seed(0)
    gp = GraphParam(6, "pm")
    assert VARIANTS["add_z_pm"] == {"additive": True, "use_z": True, "mode": "pm"}
    with torch.no_grad():
        gp.raw.sub_(0.8)
        gp.raw_neg.sub_(0.8)
    gp.project()
    assert (gp.raw >= 0).all() and (gp.raw_neg >= 0).all()
    W = gp.W()
    assert torch.allclose(W, (gp.raw - gp.raw_neg) * gp.mask)
    assert torch.allclose(torch.diagonal(W), torch.zeros(6))
    with torch.no_grad():
        gp.raw.zero_(); gp.raw_neg.fill_(0.3)
    assert (gp.W()[gp.mask > 0] < 0).all()  # can represent negative edges


def test_pm_l1_counts_both_parts():
    import torch
    from hetndl.model import GraphParam
    gp = GraphParam(4, "pm")
    with torch.no_grad():
        gp.raw.fill_(0.5); gp.raw_neg.fill_(0.5)
    assert float(gp.W().abs().sum()) == 0.0
    assert abs(float(gp.l1()) - 12.0) < 1e-6  # 0.5 + 0.5 on 12 off-diagonal entries
    gp2 = GraphParam(4, "signed")
    with torch.no_grad():
        gp2.raw.fill_(-0.5)
    assert abs(float(gp2.l1()) - 6.0) < 1e-6


def test_prox_soft_threshold_signed():
    import torch
    from hetndl.model import GraphParam
    gp = GraphParam(3, "signed")
    with torch.no_grad():
        gp.raw.copy_(torch.tensor([[0.0, 0.5, -0.5], [0.05, 0.0, -0.05], [0.2, -0.2, 0.0]]))
    gp.prox(0.1)
    exp = torch.tensor([[0.0, 0.4, -0.4], [0.0, 0.0, 0.0], [0.1, -0.1, 0.0]])
    assert torch.allclose(gp.raw, exp, atol=1e-6)


def test_prox_nonneg_and_pm_and_zero_tau():
    import torch
    from hetndl.model import GraphParam
    gp = GraphParam(2, "nonneg")
    with torch.no_grad():
        gp.raw.copy_(torch.tensor([[0.3, 0.05], [0.0, 1.0]]))
    gp.prox(0.1)
    assert torch.allclose(gp.raw, torch.tensor([[0.2, 0.0], [0.0, 0.9]]))
    pm = GraphParam(2, "pm")
    with torch.no_grad():
        pm.raw.fill_(0.3); pm.raw_neg.fill_(0.05)
    pm.prox(0.1)
    assert torch.allclose(pm.raw, torch.full((2, 2), 0.2)) and torch.allclose(pm.raw_neg, torch.zeros(2, 2))
    before = pm.raw.clone()
    pm.prox(0.0)
    assert torch.equal(before, pm.raw)


def test_z_standardize_makes_output_invariant_to_affine_z():
    import torch
    from hetndl.model import VectorField
    torch.manual_seed(0)
    vf = VectorField(1, 8, additive=True, use_z=True, d_z=2, N=6, coupling="diffusive", z_norm="standardize")
    x, W = torch.randn(3, 6, 1), torch.rand(6, 6)
    out1 = vf(x, W)
    with torch.no_grad():
        vf.Z.mul_(torch.tensor([3.0, 0.5])).add_(torch.tensor([2.0, -1.0]))  # per-dimension scale and shift
    assert torch.allclose(vf(x, W), out1, atol=1e-5)


def test_z_eff_is_standardized_per_dimension():
    import torch
    from hetndl.model import VectorField
    torch.manual_seed(0)
    vf = VectorField(1, 8, additive=True, use_z=True, d_z=2, N=7, z_norm="standardize")
    ze = vf.z_eff()
    assert ze.shape == (7, 2)
    assert torch.allclose(ze.mean(0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(ze.std(0, unbiased=False), torch.ones(2), atol=1e-4)


def test_z_norm_none_keeps_raw_z_and_rejects_unknown():
    import pytest, torch
    from hetndl.model import VectorField
    vf = VectorField(1, 8, additive=True, use_z=True, d_z=1, N=4)
    assert vf.z_norm == "none" and torch.equal(vf.z_eff(), vf.Z)
    with pytest.raises(ValueError):
        VectorField(1, 8, additive=True, use_z=True, d_z=1, N=4, z_norm="bogus")


def test_fixed_step_solver_substeps():
    # --step_size h makes a fixed-step solver (rk4) integrate [0, t_end] in t_end / h sub-steps
    import torch
    from hetndl.model import build
    gp, ode = build("add_z", 4, 1, 16, 1, 1.0, 1e-3, 1e-3, "rk4", True, step_size=0.25)
    assert ode.options.get("step_size") == 0.25
    gp2, ode2 = build("add_z", 4, 1, 16, 1, 1.0, 1e-3, 1e-3, "rk4", True)
    assert "step_size" not in ode2.options
    # a stiff-ish field: one coarse rk4 step and four sub-steps must differ measurably
    ode2.load_state_dict(ode.state_dict())
    x0 = torch.randn(2, 4, 1) * 3
    W = torch.rand(4, 4) * 5
    assert not torch.allclose(ode(x0, W), ode2(x0, W), atol=1e-6)
    gp3, ode3 = build("add_z", 4, 1, 16, 1, 1.0, 1e-3, 1e-3, "dopri5", True, step_size=0.25)
    assert "step_size" not in ode3.options                       # adaptive solvers ignore it


def test_signed_graph_sign_warmup_clamps_until_released():
    # a signed W kept non-negative during a warm-up, then free to change sign
    import torch
    from hetndl.model import GraphParam
    g = GraphParam(4, "signed")
    with torch.no_grad():
        g.raw.copy_(torch.tensor([[0., -1., 2., -3.]] * 4))
    g.clamp_signed = True; g.project()
    assert (g.raw >= 0).all() and g.raw[0, 2] == 2.0
    with torch.no_grad():
        g.raw[0, 1] = -0.5
    g.clamp_signed = False; g.project()
    assert g.raw[0, 1] == -0.5                                  # released: negative values survive
    assert GraphParam(3, "signed").clamp_signed is False        # default: never clamped
