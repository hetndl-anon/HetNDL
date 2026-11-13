"""Endpoint-type-dependent coupling: G(x_i, x_j, z_i, z_j) (flag edge_z)."""
import pytest
import torch

from hetndl.model import VectorField, build, gauge_sign


def _vf(edge_z=True, coupling="free", N=4, d=1, d_z=2):
    torch.manual_seed(0)
    return VectorField(d, 16, True, True, d_z, N, coupling=coupling, edge_z=edge_z)


def test_edge_z_requires_additive_and_z():
    with pytest.raises(ValueError):
        VectorField(1, 16, False, True, 1, 4, edge_z=True)
    with pytest.raises(ValueError):
        VectorField(1, 16, True, False, 1, 4, edge_z=True)


def test_source_z_changes_coupling_only_with_edge_z():
    x = torch.randn(2, 4, 1)
    W = torch.zeros(4, 4)
    W[0, 1] = 1.0  # 0 -> 1
    for edge_z, expect_change in ((True, True), (False, False)):
        vf = _vf(edge_z=edge_z)
        out1 = vf(x, W)[:, 1].detach().clone()
        with torch.no_grad():
            vf.Z[0] += 1.0  # the source's z
        out2 = vf(x, W)[:, 1].detach()
        assert (not torch.allclose(out1, out2)) == expect_change


def test_edge_z_additive_decomposition():
    vf = _vf()
    x = torch.randn(3, 4, 1)
    W = torch.rand(4, 4)
    base, full = vf(x, torch.zeros(4, 4)), vf(x, W)
    xi, xj = x.unsqueeze(2).expand(3, 4, 4, 1), x.unsqueeze(1).expand(3, 4, 4, 1)
    Z = vf.z_eff()
    zi, zj = Z.unsqueeze(1).expand(4, 4, 2).unsqueeze(0).expand(3, 4, 4, 2), Z.unsqueeze(0).expand(4, 4, 2).unsqueeze(0).expand(3, 4, 4, 2)
    G = vf.G(torch.cat([xi, xj, zi, zj], -1))
    manual = (G * W.T.unsqueeze(0).unsqueeze(-1)).sum(2)
    assert torch.allclose(full - base, manual, atol=1e-5)


def test_edge_z_diffusive_vanishes_at_equal_states():
    vf = _vf(coupling="diffusive")
    x = torch.randn(2, 1, 1).expand(2, 4, 1).contiguous()  # all nodes equal
    W = torch.rand(4, 4)
    assert torch.allclose(vf(x, W), vf(x, torch.zeros(4, 4)), atol=1e-6)
    # and differs from the free coupling on unequal states
    x2 = torch.randn(2, 4, 1)
    assert not torch.allclose(vf(x2, W), _vf(coupling="free")(x2, W))


def test_build_and_gauge_with_edge_z():
    gp, ode = build("add_z", 5, 1, 16, 2, 0.01, 1e-3, 1e-3, "rk4", True, coupling="diffusive", edge_z=True)
    assert ode.vf.edge_z
    s = gauge_sign(ode.vf, torch.randn(8, 5, 1))
    assert s in (1, -1)
    ode(torch.randn(2, 5, 1), gp.W())  # forward through the ODE
