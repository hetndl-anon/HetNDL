import numpy as np

from hetndl.readout import select_k, binarize_2means


def _blobs(k, n_per=20, d=6, seed=0, sep=8.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, (k, d))
    X = np.concatenate([c + rng.normal(0, 1, (n_per, d)) for c in centers])
    y = np.repeat(np.arange(k), n_per)
    return X, y


def test_select_k_finds_the_number_of_well_separated_groups():
    for k in (1, 2, 3):
        X, y = _blobs(k, seed=k)
        K, labels, bic = select_k(X, k_max=5)
        assert K == k, (k, K, bic)
        assert labels.shape == (len(X),) and len(np.unique(labels)) == K


def test_select_k_handles_one_dimensional_features():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-0.5, 0.03, 54), rng.normal(-2.5, 0.1, 64)])[:, None]   # gen / load damping slopes
    K, labels, _ = select_k(x, k_max=4)
    assert K == 2 and len(np.unique(labels[:54])) == 1 and labels[0] != labels[-1]


def test_binarize_2means_splits_scores_without_knowing_the_density():
    rng = np.random.default_rng(1)
    W = np.abs(rng.normal(0, 0.01, (30, 30))); A = np.zeros((30, 30))
    idx = rng.choice(900, 60, replace=False); W.flat[idx] = 1 + rng.random(60); A.flat[idx] = 1
    np.fill_diagonal(W, 0); np.fill_diagonal(A, 0)
    B = binarize_2means(W)
    assert B.dtype == bool and not np.diag(B).any()
    assert (B == (A > 0)).mean() > 0.99


def test_binarize_2means_returns_no_edge_for_an_all_zero_matrix():
    B = binarize_2means(np.zeros((6, 6)))
    assert B.dtype == bool and B.shape == (6, 6) and not B.any()


def test_binarize_2means_returns_no_edge_when_all_weights_are_equal():
    # one value only: there is no upper group, so no pair can be called an edge (signs do not matter: |W| is used)
    W = np.full((5, 5), 0.3); W[0, 1] = -0.3
    assert not binarize_2means(W).any()


def test_binarize_2means_ignores_the_diagonal_when_the_off_diagonal_is_constant():
    W = np.full((4, 4), 2.0); np.fill_diagonal(W, 7.0)
    assert not binarize_2means(W).any()


def test_select_k_curves_mode_ignores_flat_regions():
    # two families of curves that differ only on half of the grid; the flat half carries tiny noise that per-point
    # standardisation would blow up
    rng = np.random.default_rng(3); x = np.linspace(-1, 1, 40)
    base = [np.where(x > 0, x, 0), np.where(x > 0, np.tanh(3 * x) / 1.5, 0)]
    C = np.stack([base[i % 2] + rng.normal(0, 1e-3, 40) for i in range(40)])
    K, labels, _ = select_k(C, k_max=5, curves=True)
    assert K == 2 and len(np.unique(labels[::2])) == 1


def _vf(use_z=True, feat="trig"):
    from hetndl.model import VectorField
    return VectorField(1, 8, True, use_z, 2, 5, feat=feat, coupling="diffusive")


def test_node_constant_returns_the_constant_of_a_field_that_only_depends_on_z():
    import torch
    from hetndl.readout import node_constant
    vf = _vf()
    with torch.no_grad():
        vf.Z.copy_(torch.tensor([[0.3, 9.0], [-1.0, 9.0], [2.0, 9.0], [0.0, 9.0], [5.0, 9.0]]))

    class FirstZ(torch.nn.Module):            # F(h, z) = z_0, independent of the state
        def forward(self, inp):
            return inp[..., -2:-1]
    vf.F = FirstZ()
    states = np.linspace(-3, 3, 50)[:, None]
    out = node_constant(vf, states)
    assert out.shape == (5, 1)
    assert np.allclose(out[:, 0], [0.3, -1.0, 2.0, 0.0, 5.0], atol=1e-6)


def test_node_constant_averages_the_self_field_over_the_given_states():
    import torch
    from hetndl.readout import node_constant
    vf = _vf(feat="raw")

    class StatePlusZ(torch.nn.Module):        # F(x, z) = x + z_1 (raw features: h = x)
        def forward(self, inp):
            return inp[..., :1] + inp[..., -1:]
    vf.F = StatePlusZ()
    with torch.no_grad():
        vf.Z.zero_(); vf.Z[:, 1] = torch.arange(5, dtype=torch.float32)
    states = np.array([[1.0], [2.0], [6.0]])
    out = node_constant(vf, states)
    assert np.allclose(out[:, 0], 3.0 + np.arange(5), atol=1e-6)


def test_node_constant_without_z_gives_one_value_for_all_nodes():
    from hetndl.readout import node_constant
    vf = _vf(use_z=False)
    out = node_constant(vf, np.linspace(-1, 1, 20)[:, None])
    assert out.shape == (5, 1) and np.allclose(out, out[0])
