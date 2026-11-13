"""Helpers added for distilling the IEEE-118 swing model."""
import numpy as np


def test_coupling_output_dim_picks_the_component_that_varies():
    # swing model: the coupling acts on d omega / dt (second component); d theta / dt carries none
    from experiments.analysis.distill_pysr import coupling_output_dim
    rng = np.random.default_rng(0)
    c = np.stack([1e-4 * rng.standard_normal(50), np.sin(np.linspace(0, 6, 50))], -1)
    assert coupling_output_dim(c) == 1
    assert coupling_output_dim(c[:, ::-1]) == 0
    assert coupling_output_dim(c[:, 1:]) == 0                      # one-dimensional systems


def test_swing_class_truth_is_theta_dot_omega_and_linear_damping():
    # d theta / dt = omega ; d omega / dt = intercept + slope * omega (intercept = mean P_i / M, slope = -D / M)
    from experiments.analysis.distill_export import swing_class_truth
    grid = np.array([[0.1, -1.0], [0.3, 0.0], [-0.2, 2.0]])
    T = swing_class_truth(grid, intercept=0.4, slope=-2.5)
    assert T.shape == (3, 2)
    np.testing.assert_allclose(T[:, 0], grid[:, 1])
    np.testing.assert_allclose(T[:, 1], 0.4 - 2.5 * grid[:, 1])
