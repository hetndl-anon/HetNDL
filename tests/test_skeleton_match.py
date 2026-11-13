"""Skeleton comparison for Table 4: a distilled equation matches when, after expansion, it is a sum of exactly the
true basis terms (constants free); terms contributing less than `tol` of the expression's RMS on the sampled grid are
treated as numerical noise."""
import numpy as np


def test_cubic_class_matches_with_free_constants():
    from experiments.analysis.skeleton_match import skeleton_match
    assert skeleton_match("(cube(x) * -0.48304215) - x", "-x - 0.5*x**3", ["x"])


def test_nested_tanh_does_not_match_a_single_tanh():
    from experiments.analysis.skeleton_match import skeleton_match
    assert not skeleton_match("(tanh(tanh(x)) * 0.4148967) - x", "-1.2*x + 0.6*tanh(x)", ["x"])


def test_negligible_offset_is_dropped_only_with_a_grid():
    from experiments.analysis.skeleton_match import skeleton_match
    grid = np.linspace(-2, 2, 64)[:, None]
    assert skeleton_match("-0.0012216305 - x", "-x", ["x"], grid=grid, tol=0.01)
    assert not skeleton_match("-0.0012216305 - x", "-x", ["x"])


def test_missing_term_does_not_match():
    from experiments.analysis.skeleton_match import skeleton_match
    grid = np.linspace(-2, 2, 64)[:, None]
    assert not skeleton_match("x * -0.66137797", "-1.2*x + 0.6*tanh(x)", ["x"], grid=grid, tol=0.01)


def test_two_variable_coupling_and_symbolic_constants():
    from experiments.analysis.skeleton_match import skeleton_match
    assert skeleton_match("sin(theta_j - theta_i) * 4.921146", "sin(theta_j - theta_i)", ["theta_i", "theta_j"])
    assert skeleton_match("x*(c0*x + c1)", "0.9*x - 0.8*x**2", ["x"])
    assert skeleton_match("(v * (0.9976297 - square(u))) - u", "1.0*(1 - u**2)*v - u", ["u", "v"])
