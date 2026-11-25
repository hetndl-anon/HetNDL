"""Per-node distillation of the self-fields (pernode_distill.py): the parts that do not need PySR. One form per read-out
class with constants refitted at every node; PySR's own selection rule when no form reaches R^2 >= 0.999; skeletons are
built from PySR's expression tree without sympy's automatic evaluation."""
import time
import numpy as np
import sympy as sp


def test_knee_takes_the_steepest_drop_of_the_log_loss():
    from experiments.analysis.pernode_distill import knee
    items = [(1, 1.0, "a"), (3, 0.5, "b"), (5, 0.01, "c"), (7, 0.009, "d"), (9, 0.008, "e"), (9, 0.02, "dup"), (11, 0.0085, "worse")]
    (_, k), front = knee(items)
    assert [p for _, _, p in front] == ["a", "b", "c", "d", "e"]          # one per complexity, strictly falling loss
    assert front[-k][2] == "c"


def test_knee_ignores_candidates_far_above_the_smallest_loss():
    from experiments.analysis.pernode_distill import knee
    (_, k), front = knee([(1, 1.0, "a"), (3, 0.1, "b"), (5, 0.09, "c"), (7, 0.05, "d")])
    assert front[-k][2] == "d"                                              # 0.1 and 0.09 exceed 1.5 x 0.05


def test_skeleton_frees_every_constant_and_keeps_the_function():
    from experiments.analysis.pernode_distill import skeleton
    x = sp.Symbol("x")
    sk, cs, p0 = skeleton("(cube(x) * -0.48304215) - x", [x])
    assert len(cs) == 1 and np.allclose(p0, [-0.48304215])
    g = np.linspace(-1, 1, 7)
    assert np.allclose(sp.lambdify([x] + list(cs), sk)(g, *p0), -0.48304215 * g ** 3 - g)


def test_skeleton_of_a_deeply_nested_expression_is_built_at_once():
    from experiments.analysis.pernode_distill import skeleton
    x = sp.Symbol("x")
    eq = "(tanh(tanh(square(cube(cube(cube(cube((-0.29271364 - x) - tanh(x)) - x)))) + x)) * -0.012083677) - (x * 0.9960371)"
    t = time.time(); sk, cs, p0 = skeleton(eq, [x])
    assert time.time() - t < 5 and len(cs) == 3


def test_refit_recovers_node_specific_constants():
    from experiments.analysis.pernode_distill import refit, skeleton
    x = sp.Symbol("x"); g = np.linspace(-1.5, 1.5, 64)
    sk, cs, p0 = skeleton("(x * -1.0) + (cube(x) * -0.5)", [x])
    true = np.array([[-0.8, -0.3], [-1.2, -0.6], [-1.0, -0.45]])          # (linear, cubic) coefficient per node
    F = np.stack([a * g + b * g ** 3 for a, b in true])
    fit, P = refit(sk, cs, p0, [x], [g], F)
    assert np.allclose(fit, F, atol=1e-8) and P.shape == (3, 2)
    # constants are ordered by their PySR value (-1.0 < -0.5): c0 is the linear, c1 the cubic coefficient
    assert np.allclose(P, true, atol=1e-6)
