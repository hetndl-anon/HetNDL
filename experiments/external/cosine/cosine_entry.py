"""Run COSINE's run_discovery.main() unchanged except for one extra guard and one disabled plot: attributes of torch / F used in
LLM-generated basis expressions are limited to elementwise math, so an expression cannot reach file I/O such as
torch.save / torch.load. A rejected term is skipped by COSINE's own compile-failure path (RuntimeWarning), exactly as
for any other uncompilable term. The COSINE source in ./src is the anonymous repository COSINE-6D43
(zip in this directory, sha256 40e98da0...), not modified."""
import ast
import os
import runpy
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC)
from model.basis_engine import FlexibleMathEvaluator  # noqa: E402

ALLOWED = {
    # torch elementwise math and reductions named in COSINE's system prompt, plus common safe relatives
    "pow", "sqrt", "rsqrt", "clamp", "clip", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh",
    "tanh", "asinh", "acosh", "atanh", "exp", "expm1", "exp2", "log", "log1p", "log2", "log10", "sigmoid", "abs",
    "sign", "square", "reciprocal", "minimum", "maximum", "where", "erf", "floor", "ceil", "round", "fmod",
    "remainder", "norm", "zeros_like", "ones_like", "mean", "sum", "prod", "amax", "amin", "max", "min", "relu",
    "softplus", "gelu", "elu", "leaky_relu", "silu", "logsigmoid", "softsign", "tanhshrink", "hardtanh", "selu",
    "celu", "mish", "hardsigmoid", "hardswish", "softmax", "log_softmax", "logsumexp", "cumsum", "heaviside",
    "sinc", "deg2rad", "rad2deg", "float_power", "true_divide", "div", "mul", "add", "sub", "neg", "frac",
    "trunc", "nan_to_num", "isfinite", "sgn", "hypot", "cos_", "relu6",
}
_orig = FlexibleMathEvaluator._validate_attribute


def _guard(node):
    _orig(node)
    if isinstance(node.value, ast.Name) and node.value.id in ("torch", "F") and node.attr not in ALLOWED:
        raise ValueError("attribute %s.%s is not in the elementwise-math whitelist" % (node.value.id, node.attr))


FlexibleMathEvaluator._validate_attribute = staticmethod(_guard)

# COSINE's training-curve plot (utils.logger.TrainingLogger._plot_combined_curves) fails with the installed pandas /
# matplotlib pair ("Multi-dimensional indexing ... no longer supported"); the plot is cosmetic, so it is skipped.
from utils import logger as _cosine_logger  # noqa: E402
_cosine_logger.TrainingLogger._plot_combined_curves = lambda self, *a, **k: None
os.chdir(SRC)
sys.argv = [os.path.join(SRC, "run_discovery.py")] + sys.argv[1:]
runpy.run_path(os.path.join(SRC, "run_discovery.py"), run_name="__main__")
