import numpy as np

from hetndl.baselines.granger import granger_lasso, lagged_design
from hetndl.metrics import auroc_auprc
from tests.test_nri import _linear_system


def test_lagged_design_shapes_and_alignment():
    data = np.arange(2 * 6 * 3 * 2, dtype=np.float32).reshape(2, 6, 3, 2)
    X, Y = lagged_design(data, np.array([0, 1]), lags=2)
    assert X.shape == (2 * (6 - 2), 2 * 3 * 2) and Y.shape == (2 * (6 - 2), 3 * 2)
    # first row of trajectory 0: target x_2, regressors [x_1, x_0] (most recent lag first)
    assert np.array_equal(Y[0], data[0, 2].ravel()) and np.array_equal(X[0, :6], data[0, 1].ravel()) and np.array_equal(X[0, 6:], data[0, 0].ravel())


def test_granger_recovers_linear_network():
    data, A = _linear_system()
    W, info = granger_lasso(data, seed=0)
    assert W.shape == A.shape and np.all(np.diag(W) == 0) and (W >= 0).all()
    au, ap = auroc_auprc(W, A)
    assert au > 0.95, au
    assert info["alpha"] > 0 and np.isfinite(info["val_mse"])
