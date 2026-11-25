import numpy as np

from hetndl.metrics import (offdiag, auroc_auprc, sign_accuracy, spurious_per_node,
                            spearman_z, cluster_ari)


def test_offdiag():
    M = np.arange(9).reshape(3, 3)
    assert offdiag(M).tolist() == [1, 2, 3, 5, 6, 7]


def test_auroc_perfect_sign_and_spurious():
    rng = np.random.default_rng(0)
    A = (rng.random((20, 20)) < 0.2).astype(np.float32)
    np.fill_diagonal(A, 0)
    S = A * np.where(rng.random((20, 20)) < 0.5, -1.0, 1.0)
    au, ap = auroc_auprc(S, A)  # |S| == A -> perfect ranking
    assert au == 1.0 and ap == 1.0
    assert sign_accuracy(S, S) == 1.0 and sign_accuracy(-S, S) == 0.0
    sp = spurious_per_node(A, A)
    assert sp.shape == (20,) and sp.sum() == 0
    # one wrong edge placed into target node 3 with the largest weight
    Wbad = A.copy()
    j = int(np.where(A[:, 3] == 0)[0][1])
    Wbad[j, 3] = 5.0
    assert spurious_per_node(Wbad, A)[3] >= 1


def test_spearman_and_ari():
    rng = np.random.default_rng(0)
    p = rng.random(50)
    Z = (2 * p + 1)[:, None]
    assert spearman_z(Z, p) > 0.999
    assert spearman_z(-Z, p) > 0.999  # sign of z is a free gauge
    Z3 = np.concatenate([rng.normal(c, 0.05, (30, 2)) for c in (-5, 0, 5)])
    c3 = np.repeat([0, 1, 2], 30)
    assert cluster_ari(Z3, c3, 3) == 1.0
