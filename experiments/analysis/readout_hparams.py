"""How the fixed hyperparameters of the class read-out (Sect. 4.6) affect the
number of classes K and the ARI. The defaults (3 principal components, K <= 5, BIC tolerance 10, covariance floor 0.05)
were fixed before the main runs and stay the defaults whatever this script prints. Fixed before the first run:
  runs      all 15 reported runs with node classes: MixRelax, MixRelax (BA), MixOsc N=30, MixOsc N=60, Grid-118, data
            seeds 1-3 (runs/sel/selected.json) - every easy and every failing case, so that no example is picked;
  features  exactly those of the paper's read-out (readout_split_check.readout with all trajectories; captured from it);
  changes   one at a time from the default:
              n_pc       2, 3*, 5, 10        (number of principal components)
              k_max      3, 5*, 8            (largest K tried)
              tol        0, 2, 6, 10*, 20    (smallest K whose BIC is within tol of the minimum; 0 = the BIC minimum)
              reg        0.01, 0.05*, 0.1    (covariance floor in standardised units)
            (* = default; the default row must reproduce the paper's K and ARI, asserted);
  output    K and ARI per run and change; per setting the mean ARI and the K values; a count of runs whose K changes.
Reading: report in the supplement which choices change K or ARI on which runs, whether the failing runs (MixOsc N=60
seed 1, MixRelax (BA)) are rescued or the easy ones broken. No default is changed. Truth (classes) only in the ARI.
Run from the project root with PYTHONPATH=. (-> results/supplement/readout_hparams.txt)."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from hetndl.edge_readout import load_run
from hetndl.readout import select_k
import experiments.analysis.readout_split_check as rsc
from experiments.analysis.baseline_table import B, PAPER_NAME, sel, sel_key
SETTINGS = ("MixRelax", "MixRelax (BA)", "MixOsc N=30", "MixOsc N=60", "Grid-118")
DEFAULT = dict(n_pc=3, k_max=5, tol=10.0, reg=0.05)
GRID = [("n_pc", (2, 3, 5, 10)), ("k_max", (3, 5, 8)), ("tol", (0.0, 2.0, 6.0, 10.0, 20.0)), ("reg", (0.01, 0.05, 0.1))]


def select_k_p(X, n_pc, k_max, tol, reg, curves, seed=0):
    """hetndl.readout.select_k with its four constants as arguments (same code otherwise)."""
    X = np.asarray(X, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    Z = (X - X.mean(0)) / ((X - X.mean(0)).std() + 1e-12) if curves else (X - X.mean(0)) / (X.std(0) + 1e-12)
    if Z.shape[1] > n_pc:
        Z = PCA(n_pc, random_state=seed).fit_transform(Z)
    k_max = int(min(k_max, len(Z) - 1)); bics, models = [], []
    for k in range(1, k_max + 1):
        g = GaussianMixture(k, covariance_type="diag", n_init=5, random_state=seed, reg_covar=reg).fit(Z)
        bics.append(float(g.bic(Z))); models.append(g)
    K = int(min(k for k in range(1, k_max + 1) if bics[k - 1] <= min(bics) + tol))
    return K, models[K - 1].predict(Z)


def features(vf, X, system):
    cap = {}
    def grab(feat, k_max=5, curves=False, **kw):
        cap["feat"], cap["curves"] = feat, curves; return select_k(feat, k_max=k_max, curves=curves, **kw)
    orig = rsc.select_k; rsc.select_k = grab
    try:
        K, lab, _ = rsc.readout(vf, X, system)
    finally:
        rsc.select_k = orig
    return cap["feat"], cap["curves"], K, lab


def main():
    changes = [(p, v) for p, vs in GRID for v in vs if v != DEFAULT[p]]
    head = ["default"] + ["%s=%g" % c for c in changes]
    print("Class read-out hyperparameters, one change at a time from the default (n_pc 3, k_max 5, tol 10, reg 0.05)")
    print("cells: K / ARI\n%-14s %4s %s" % ("setting", "seed", " ".join("%-12s" % h for h in head)))
    allres = {}
    for lab, cells in B:
        name = PAPER_NAME[lab]
        if name not in SETTINGS:
            continue
        for ds, s in cells:
            m, d, vf, W = load_run(sel[sel_key(lab, ds, s)]["run"]); X = np.asarray(d["data"], np.float32); cls = d["classes"]
            feat, curves, K0, lab0 = features(vf, X, d["meta"]["system"])
            out = []
            for c in [None] + changes:
                p = dict(DEFAULT)
                if c:
                    p[c[0]] = c[1]
                K, l = select_k_p(feat, curves=curves, **p)
                if c is None:
                    assert K == K0 and adjusted_rand_score(l, lab0) == 1.0, (name, s)
                out.append((K, adjusted_rand_score(cls, l)))
            allres.setdefault(name, []).append(out)
            print("%-14s %4d %s" % (name, s, " ".join("%-12s" % ("%d / %.3f" % o) for o in out)))
    print("\nmean ARI over the three seeds (K per seed in brackets)")
    for k, h in enumerate(head):
        print("%-12s %s" % (h, "  ".join("%s %.3f [%s]" % (n, np.mean([r[k][1] for r in rows]), ",".join(str(r[k][0]) for r in rows))
                                        for n, rows in allres.items())))
    n = sum(len(r) for r in allres.values())
    for k, h in enumerate(head[1:], 1):
        ch = [(nm, i + 1) for nm, rows in allres.items() for i, r in enumerate(rows) if r[k][0] != r[0][0]]
        up = [(nm, i + 1) for nm, rows in allres.items() for i, r in enumerate(rows) if r[k][1] > r[0][1] + 1e-9]
        dn = [(nm, i + 1) for nm, rows in allres.items() for i, r in enumerate(rows) if r[k][1] < r[0][1] - 1e-9]
        print("%-12s K changes on %2d of %d runs; ARI up on %s; down on %s" % (
            h, len(ch), n, ", ".join("%s s%d" % x for x in up) or "none", ", ".join("%s s%d" % x for x in dn) or "none"))


if __name__ == "__main__":
    main()
