"""Two-stage mechanism / parameter read-out baseline ("per-node basis identification -> cluster the fitted
functions"), from the ARNI runs already done (runs/arni/<dataset>/full_seed<k>/self_coef.npy: each node's self-basis
coefficients in the original scale — cubic polynomial basis, or 1 / sin / cos for Kuramoto phases).
  classes  : select_k on the per-node coefficient vectors (truth-free K, the protocol), ARI against the true classes;
             KMeans with the true K on standardised coefficients as the known-K reference column (as granger_mechanism.py)
  Kuramoto : |Spearman(constant coefficient, omega)| — ARNI's own node-constant read-out.
Truth only in the scores. Run from the project root with PYTHONPATH=."""
import glob, os, numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from hetndl.gen.kuramoto import load_dataset
from hetndl.readout import select_k
from experiments.analysis.paper_seeds import PAPER_SEEDS
R = "experiments/runs/arni"
CLS = [("A1", ["a1v2_N30_seed%d" % s for s in (1, 2, 3)]), ("A2 N30", ["a2_N30_seed%d" % s for s in PAPER_SEEDS]),
       ("A2 N60", ["a2_N60_seed%d" % s for s in (1, 2, 3)]), ("A3", ["a3_N30_seed%d" % s for s in (1, 2, 3)]),
       ("IEEE-118 grid", ["grid118_seed%d" % s for s in (1, 2, 3)]), ("BA A1", ["a1v2_BA30_m2_seed%d" % s for s in (1, 2, 3)])]
KUR = [(c, ["kura_ER100_K2_%s%s" % (c, "" if s == 1 else "_g%d" % s) for s in (1, 2, 3)]) for c in ("s0.3_n0", "s2_n0", "s0.3_n0.2", "s0.3_n0.5", "s0.6_n0", "s1_n0")]   # last two: Fig. 1 sigma sweep
KUR += [("BA s0.3_n0.5", ["kura_BA100_m2_K2_s0.3_n0.5_g%d" % s for s in (1, 2, 3)])]


def coef(ds):
    p = sorted(glob.glob(os.path.join(R, ds, "full_seed*", "self_coef.npy")))
    return np.load(p[0]) if p else None


print("== ARNI per-node coefficients -> classes (ARI)")
for lab, dss in CLS:
    out = []
    for ds in dss:
        C = coef(ds); d = load_dataset("experiments/data/%s.npz" % ds)
        if C is None or "classes" not in d:
            out.append("-"); continue
        c = d["classes"]; K, l, _ = select_k(C, k_max=5)
        Cs = (C - C.mean(0)) / (C.std(0) + 1e-12)
        ref = adjusted_rand_score(c, KMeans(len(np.unique(c)), n_init=20, random_state=0).fit_predict(Cs))
        out.append("K %d ARI %.3f (known-K %.3f)" % (K, adjusted_rand_score(c, l), ref))
    print("%-14s %s" % (lab, " | ".join(out)))
print("== ARNI constant coefficient vs omega (|Spearman|)")
for lab, dss in KUR:
    v = []
    for ds in dss:
        C = coef(ds); d = load_dataset("experiments/data/%s.npz" % ds)
        v.append("%.3f" % abs(spearmanr(C[:, 0], d["omega"]).correlation) if C is not None else "-")
    print("%-14s %s" % (lab, " ".join(v)))
