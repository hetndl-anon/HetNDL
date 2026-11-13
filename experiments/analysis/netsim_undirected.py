"""Undirected baselines on NetSim: partial correlation (PC) and graphical lasso (GL), the strong
methods of the original NetSim evaluation (Smith et al. 2011), on the same converted data and trajectory split as the
single-run baselines. Analysis only, no training of neural models, CPU only. Settings, scoring and the reading rule below
were fixed before any number of this script was computed.
Run from the project root:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=. python experiments/analysis/netsim_undirected.py \
      > results/supplement/netsim_undirected.txt

Data and split. experiments/data/netsim_sim<k>.npz, k = 1..28 without 13 (27 sims), data (50, T, N, 1) as converted
  (standardised per node over all subjects, see netsim_protocol.py). seed and n_traj are read from
  experiments/runs/granger/netsim_sim<k>/seed1/metrics.json args (seed 1; n_traj 16 for sims 5, 6, 4 for 7, 9, 8 for 19, 20,
  None otherwise), asserted equal for NRI, AIDD, COSINE, MTGNN, SGODE v1 / v3 (ARNI ran without the cap and is not used for
  the split). keep = hetndl.data.select_trajectories(50, n_traj, seed); tr, va, _ = hetndl.data.trajectory_split(len(keep),
  seed), as hetndl/train.py and every baseline. X_tr = all time points of the subjects keep[tr] pooled (rows = time points,
  columns = nodes, float64), X_va the same for keep[va]. No noise, no smoothing (as all NetSim runs). Test subjects unused.
Partial correlation. C = sklearn.covariance.empirical_covariance(X_tr) (centred on the pooled training mean, ddof 0),
  P = inv(C), rho_ij = -P_ij / sqrt(P_ii P_jj), edge score W = |rho| with a zero diagonal.
Graphical lasso. sklearn.covariance.GraphicalLasso(alpha, mode="cd", tol=1e-4, max_iter=1000, assume_centered=False) fit on
  X_tr for every alpha of the fixed grid np.logspace(-4, 0, 20); the alpha with the highest mean Gaussian log-likelihood of
  X_va under the fitted model (GraphicalLasso.score, centred on the training mean) is selected; ties -> the first (smallest)
  alpha. A fit that raises or returns a non-finite precision is dropped from the choice; a ConvergenceWarning alone does not
  drop it (both counts printed). A selected alpha at either end of the grid is flagged, not acted upon. Edge score W =
  |partial correlation of the selected precision|. Ground truth never enters either method.
Neither method gives a direction or an edge sign: W is symmetric and non-negative.
Scoring (ground truth A = the npz's directed adjacency, A[j, i] = 1 for j -> i; used for scoring only).
 (a) The paper's protocol: hetndl.metrics.auroc_auprc(W, A), ordered pairs (j, i), j != i; the symmetric score is assigned to
     both directions. For reference the AUROC ceiling of a symmetric score under (a) is printed per sim: with E ordered true
     edges of which E1 are one-way and Neg = N(N-1) - E, a perfect skeleton ranking gives 1 - E1^2 / (2 E Neg).
 (b) Skeleton: unordered pairs i < j, label (A + A^T > 0), score |W_ij| + |W_ji|, AUROC / AUPRC by sklearn roc_curve /
     precision_recall_curve / auc exactly as in hetndl.metrics. Methods: PC, GL; HetNDL = the three selected runs per sim of
     experiments/runs/sel/selected.json (reported snapshot W_best.npy), skeleton AUROC / AUPRC per run, then the mean over
     the three seeds (as the main table averages per-run scores); NOGI = netsim_rk4sub/netsim_sim<k>/orig_seed1..3 (reported
     snapshot), mean over the three runs, a sim used only when all three exist (as baseline_table.py); every single-run
     baseline with a saved W.npy, run directories as in baseline_table.py: ARNI (arni/<ds>/full_seed1, the only ARNI run on
     NetSim), Granger, NRI, AIDD (first run directory), COSINE (seed1; a failed run is missing), MTGNN (raw W.npy, before its
     symmetrised (a)-scoring), SGODE (variant picked by validation loss, experiments/analysis/sgode_table.pick). The same
     symmetrisation |W_ij| + |W_ji| for every method.
 Check: (a) recomputed from the same matrices (baseline_table.score_matrix; W_best.npy for HetNDL / NOGI) must reproduce the
     numbers baseline_table.py reports (max |difference| printed per method).
Statistics. Means over the sims with a result (n printed). Paired two-sided Wilcoxon signed-rank test over sims,
  scipy.stats.wilcoxon with its defaults (as baseline_table.py), HetNDL's per-sim seed mean vs each method, AUROC and AUPRC
  separately: under (b) against every method, under (a) against PC and GL. W/T/L with |AUROC difference| <= 1e-3 = tie
  (as the paper). No multiple-comparison correction; p-values printed as they are.
Reading rule.
 (a) PC / GL numbers go next to Table 2's NetSim row (HetNDL 0.890 mean AUROC). If the PC or GL mean AUROC is >= HetNDL's,
     HetNDL is not the best NetSim method under the paper's own protocol and the paper must say so. (a) caps any undirected
     score (ceiling column), so (a) alone does not measure what PC / GL recover.
 (b) is the comparison on what PC / GL can recover. HetNDL mean higher and p < 0.05: HetNDL recovers the skeleton better
     than the classical undirected method. HetNDL mean lower and p < 0.05: the classical method beats HetNDL on the
     skeleton, and HetNDL's NetSim advantage in Table 2 rests on direction (and on the protocol). p >= 0.05: no significant
     difference. Same reading for HetNDL vs every other baseline under (b)."""
import glob
import json
import os
import warnings

import numpy as np
from scipy.stats import wilcoxon
from sklearn.covariance import GraphicalLasso, empirical_covariance
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from experiments.analysis.baseline_table import met, score, score_matrix, sel as SEL
from experiments.analysis.sgode_table import pick as sgode_pick
from hetndl.data import select_trajectories, trajectory_split
from hetndl.gen.kuramoto import load_dataset
from hetndl.metrics import auroc_auprc
from hetndl.snapshot import array_path, metrics_of

R = "experiments/runs"
SIMS = [k for k in range(1, 29) if k != 13]
ALPHAS = np.logspace(-4, 0, 20)
SINGLE = ("ARNI", "Granger", "NRI", "AIDD", "COSINE", "MTGNN", "SGODE")


def skeleton(W, A):
    iu = np.triu_indices(len(A), 1)
    y = ((A + A.T) > 0)[iu].astype(int)
    s = (np.abs(W) + np.abs(W.T))[iu].astype(np.float64)
    fpr, tpr, _ = roc_curve(y, s)
    p, r, _ = precision_recall_curve(y, s)
    return float(auc(fpr, tpr)), float(auc(r, p))


def ceiling(A):
    off = ~np.eye(len(A), dtype=bool)
    P = (A > 0) & off
    E = int(P.sum()); E1 = int((P & ~P.T).sum()); neg = int(off.sum()) - E
    return 1.0 - E1 ** 2 / (2.0 * E * neg)


def pcor(P):
    d = np.sqrt(np.diag(P))
    Rm = -P / np.outer(d, d)
    np.fill_diagonal(Rm, 0.0)
    return np.abs(Rm)


def split_of(k):
    ds = "netsim_sim%d" % k
    a = json.load(open(os.path.join(R, "granger", ds, "seed1", "metrics.json")))["args"]
    seed, nt = int(a["seed"]), a["n_traj"]
    for d in ("nri/%s/seed1", "aidd/%s/seed1", "cosine/%s/seed1", "mtgnn/%s/seed1", "sgode/v1/%s/seed1", "sgode/v3/%s/seed1"):
        b = json.load(open(os.path.join(R, d % ds, "metrics.json")))["args"]
        assert (int(b["seed"]), b["n_traj"]) == (seed, nt), (d % ds, b["seed"], b["n_traj"], seed, nt)
    return seed, nt


def fit_gl(Xtr, Xva):
    best, dropped, warned = None, 0, 0
    for a in ALPHAS:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                g = GraphicalLasso(alpha=float(a), mode="cd", tol=1e-4, max_iter=1000, assume_centered=False).fit(Xtr)
            warned += any(issubclass(x.category, ConvergenceWarning) for x in w)
            P = g.precision_
            if not np.all(np.isfinite(P)):
                dropped += 1; continue
            ll = float(g.score(Xva))
            if not np.isfinite(ll):
                dropped += 1; continue
        except Exception:
            dropped += 1; continue
        if best is None or ll > best[0]:
            best = (ll, float(a), P)
    return best, dropped, warned


def base_dir(name, k):
    """Run directory (relative to experiments/runs) of a single-run baseline on sim k, None if missing or failed."""
    ds = "netsim_sim%d" % k
    if name == "SGODE":
        p = sgode_pick(ds, 1)
        return p[1] if p else None
    if name == "COSINE":
        return "cosine/%s/seed1" % ds if met("cosine/%s/seed1" % ds) else None
    sub = {"ARNI": "arni", "Granger": "granger", "NRI": "nri", "AIDD": "aidd", "MTGNN": "mtgnn"}[name]
    ps = sorted(glob.glob(os.path.join(R, sub, ds, "*", "metrics.json")))
    if name == "ARNI":
        assert len(ps) <= 1, ps
    if not ps:
        return None
    rel = os.path.relpath(os.path.dirname(ps[0]), R)
    return rel if met(rel) else None


def wx(x, y):
    try:
        return float(wilcoxon(x, y).pvalue)
    except ValueError:
        return float("nan")


def main():
    res = {}                       # method -> {sim: ((a_auroc, a_auprc) or None, (b_auroc, b_auprc))}
    check = {}                     # method -> max |(a) recomputed - reported|
    diag = []
    note = lambda m, d: check.__setitem__(m, max(check.get(m, 0.0), d))
    print("NetSim, 27 simulations: partial correlation (PC) and graphical lasso (GL) on the pooled TRAINING subjects; "
          "GL alpha by the Gaussian log-likelihood of the pooled VALIDATION subjects.")
    print("PC and GL give NO edge direction and NO edge sign (symmetric, non-negative score).")
    print("(a) = paper protocol, ordered pairs (j, i) vs the directed adjacency, symmetric score on both directions; "
          "(b) = unordered pairs vs the skeleton A + A^T > 0, score |W_ij| + |W_ji|.")
    print("ceil(a) = AUROC ceiling of a perfect undirected score under (a). HetNDL = mean over its three selected runs.\n")
    print("%-6s %3s %3s %6s %-7s %-9s %-15s %-15s %-15s %7s | %-15s %-15s %-15s" % (
        "sim", "N", "E", "n_traj", "tr/va", "GL alpha", "PC (a)", "GL (a)", "HetNDL (a)", "ceil(a)", "PC (b)", "GL (b)", "HetNDL (b)"))
    f = lambda x: "%.3f / %.3f" % x
    for k in SIMS:
        ds = "netsim_sim%d" % k
        d = load_dataset("experiments/data/%s.npz" % ds)
        A = np.asarray(d["A"], np.float64); X = np.asarray(d["data"][..., 0], np.float64)
        S, T, N = X.shape
        seed, nt = split_of(k)
        keep = select_trajectories(S, nt, seed)
        tr, va, _ = trajectory_split(len(keep), seed)
        Xtr = X[keep[tr]].reshape(-1, N); Xva = X[keep[va]].reshape(-1, N)
        W_pc = pcor(np.linalg.inv(empirical_covariance(Xtr)))
        best, dropped, warned = fit_gl(Xtr, Xva)
        W_gl = pcor(best[2])
        edge = " (grid edge)" if best[1] in (float(ALPHAS[0]), float(ALPHAS[-1])) else ""
        diag.append((k, best[1], dropped, warned, edge))
        res.setdefault("PC", {})[k] = (auroc_auprc(W_pc, A), skeleton(W_pc, A))
        res.setdefault("GL", {})[k] = (auroc_auprc(W_gl, A), skeleton(W_gl, A))
        # HetNDL: the three selected runs
        a_, b_ = [], []
        for s in (1, 2, 3):
            e = SEL["%s/add_z_signed_diff_lz0_seed%d" % (ds, s)]
            m = json.load(open(os.path.join(e["run"], "metrics.json")))
            W = np.load(array_path(e["run"], "W", m))
            x = auroc_auprc(W, A); note("HetNDL", max(abs(x[0] - e["auroc"]), abs(x[1] - e["auprc"])))
            a_.append((e["auroc"], e["auprc"])); b_.append(skeleton(W, A))
        res.setdefault("HetNDL", {})[k] = (tuple(np.mean(a_, 0)), tuple(np.mean(b_, 0)))
        # NOGI: three runs, all required
        ms = [(p, met(p)) for p in ("netsim_rk4sub/%s/orig_seed%d" % (ds, s) for s in (1, 2, 3))]
        if all(m is not None for _, m in ms):
            a_, b_ = [], []
            for p, m in ms:
                W = np.load(array_path(os.path.join(R, p), "W", m))
                x = auroc_auprc(W, A); y = metrics_of(m); note("NOGI", max(abs(x[0] - y["auroc"]), abs(x[1] - y["auprc"])))
                a_.append((y["auroc"], y["auprc"])); b_.append(skeleton(W, A))
            res.setdefault("NOGI", {})[k] = (tuple(np.mean(a_, 0)), tuple(np.mean(b_, 0)))
        for name in SINGLE:
            p = base_dir(name, k)
            if p is None:
                continue
            Wraw = np.load(os.path.join(R, p, "W.npy"))
            rep = score(met(p)); x = auroc_auprc(score_matrix(p), A)
            note(name, max(abs(x[0] - rep[0]), abs(x[1] - rep[1])))
            res.setdefault(name, {})[k] = (rep, skeleton(Wraw, A))
        print("%-6s %3d %3d %6s %-7s %-9.2e %-15s %-15s %-15s %7.3f | %-15s %-15s %-15s" % (
            "sim%d" % k, N, int(A.sum()), str(nt), "%d/%d" % (len(tr), len(va)), best[1],
            f(res["PC"][k][0]), f(res["GL"][k][0]), f(res["HetNDL"][k][0]), ceiling(A),
            f(res["PC"][k][1]), f(res["GL"][k][1]), f(res["HetNDL"][k][1])))

    print("\nGL fits: dropped (raised / non-finite) %d of %d; fits with a ConvergenceWarning %d; selected alpha at a grid edge: %s"
          % (sum(x[2] for x in diag), len(diag) * len(ALPHAS), sum(x[3] for x in diag),
             ", ".join("sim%d" % x[0] for x in diag if x[4]) or "none"))
    print("selected GL alpha: min %.2e, median %.2e, max %.2e" % (
        min(x[1] for x in diag), float(np.median([x[1] for x in diag])), max(x[1] for x in diag)))
    print("check, (a) recomputed from the saved matrices vs the reported numbers, max |diff|: %s" % (
        ", ".join("%s %.1e" % (m, v) for m, v in check.items())))
    print("ceil(a) mean over the 27 sims: %.3f" % np.mean([ceiling(np.asarray(load_dataset("experiments/data/netsim_sim%d.npz" % k)["A"], np.float64)) for k in SIMS]))

    H = res["HetNDL"]

    def table(idx, methods, title):
        print("\n" + title)
        print("%-8s %3s  %-15s  %-9s %-11s %-11s" % ("method", "n", "AUROC / AUPRC", "W/T/L", "p AUROC", "p AUPRC"))
        h = np.array([H[k][idx] for k in SIMS])
        print("%-8s %3d  %-15s" % ("HetNDL", len(SIMS), f(tuple(h.mean(0)))))
        for name in methods:
            ks = [k for k in SIMS if k in res.get(name, {}) and res[name][k][idx] is not None]
            if not ks:
                print("%-8s no runs" % name); continue
            x = np.array([H[k][idx] for k in ks]); y = np.array([res[name][k][idx] for k in ks]); dd = x[:, 0] - y[:, 0]
            print("%-8s %3d  %-15s  %d/%d/%-5d %-11.3g %-11.3g" % (
                name, len(ks), f(tuple(y.mean(0))), (dd > 1e-3).sum(), (abs(dd) <= 1e-3).sum(), (dd < -1e-3).sum(),
                wx(x[:, 0], y[:, 0]), wx(x[:, 1], y[:, 1])))

    print("\nW/T/L = sims on which HetNDL wins / ties (|AUROC diff| <= 1e-3) / loses; p = paired two-sided Wilcoxon signed-rank over sims, "
          "HetNDL vs the method")
    table(0, ("PC", "GL"), "(a) paper protocol (ordered pairs, directed adjacency); PC / GL get the same score on both directions")
    table(1, ("PC", "GL", "NOGI") + SINGLE, "(b) skeleton (unordered pairs, A + A^T > 0), score |W_ij| + |W_ji| for every method")


if __name__ == "__main__":
    main()
