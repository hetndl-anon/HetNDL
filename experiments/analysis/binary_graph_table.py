"""Binary graphs from the truth-free 2-means cut: for every main-table setting and data seed 1-3, the edge
set hetndl.readout.binarize_2means(|W|) of our reported run (experiments/runs/sel/selected.json) and of every baseline's
score matrix (the runs of baseline_table.py through its run_dirs / score_matrix, i.e. the matrix behind each reported
AUROC, for MTGNN the symmetrised W + W^T; ARNI's full and greedy modes separately, never the better of the two).
Printed: precision, recall, F1, predicted and true edge counts per seed and their means.
Settings with inhibitory edges: the three-way sign table of the methods that output signs, predicted {+, -, no edge}
against true {+, -, no edge} over all ordered node pairs, summed over the seeds. Predicted sign of an edge of the cut:
ours = sign of W after the model's own sign convention (the gauge stored with the reported snapshot, the one behind the
reported edge-sign accuracy); ARNI = its Jacobian sign where the run saved one (W_jac.npy with a finite sign_acc_jac, as
in baseline_table.py), else its coefficient sign. Ground truth (A, S) enters only the scores.
Run from the project root with PYTHONPATH=."""
import json, os, numpy as np
from hetndl.readout import binarize_2means
from hetndl.snapshot import metrics_of
from experiments.analysis.baseline_table import B, PAPER_NAME, PER_SEED, R, met, run_dirs, score_matrix, sel, sel_key
from hetndl.gen.kuramoto import load_dataset
METHODS = ("ours",) + PER_SEED


def prf(Bn, A):
    off = ~np.eye(len(A), dtype=bool); tp = int((Bn & A)[off].sum()); n_pred, n_true = int(Bn[off].sum()), int(A[off].sum())
    P, Rc = tp / max(n_pred, 1), tp / max(n_true, 1)
    return P, Rc, 2 * P * Rc / max(P + Rc, 1e-12), n_pred, n_true, n_pred - tp


def confusion(Bn, Wsigned, S):
    """Rows: predicted +, -, no edge; columns: true +, -, no edge (off-diagonal ordered pairs)."""
    off = ~np.eye(len(S), dtype=bool)
    pred = np.where(Bn, np.where(Wsigned > 0, 0, 1), 2)[off]; true = np.where(S > 0, 0, np.where(S < 0, 1, 2))[off]
    return np.array([[int(((pred == a) & (true == b)).sum()) for b in (0, 1, 2)] for a in (0, 1, 2)])


def signed_scores(method, path):
    """(signed score matrix, source of the sign) for the methods that output edge signs, else None."""
    m = met(path); p = os.path.join(R, path)
    if method == "ours":
        g = metrics_of(m).get("gauge", 1)
        return score_matrix(path) * (g if isinstance(g, int) else 1), "gauge %+d" % g
    if method.startswith("ARNI"):
        j = m.get("sign_acc_jac")
        if j is not None and np.isfinite(j) and os.path.exists(os.path.join(p, "W_jac.npy")):
            return np.load(os.path.join(p, "W_jac.npy")), "Jacobian sign"
        return np.load(os.path.join(p, "W.npy")), "coefficient sign"
    return None


def stored_sign(method, path, s):
    """Sign accuracy the tables report for this run: selected.json for ours, baseline_table.py's rule for ARNI."""
    if method == "ours":
        return s["sign"]
    m = met(path); j = m.get("sign_acc_jac")
    return j if j is not None and np.isfinite(j) else m.get("sign_acc")


def main():
    print("2-means cut on |W| (hetndl.readout.binarize_2means), data seeds 1-3 of the twelve main settings")
    print("%-20s %-12s %4s  %9s %7s %7s  %5s %5s %5s" % ("setting", "method", "seed", "precision", "recall", "F1", "pred", "true", "false"))
    means, conf, exact = {}, {}, {}
    for lab, cells in B:
        name = PAPER_NAME[lab]; rows = {k: [] for k in METHODS}
        for ds, s in cells:
            g = sel_key(lab, ds, s)
            if g not in sel:
                continue
            d = load_dataset("experiments/data/%s.npz" % ds); A = d["A"] > 0; S = d["S"]
            runs = dict(run_dirs(lab, ds, s), ours=os.path.relpath(sel[g]["run"], R))
            for k in METHODS:
                if runs[k] is None:
                    continue
                Bn = binarize_2means(score_matrix(runs[k])); r = prf(Bn, A); rows[k].append(r)
                print("%-20s %-12s %4d  %9.3f %7.3f %7.3f  %5d %5d %5d" % ((name, k, s) + r))
                sg = signed_scores(k, runs[k]) if (S < 0).any() else None
                if sg is not None:
                    c = conf.setdefault((name, k), dict(M=np.zeros((3, 3), int), src=set(), acc=[], ok=[]))
                    c["M"] += confusion(Bn, sg[0], S); c["src"].add(sg[1])
                    c["acc"].append(float((np.sign(sg[0][S != 0]) == np.sign(S[S != 0])).mean()))
                    c["ok"].append(abs(c["acc"][-1] - stored_sign(k, runs[k], sel[g])) < 1e-9)
        for k in METHODS:
            if rows[k]:
                v = np.array(rows[k], float); means[(name, k)] = (len(v),) + tuple(v[:, :3].mean(0)) + (int(v[:, 5].sum()),)
                exact[(name, k)] = bool((v[:, 5] == 0).all() and (v[:, 3] > 0).all())
    print("\nmeans over the data seeds (false = false-positive edges summed over the seeds)")
    print("%-20s %-12s %4s  %9s %7s %7s  %5s" % ("setting", "method", "n", "precision", "recall", "F1", "false"))
    for (name, k), v in means.items():
        print("%-20s %-12s %4d  %9.3f %7.3f %7.3f  %5d" % ((name, k) + v))
    ours = [(n, v) for (n, k), v in means.items() if k == "ours"]
    print("\nours: mean precision exactly 1 (no false-positive edge on any seed) in %d of %d settings: %s" % (
        sum(exact[(n, "ours")] for n, _ in ours), len(ours), ", ".join(n for n, _ in ours if exact[(n, "ours")])))
    print("ours: mean precision below 1 in: %s" % ", ".join("%s (%.3f, %d false edges)" % (n, v[1], v[4]) for n, v in ours if not exact[(n, "ours")]))
    print("\nsign table on the 2-means graph, summed over the seeds: rows predicted (+, -, no edge), columns true (+, -, no edge)")
    for (name, k), c in conf.items():
        M = c["M"]; kept = M[:2, :2].sum()
        print("%-20s %-12s [%s]  sign accuracy on all true edges per seed: %s (%s)" % (
            name, k, ", ".join(sorted(c["src"])), " ".join("%.3f" % a for a in c["acc"]),
            "equal to the stored sign accuracy of the run" if all(c["ok"]) else "DIFFERS from the stored sign accuracy of the run"))
        for lab_, row in zip(("pred +", "pred -", "no edge"), M):
            print("    %-8s %6d %6d %6d" % ((lab_,) + tuple(row)))
        print("    recovered true edges %d of %d, of which sign correct %d (%.3f); false edges %d" % (
            kept, M[:, :2].sum(), M[0, 0] + M[1, 1], (M[0, 0] + M[1, 1]) / max(kept, 1), M[:2, 2].sum()))


if __name__ == "__main__":
    main()
