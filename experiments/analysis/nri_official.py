"""The original NRI code (experiments/launch/nri_original_run.py, mode bench) on the benchmark cells of Table 2 with
N <= 60 and on the 27 NetSim simulations (runs nri_official/<dataset>/seed<k> of experiments/reproduce/jobs.json).
Kuramoto (N = 100) and Grid-118 were not run: one run there takes about 35 GB of GPU memory and 5-9 h on an A100.
Per cell: the exp*/ folder of the run that holds test_prob.npy (the attempt that fitted into memory; batch.txt); the
probability of edge type 1 averaged over the test windows; edge e of the code has receiver where(off_diag)[0][e] and
sender where(off_diag)[1][e], so W[sender, receiver] = that mean; AUROC / AUPRC with hetndl.metrics.auroc_auprc against
the true adjacency (truth only here).
Output: per Table-2 setting the mean AUROC / AUPRC of the original code next to our re-implementation (the NRI column
of Table 2, baseline_table.one("nri", ds); on NetSim baseline_table.netsim_mean("NRI", k), the mean over training seeds
1-3); NetSim: the means, W/T/L of HetNDL against the original code (seed 1) and the paired Wilcoxon test (HetNDL per
simulation averaged over its three training seeds, as in Table 2).
Run from the repository root with PYTHONPATH=. (-> results/supplement/nri_official.txt)."""
import glob, os, re
import numpy as np
from scipy.stats import wilcoxon
from hetndl.gen.kuramoto import load_dataset
from hetndl.metrics import auroc_auprc
from experiments.analysis.baseline_table import B, PAPER_NAME, netsim_mean, one, sel
R = "experiments/runs/nri_official"


def official_W(run_dir):
    """Edge-score matrix W[sender, receiver] of a finished run of the original code and its batch size."""
    ps = glob.glob(os.path.join(run_dir, "exp*", "test_prob.npy"))
    if not os.path.exists(os.path.join(run_dir, "done")) or len(ps) != 1:
        return None, None
    P = np.load(ps[0])[..., 1].mean(0)                                   # (E,)
    N = int(round((1 + np.sqrt(1 + 4 * len(P))) / 2)); off = np.ones((N, N)) - np.eye(N)
    rec, snd = np.where(off); W = np.zeros((N, N)); W[snd, rec] = P
    b = open(os.path.join(run_dir, "batch.txt")).read().strip() if os.path.exists(os.path.join(run_dir, "batch.txt")) else "?"
    return W, b


def score(ds, k, root=R):
    W, b = official_W(os.path.join(root, ds, "seed%d" % k))
    if W is None:
        return None
    return auroc_auprc(W, load_dataset("experiments/data/%s.npz" % ds)["A"]) + (b,)


def main():
    print("Original NRI code (CNN encoder, skip-first) on the Table-2 cells with N <= 60 and on NetSim\n")
    print("%-22s %-28s %-28s %s" % ("setting", "original NRI AUROC / AUPRC", "our NRI (Table 2)", "per seed (original; batch)"))
    for lab, cells in B:
        off = [score(ds, s) for ds, s in cells]
        cur = [one("nri", ds) for ds, s in cells]
        name = PAPER_NAME[lab]
        if any(x is None for x in off):
            print("%-22s %-28s %.3f / %.3f" % (name, "not run" if all(x is None for x in off) else "missing runs", *np.mean(cur, 0)))
            continue
        o = np.array([x[:2] for x in off])
        print("%-22s %.3f / %.3f                %.3f / %.3f                %s" % (name, *o.mean(0), *np.mean(cur, 0),
                                                                           "  ".join("%.3f(b%s)" % (x[0], x[2]) for x in off)))
    ours = {}
    for g, s in sel.items():
        mm = re.match(r"netsim_sim(\d+)/.*_seed(\d+)$", g)
        if mm:
            ours.setdefault(int(mm.group(1)), []).append((s["auroc"], s["auprc"]))
    ks = [k for k in sorted(ours) if len(ours[k]) == 3]
    off = {k: score("netsim_sim%d" % k, 1) for k in ks}
    done = [k for k in ks if off[k] is not None]
    print("\nNetSim: %d of %d simulations done" % (len(done), len(ks)))
    if done:
        x = np.array([np.mean(ours[k], 0) for k in done]); y = np.array([off[k][:2] for k in done])
        cur = np.array([netsim_mean("NRI", k)[0] for k in done])
        d = x[:, 0] - y[:, 0]
        print("  HetNDL %.3f / %.3f   original NRI %.3f / %.3f   our NRI %.3f / %.3f   W/T/L %d/%d/%d   Wilcoxon p AUROC %.2g" % (
            *x.mean(0), *y.mean(0), *cur.mean(0), (d > 1e-3).sum(), (abs(d) <= 1e-3).sum(), (d < -1e-3).sum(),
            wilcoxon(x[:, 0], y[:, 0]).pvalue if len(done) > 5 else float("nan")))


if __name__ == "__main__":
    main()
