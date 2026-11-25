"""The original NRI code (experiments/launch/nri_original_run.py) in the regime of the NRI paper and on one fixed graph.
  * regime: NRI's own Kuramoto setting, a new random graph for every simulation, 50,000 training simulations (run
    nri_regime/seed1 of experiments/reproduce/jobs.json); the NRI paper reports 96.0 % edge accuracy for it.
  * fixed graph: the same configuration on kura5_nri_seed{1,2,3} (one graph per dataset, 10,000 trajectories; runs
    kuramoto5/nri_official/<dataset>/seed<k>), with omega among the inputs as in the paper.
Scores from the test_prob.npy / test_rel.npy that the patched code saves (exp*/ of the run): the probability of edge
type 1 (type 0 = no interaction under --skip-first) of every test window against the true graph of its simulation;
pooled AUROC over (window, ordered pair), and edge accuracy = argmax of the two types. Truth only in these scores.
Run from the repository root with PYTHONPATH=. (-> results/supplement/nri_regime.txt)."""
import glob, os
import numpy as np
from sklearn.metrics import roc_auc_score
R = "experiments/runs"; SEEDS = (1, 2, 3)


def official_scores(run):
    """(pooled AUROC, edge accuracy) of a finished run of the original code, None if it is not finished."""
    ps = glob.glob(os.path.join(R, run, "exp*", "test_prob.npy"))
    if len(ps) != 1 or not os.path.exists(os.path.join(R, run, "done")):
        return None
    P = np.load(ps[0]); rel = np.load(ps[0].replace("test_prob", "test_rel"))
    return float(roc_auc_score(rel.ravel(), P[..., 1].ravel())), float((P.argmax(-1) == rel).mean())


def fixed_graph_runs():
    return ["kuramoto5/nri_official/kura5_nri_seed%d/seed%d" % (k, k) for k in SEEDS]


def main():
    print("Original NRI code: AUROC / edge accuracy per seed\n")
    rows = [("NRI's regime: one random graph per simulation, 50,000 training simulations", ["nri_regime/seed1"]),
            ("one fixed graph per dataset (kura5_nri_seed1-3), 6,000 training trajectories", fixed_graph_runs())]
    for what, runs in rows:
        v = [official_scores(r) for r in runs]
        done = [x for x in v if x is not None]
        mean = "AUROC %.3f acc %.3f" % tuple(np.mean(done, 0)) if len(done) == len(v) else "-"
        print("%-78s %s   mean %s" % (what, "  ".join("    missing    " if x is None else "%.3f / %.3f" % x for x in v), mean))


if __name__ == "__main__":
    main()
