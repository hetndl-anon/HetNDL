"""NRI and AIDD (our re-implementations) in AIDD's own setting: the spring network of AIDD (Zhang et al., Phys. Rev. E
106, 034315, 2022, Table III, row ER-10: one fixed undirected Erdos-Renyi graph on 10 nodes, the same dynamics at every
node; published AUC: AIDD 1.0, NRI 0.9998). Data spring10_seed{1,2,3} (hetndl.gen.springs: edge probability 0.2, no
isolated node, 1,000 trajectories of 50 states, node state (r_x, r_y, v_x, v_y)).
Runs (experiments/reproduce/jobs.json): spring10/{nri,aidd}/<dataset> = our NRI and AIDD with the arguments of their
Table-2 MixRelax runs (600 epochs, hidden 64, window 50, 10 prediction steps, lr 5e-4, batch 32, tau 0.5; AIDD
lr_graph 1e-2), raw input, data replaced, seed = data seed; spring10/nri_official/<dataset>/seed<k> = the original NRI
code with the configuration of nri_official.py (experiments/launch/nri_original_run.py, mode bench).
Scores: AUROC / AUPRC against the true adjacency (truth only here): from metrics.json for our runs; for the original
code the mean probability of edge type 1 over the test windows, W[sender, receiver] (nri_official.official_W).
Run from the repository root with PYTHONPATH=. (-> results/supplement/spring_check.txt)."""
import json, os
import numpy as np
from experiments.analysis.nri_official import score as official_score
R = "experiments/runs/spring10"; SEEDS = (1, 2, 3)
PUB = {"AIDD": 1.0, "NRI": 0.9998}                     # AIDD paper, Table III, Spring ER-10


def run_score(m, ds):
    p = os.path.join(R, m, ds, "metrics.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p)); return b["auroc"], b["auprc"]


def original(ds, k):
    x = official_score(ds, k, os.path.join(R, "nri_official"))
    return None if x is None else x[:2]


def main():
    print("Spring network (one fixed undirected ER graph, N = 10, the same dynamics at every node, 1,000 trajectories"
          " of 50 states)\n")
    print("Spring ER-10 (published AUC: AIDD %.4f, NRI %.4f)" % (PUB["AIDD"], PUB["NRI"]))
    means = {}
    for lab, pub, f in (("our AIDD", "AIDD", lambda ds, k: run_score("aidd", ds)),
                        ("our NRI", "NRI", lambda ds, k: run_score("nri", ds)), ("original NRI code", "NRI", original)):
        r = [f("spring10_seed%d" % k, k) for k in SEEDS]
        if any(x is None for x in r):
            print("  %-24s missing runs" % lab); continue
        a = np.array(r); means[lab] = (a[:, 0].mean(), pub)
        print("  %-24s %s   mean AUROC %.3f AUPRC %.3f" % (lab, "  ".join("%.3f / %.3f" % tuple(x) for x in r), *a.mean(0)))
    if means:
        print("\nmean AUROC minus the published value: %s" % ", ".join("%s %+.3f" % (lab, m - PUB[p]) for lab, (m, p) in means.items()))


if __name__ == "__main__":
    main()
