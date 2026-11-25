"""NRI and AIDD (our re-implementations) on the five-oscillator Kuramoto system of the NRI paper with one fixed graph.
Data kura5_nri_seed{1,2,3} (Kipf et al. 2018, App. B.3, with one graph per dataset): N = 5, undirected edges with
probability 0.5 (graph seed = data seed), k_ij = 1 without in-degree normalisation, omega_i ~ U[1, 10), initial phases
~ U[0, 2 pi), all edges excitatory, states every 0.1, 10,000 trajectories of 50 states. Runs
kuramoto5/{nri,aidd}/{raw,trig}/<dataset> of experiments/reproduce/jobs.json: the arguments of the Table-2 Kuramoto runs
(600 epochs, hidden 64, window 50, 10 prediction steps, lr 5e-4, tau 0.5; NRI batch 32 and two edge types; AIDD batch 8
and lr_graph 1e-2), seed = data seed, with two inputs: "raw" = the input of Table 2 (the observed, unwrapped phases) and
"trig" = (cos theta, sin theta), the phase rule that HetNDL, NOGI and MTGNN use on Kuramoto. NRI's own input
(d theta / dt, sin theta, omega) is not used: omega_i is a node parameter that the read-outs are scored on.
Score = AUROC (and AUPRC) of the run's edge scores against the true graph, from its metrics.json; truth enters only
there. kuramoto5_recovery.py gives HetNDL, ARNI and Granger on the same data.
Run from the repository root with PYTHONPATH=. (-> results/supplement/positive_control.txt)."""
import json, os
import numpy as np
from hetndl.gen.kuramoto import load_dataset
R = "experiments/runs/kuramoto5"; SEEDS = (1, 2, 3); DS = "kura5_nri_seed%d"


def run(method, arm, s):
    p = os.path.join(R, method, arm, DS % s, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return None if m.get("failed") else m


def means():
    """{(method, arm): mean AUROC over the three seeds, or None if a run is missing}."""
    out = {}
    for method in ("nri", "aidd"):
        for arm in ("raw", "trig"):
            au = [m["auroc"] for m in (run(method, arm, s) for s in SEEDS) if m is not None]
            out[(method, arm)] = float(np.mean(au)) if len(au) == len(SEEDS) else None
    return out


def main():
    print("NRI and AIDD on the 5-oscillator Kuramoto of the NRI paper, one fixed graph per dataset (AUROC / AUPRC per data seed)")
    for s in SEEDS:
        A = load_dataset("experiments/data/%s.npz" % (DS % s))["A"]
        print("  %s: %d undirected edges of 10, edge density %.2f" % (DS % s, int(A.sum()) // 2, A.sum() / 20.0))
    mean = means()
    for method in ("nri", "aidd"):
        for arm in ("raw", "trig"):
            ms = [run(method, arm, s) for s in SEEDS]
            cells = ["   missing   " if m is None else "%.3f / %.3f" % (m["auroc"], m["auprc"]) for m in ms]
            print("%-5s %-4s  %s   mean AUROC %s   best epochs %s" % (
                method.upper(), arm, "  ".join(cells), "missing" if mean[(method, arm)] is None else "%.3f" % mean[(method, arm)],
                " ".join("-" if m is None else str(m["best_epoch"]) for m in ms)))
    if all(v is not None for v in mean.values()):
        b = {m: max(mean[(m, a)] for a in ("raw", "trig")) for m in ("nri", "aidd")}
        print("\nbest mean AUROC over the two inputs: NRI %.3f, AIDD %.3f" % (b["nri"], b["aidd"]))


if __name__ == "__main__":
    main()
