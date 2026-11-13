"""Is the fixed graph of the five-oscillator Kuramoto data (kura5_nri_seed{1,2,3}, see positive_control.py) recoverable
from these trajectories? If no method recovers it, the failure of NRI and AIDD there says little about their
implementations; if some method does, it is a failure of those methods with one fixed graph.
  * HetNDL: the two candidates with the arguments of the main-table members of kura_ER100_K2_s0.3_n0 data seed k
    (Kuramoto-ω, σ/κ 0.15; runs/sel/manifest.json), data replaced by kura5_nri_seed<k>, seed k (runs
    kuramoto5/hetndl/<dataset>/{nonneg,nowu} of experiments/reproduce/jobs.json); the candidate with the lower
    validation error (val_mse_best) is read (the selection rule of the paper), its reported snapshot
    (hetndl.snapshot.metrics_of).
  * ARNI (full basis and greedy) and Granger: the arguments of their Table-2 runs on kura_ER100_K2_s0.3_n0 seed 1, data
    replaced, seed k (kuramoto5/{arni_full,arni_greedy,granger}/<dataset>).
  * score: graph AUROC / AUPRC against the true adjacency (computed by each run; truth only there).
For reference, the means of NRI and AIDD (positive_control.py) and of the original NRI code (nri_regime.py) on the same
data are printed from their runs.
Run from the repository root with PYTHONPATH=. (-> results/supplement/kuramoto5_recovery.txt)."""
import json, os
import numpy as np
from hetndl.snapshot import metrics_of
from experiments.analysis import nri_regime, positive_control
R = "experiments/runs/kuramoto5"; DS = "kura5_nri_seed%d"; SEEDS = (1, 2, 3)


def hetndl(ds):
    c = []
    for member in ("nonneg", "nowu"):
        p = os.path.join(R, "hetndl", ds, member, "metrics.json")
        if not os.path.exists(p):
            return None
        m = json.load(open(p))
        if not m.get("failed"):
            c.append((m["val_mse_best"], member, m))
    if not c:
        return None
    v, member, m = min(c, key=lambda x: x[0]); b = metrics_of(m)
    return b["auroc"], b["auprc"], member


def base(m, ds):
    p = os.path.join(R, m, ds, "metrics.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p)); return b["auroc"], b["auprc"], ""


def main():
    print("Graph recovery on the 5-oscillator Kuramoto of the NRI paper (N = 5, one fixed graph per seed, 10,000 trajectories)\n")
    rows = [("HetNDL (selected candidate)", hetndl), ("ARNI full basis", lambda d: base("arni_full", d)),
            ("ARNI greedy", lambda d: base("arni_greedy", d)), ("Granger", lambda d: base("granger", d))]
    for name, f in rows:
        r = [f(DS % k) for k in SEEDS]
        if any(x is None for x in r):
            print("%-30s missing runs" % name); continue
        a = np.array([x[:2] for x in r])
        print("%-30s %s   mean AUROC %.3f AUPRC %.3f" % (name, "  ".join("%.3f / %.3f%s" % (x[0], x[1], (" (%s)" % x[2]) if x[2] else "")
                                                                         for x in r), *a.mean(0)))
    pc = positive_control.means(); off = [nri_regime.official_scores(r) for r in nri_regime.fixed_graph_runs()]
    if all(v is not None for v in pc.values()) and all(x is not None for x in off):
        print("\nreference on the same data: NRI raw %.3f, trig %.3f; AIDD raw %.3f, trig %.3f (positive_control.py);"
              " original NRI code %.3f (nri_regime.py)" % (pc[("nri", "raw")], pc[("nri", "trig")], pc[("aidd", "raw")],
                                                          pc[("aidd", "trig")], np.mean([x[0] for x in off])))


if __name__ == "__main__":
    main()
