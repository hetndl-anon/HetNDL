"""SGODE baseline read-out (the runs sgode/<variant>/<dataset>/seed<k> of experiments/reproduce/jobs.json; method settings
in hetndl/baselines/sgode.py). Rule fixed before any run was scored: per benchmark cell the reported run is the variant
(v1 = free matrix, v3 = difference of two embedding products) with the lower validation loss among the variants that
did not fail; no ground truth enters the choice. pick(ds, s) is the single code path for that choice and is used by
baseline_table.py for the main table.
Output: per cell both variants (AUROC / AUPRC / sign accuracy raw and under the better global sign / validation loss
against the loss of repeating the first snapshot / best epoch) and the pick; per-setting means of the pick and of each
variant; NetSim means; how many runs forecast better than repeating the first snapshot.
NetSim: every simulation has training seeds 1-3; the pick is made per seed; the NetSim MEAN lines average each simulation
over its seeds (failed runs left out) and then over the simulations (n = simulations), as Table 2 does for every method.
Run from the project root: PYTHONPATH=. python experiments/analysis/sgode_table.py > results/supplement/sgode.txt"""
import json, os, re
import numpy as np
R = "experiments/runs"; VARIANTS = ("v1", "v3"); JOBS = "experiments/reproduce/jobs.json"; NETSIM_SEEDS = (1, 2, 3)


def met(variant, ds, s):
    p = os.path.join(R, "sgode", variant, ds, "seed%d" % s, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return None if m.get("failed") else m


def job_cells():
    """(dataset, data seed) of every SGODE cell, in the order of the job list."""
    out = []
    for j in json.load(open(JOBS)):
        m = re.match(r"sgode/(?:%s)/([^/]+)/seed(\d+)$" % "|".join(VARIANTS), j["name"])
        if m and (m.group(1), int(m.group(2))) not in out:
            out.append((m.group(1), int(m.group(2))))
    return out


def pick(ds, s):
    """(variant, run directory relative to experiments/runs, metrics) of the reported SGODE run, or None."""
    c = [(m["val_l1_best"], v, m) for v in VARIANTS for m in [met(v, ds, s)] if m is not None]
    if not c:
        return None
    _, v, m = min(c, key=lambda x: x[0])
    return v, os.path.join("sgode", v, ds, "seed%d" % s), m


def main():
    from experiments.analysis.baseline_table import B, PAPER_NAME
    CELLS = job_cells()
    f = lambda m: "      missing / failed      " if m is None else "%.3f / %.3f sign %.3f (flip %.3f) val %.4f vs %.4f ep %4d" % (
        m["auroc"], m["auprc"], m["sign_acc"], m["sign_acc_flip"], m["val_l1_best"], m["val_l1_persistence"], m["best_epoch"])
    print("SGODE (our re-implementation): AUROC / AUPRC, sign accuracy on true edges (raw; flip = better of the two global signs), "
          "12-step validation L1 vs repeating the first snapshot, best epoch; pick = variant with the lower validation loss")
    below = total = 0
    for lab, cells in B:
        rows = []
        for ds, s in cells:
            ms = {v: met(v, ds, s) for v in VARIANTS}; p = pick(ds, s)
            print("%-20s seed %d  v1 %s | v3 %s | pick %s" % (PAPER_NAME[lab], s, f(ms["v1"]), f(ms["v3"]), p[0] if p else "-"))
            for m in ms.values():
                if m is not None:
                    total += 1; below += m["val_l1_best"] < m["val_l1_persistence"]
            rows.append((p[2] if p else None, ms["v1"], ms["v3"]))
        for k, name in enumerate(("pick", "v1", "v3")):
            v = [(r[k]["auroc"], r[k]["auprc"], r[k]["sign_acc_flip"]) for r in rows if r[k] is not None]
            if v:
                print("%-20s MEAN %-4s n %d  AUROC %.3f AUPRC %.3f sign (flip) %.3f" % (PAPER_NAME[lab], name, len(v), *np.nanmean(np.array(v, float), 0)))
    sims = sorted({ds for ds, s in CELLS if ds.startswith("netsim")}, key=lambda d: int(d.split("sim")[2]))
    print("NetSim, %d simulations x training seeds %s (trajectory caps of the other baselines)" % (len(sims), NETSIM_SEEDS))
    acc = {"pick": {}, "v1": {}, "v3": {}}
    for ds in sims:
        for s in NETSIM_SEEDS:
            for v in VARIANTS:
                assert os.path.exists(os.path.join(R, "sgode", v, ds, "seed%d" % s, "metrics.json")), ("not finished", v, ds, s)
            ms = {v: met(v, ds, s) for v in VARIANTS}; p = pick(ds, s)
            print("%-20s seed %d  v1 %s | v3 %s | pick %s" % (ds, s, f(ms["v1"]), f(ms["v3"]), p[0] if p else "-"))
            for m in ms.values():
                if m is not None:
                    total += 1; below += m["val_l1_best"] < m["val_l1_persistence"]
            for k, m in (("pick", p[2] if p else None), ("v1", ms["v1"]), ("v3", ms["v3"])):
                if m is not None:
                    acc[k].setdefault(ds, []).append((m["auroc"], m["auprc"]))
    for k, v in acc.items():
        if v:
            print("%-20s MEAN %-4s n %d  AUROC %.3f AUPRC %.3f   (runs %d)" % (
                "NetSim", k, len(v), *np.mean([np.mean(x, 0) for x in v.values()], 0), sum(len(x) for x in v.values())))
    print("runs whose validation loss is below that of repeating the first snapshot: %d of %d" % (below, total))
    wall = {}
    for ds, s in CELLS:
        if ds.startswith("netsim") and s != 1:                 # wall times: the seed-1 runs, which shared one GPU
            continue
        for v in VARIANTS:
            m = met(v, ds, s)
            if m is not None:
                wall.setdefault("NetSim" if ds.startswith("netsim") else "synthetic", []).append(m["wall_sec"] / 3600)
    for k in sorted(wall):
        print("wall time per run, %s (%d runs%s; 14 jobs shared one GPU, not timed alone): %.1f to %.1f h, sum %.0f h"
              % (k, len(wall[k]), ", seed 1 only" if k == "NetSim" else "", min(wall[k]), max(wall[k]), sum(wall[k])))


if __name__ == "__main__":
    main()
