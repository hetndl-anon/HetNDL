"""MTGNN sanity check: did the forecaster learn, i.e. is a chance-level graph a training failure or a property of
the learned adjacency? For every runs/mtgnn run, the snapshot's validation MAE (first state feature, original units,
12-step horizon; metrics.json val_mae_best) is compared with persistence (repeat the last input value) on the same
validation windows. Ratio = persistence MAE / MTGNN MAE (> 1: MTGNN beats persistence). No ground-truth graph is used.
Usage (project root): PYTHONPATH=. python experiments/analysis/mtgnn_forecast_check.py"""
import collections, glob, json
import numpy as np
from experiments.analysis.paper_seeds import PAPER_SEEDS
from hetndl.data import select_trajectories, trajectory_split
from hetndl.gen.kuramoto import load_dataset
L_IN = L_OUT = 12


def persistence_mae(m):
    d = load_dataset(m["args"]["data"]); X = np.asarray(d["data"], np.float64)
    keep = select_trajectories(X.shape[0], m["args"]["n_traj"], m["args"]["seed"]); X = X[keep]
    if d["meta"].get("system") == "kuramoto":
        X = np.concatenate([np.cos(X), np.sin(X)], -1)
    _, va, _ = trajectory_split(len(keep), m["args"]["seed"])
    e = [np.abs(X[k, t + L_IN:t + L_IN + L_OUT, :, 0] - X[k, t + L_IN - 1, :, 0]).mean()
         for k in va for t in range(X.shape[1] - L_IN - L_OUT + 1)]
    return float(np.mean(e))


def group(ds):
    if ds.startswith("netsim"):
        return "netsim"
    for g in ("_g1", "_g2", "_g3"):
        ds = ds.replace(g, "")
    return ds.rsplit("_seed", 1)[0]


if __name__ == "__main__":
    g = collections.defaultdict(list)
    for p in sorted(glob.glob("experiments/runs/mtgnn/*/seed*/metrics.json")):
        m = json.load(open(p))
        if int(p.split("/")[4].replace("seed", "")) not in PAPER_SEEDS:   # paper seeds only
            continue
        if m.get("failed"):
            print("skip (failed)", p); continue
        g[group(p.split("/")[3])].append(persistence_mae(m) / m["val_mae_best"])
    allr = []
    for k, v in sorted(g.items()):
        allr += v; print("%-30s n %2d  persistence / MTGNN val MAE: min %5.1f  median %5.1f  max %5.1f" % (k, len(v), min(v), np.median(v), max(v)))
    print("all %d runs: min %.1f  median %.1f  max %.1f; runs with ratio <= 1: %d" % (len(allr), min(allr), np.median(allr), max(allr), sum(r <= 1 for r in allr)))
