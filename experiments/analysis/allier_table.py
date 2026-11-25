"""Allier et al. 2026 (their code, run by experiments/launch/allier_run.py; protocol in its docstring) against HetNDL on
the cells their trainer can run (one state variable per node). Truth enters only here.
  * graph: AUROC / AUPRC of |W| over ordered pairs (hetndl.metrics.auroc_auprc, as for every method), mean over data
    seeds 1-3; per seed HetNDL wins / ties (|difference| <= 1e-3) / loses; HetNDL = the selected runs of the main table
    (experiments/runs/sel/selected.json via baseline_table.sel_key).
  * signs: hetndl.metrics.sign_accuracy on the true edges of the signed cells (W as their sign convention leaves it:
    excitatory positive under their monotone transfer function).
  * node types: ARI of their read-out (K-means on the latents, K = 2..10 by silhouette, saved by the runner) on the cells
    with several node laws (MixRelax ER and BA); HetNDL's ARI per seed from results/table3_classes.txt (node_table.py).
  * NetSim: per simulation the mean over training seeds 1-3 (the NetSim rule of baseline_table), W/T/L and the Wilcoxon
    signed-rank test over the 27 simulations against HetNDL's per-simulation means (baseline_table.netsim_ours).
Run from the project root: PYTHONPATH=. python experiments/analysis/allier_table.py [--model PDE_N5] [--partial]
(--partial: skip missing runs, for a preview) > results/supplement/allier_rerun.txt"""
import argparse, json, os, re, sys
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score
from hetndl.gen.kuramoto import load_dataset
from hetndl.metrics import auroc_auprc, sign_accuracy
from experiments.analysis.baseline_table import B, sel, sel_key, netsim_ours, NETSIM_SIMS, NETSIM_SEEDS

R = "experiments/runs/allier"
CELLS = ("Kuramoto σ/K 0.15", "Kuramoto σ/K 1.0", "Kuramoto p⁻ 0.2", "Kuramoto p⁻ 0.5", "BA Kuramoto p⁻ 0.5",
         "A1 (N30)", "BA A1", "GLV (signed)")
SHOW = {"A1 (N30)": "MixRelax (ER)", "BA A1": "MixRelax (BA)", "GLV (signed)": "GLV-±"}
TYPES = {"A1 (N30)": "MixRelax ", "BA A1": "MixRelax (BA)"}


def ours_ari():
    """{(node_table label, seed): HetNDL ARI} from results/table3_classes.txt (output of node_table.py)."""
    o = {}
    for line in open("results/table3_classes.txt"):
        m = re.match(r"(MixRelax(?: \(BA\))?)\s+seed (\d+)\s+HetNDL K \d+ ARI ([0-9.]+)", line)
        if m:
            o[(m.group(1).strip(), int(m.group(2)))] = float(m.group(3))
    return o


def run(model, ds, s):
    d = os.path.join(R, model, ds, "seed%d" % s)
    if not os.path.exists(os.path.join(d, "metrics.json")):
        return None
    return dict(W=np.load(os.path.join(d, "W.npy")), labels=np.load(os.path.join(d, "labels.npy")),
                m=json.load(open(os.path.join(d, "metrics.json"))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="PDE_N5"); ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    oari = ours_ari()
    print("Allier et al. 2026 (their code, model %s, last epoch, no ground truth in the run) vs HetNDL; truth only in this "
          "scoring. Graph: AUROC / AUPRC of |W|, mean over data seeds 1-3; W/T/L = seeds on which HetNDL wins / ties "
          "(within 0.001) / loses." % a.model)
    print("%-20s %-15s %-15s %-7s %-14s %s" % ("setting", "HetNDL", "Allier 2026", "W/T/L", "signs (H / A)", "types ARI (H / A), K"))
    for lab, cells in B:
        if lab not in CELLS:
            continue
        o, t, w, sg, ty, ks = [], [], [0, 0, 0], [], [], []
        for ds, s in cells:
            r = run(a.model, ds, s)
            if r is None:
                if a.partial:
                    continue
                raise SystemExit("missing run %s/%s/%s/seed%d" % (R, a.model, ds, s))
            d = load_dataset("experiments/data/%s.npz" % ds)
            g = sel[sel_key(lab, ds, s)]
            x = auroc_auprc(r["W"], d["A"]); o.append((g["auroc"], g["auprc"])); t.append(x)
            diff = g["auroc"] - x[0]; w[0 if diff > 1e-3 else 1 if abs(diff) <= 1e-3 else 2] += 1
            if (d["S"] < 0).any():
                sg.append((g["sign"], sign_accuracy(r["W"], d["S"])))
            if lab in TYPES:
                ty.append((oari.get((TYPES[lab].strip(), s), float("nan")), adjusted_rand_score(d["classes"], r["labels"])))
                ks.append(int(r["m"]["K"]))
        if not t:
            continue
        f = lambda v: "%.3f / %.3f" % tuple(np.mean(v, 0))
        sgs = ("%.3f / %.3f" % tuple(np.mean(sg, 0))) if sg else "—"
        tys = ("%.3f / %.3f, K %s" % (*np.mean(ty, 0), ",".join(map(str, ks)))) if ty else "—"
        print("%-20s %-15s %-15s %d/%d/%d   %-14s %s%s" % (SHOW.get(lab, lab), f(o), f(t), *w, sgs, tys,
                                                     "" if len(t) == len(cells) else "  (%d of %d seeds)" % (len(t), len(cells))))
    # NetSim
    ours = netsim_ours(); theirs, miss = {}, []
    for k in NETSIM_SIMS:
        v = []
        for s in NETSIM_SEEDS:
            r = run(a.model, "netsim_sim%d" % k, s)
            if r is None:
                miss.append("netsim_sim%d/seed%d" % (k, s)); continue
            v.append(auroc_auprc(r["W"], load_dataset("experiments/data/netsim_sim%d.npz" % k)["A"]))
        if v and (len(v) == len(NETSIM_SEEDS) or a.partial):
            theirs[k] = tuple(np.mean(v, 0))
    if miss and not a.partial:
        raise SystemExit("missing NetSim runs: %s" % ", ".join(miss[:5]))
    ks = sorted(theirs)
    if ks:
        oo = np.array([ours[k] for k in ks]); tt = np.array([theirs[k] for k in ks]); dd = oo[:, 0] - tt[:, 0]
        wtl = (int((dd > 1e-3).sum()), int((abs(dd) <= 1e-3).sum()), int((dd < -1e-3).sum()))
        p = wilcoxon(oo[:, 0], tt[:, 0]).pvalue if len(ks) > 5 else float("nan")
        print("\nNetSim (%d of %d simulations; per simulation the mean over training seeds 1-3): HetNDL %.3f / %.3f, "
              "Allier 2026 %.3f / %.3f; W/T/L %d/%d/%d; Wilcoxon p = %.3g" % (len(ks), len(NETSIM_SIMS), *oo.mean(0),
                                                                          *tt.mean(0), *wtl, p))
    if miss:
        print("missing runs (partial preview): %d, e.g. %s" % (len(miss), ", ".join(miss[:3])))


if __name__ == "__main__":
    main()
