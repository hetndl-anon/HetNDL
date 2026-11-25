"""Convergence and budget checks of the Allier et al. 2026 runs (protocol in experiments/launch/allier_run.py).
  convergence (no ground truth): for every run under runs/allier/<root>: training loss of the first and the last epoch
    (their printed 'epoch k. loss'), the relative change of W from its initialisation ||W - W0|| / ||W0|| and the
    correlation of W with W0 over the ordered pairs (metrics.json, recorded by allier_run.py); medians per setting.
  budget (truth only in the scores): the sensitivity subset run with another number of epochs (runs/allier/<root2>, e.g.
    half the budget) against the same cells of <root>: graph AUROC / AUPRC (hetndl.metrics.auroc_auprc) and, on MixRelax, the type ARI.
usage (project root): PYTHONPATH=. python experiments/analysis/allier_checks.py --root <dir> [--root2 <dir>]"""
import argparse, glob, json, os, re
import numpy as np
from sklearn.metrics import adjusted_rand_score
from hetndl.gen.kuramoto import load_dataset
from hetndl.metrics import auroc_auprc

R = "experiments/runs/allier"
NAMES = [("kura_ER100_K2_s0.3_n0.5", "Kuramoto p⁻ 0.5"), ("kura_ER100_K2_s0.3_n0.2", "Kuramoto p⁻ 0.2"),
         ("kura_ER100_K2_s0.3_n0", "Kuramoto σ/K 0.15"), ("kura_ER100_K2_s2_n0", "Kuramoto σ/K 1.0"),
         ("kura_BA100", "Kuramoto p⁻ 0.5 (BA)"), ("a1v2_BA30", "MixRelax (BA)"), ("a1v2_N30", "MixRelax (ER)"),
         ("glv_N30", "GLV-±"), ("netsim_sim", "NetSim"), ("allier_control", "control (their model)")]


def setting(ds):
    return next(lab for pre, lab in NAMES if ds.startswith(pre))


def losses(d):
    f = os.path.join(d, "epochs.log") if os.path.exists(os.path.join(d, "epochs.log")) else os.path.join(d, "stdout.log")
    t = open(f, errors="replace").read().replace("\r", "\n")
    return [float(x) for x in re.findall(r"^epoch \d+\. loss: ([0-9.]+)", t, re.M)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--root2", default=None)
    a = ap.parse_args()
    print("Convergence of runs/allier/%s (no ground truth): medians per setting" % a.root)
    print("%-24s %4s  %-22s %12s %10s" % ("setting", "runs", "loss last / first", "|W-W0|/|W0|", "corr(W,W0)"))
    rows = {}
    for d in sorted(glob.glob(os.path.join(R, a.root, "*", "seed*"))):
        if not os.path.exists(os.path.join(d, "metrics.json")):
            continue
        m = json.load(open(os.path.join(d, "metrics.json"))); ls = losses(d)
        rows.setdefault(setting(d.split("/")[-2]), []).append((ls[-1] / ls[0] if ls else np.nan, m.get("w_rel_change", np.nan), m.get("w_corr_init", np.nan)))
    for lab, v in rows.items():
        v = np.array(v); md = np.nanmedian(v, 0)
        print("%-24s %4d  %-22s %12.3f %10.3f" % (lab, len(v), "%.3f" % md[0], md[1], md[2]))
    if a.root2:
        print("\nBudget: runs/allier/%s (column 2) against runs/allier/%s (column 1) on the same cells; dAUROC = column 2 - column 1 (truth only in these scores)" % (a.root2, a.root))
        print("%-34s %-17s %-17s %8s %s" % ("cell", "AUROC/AUPRC col1", "AUROC/AUPRC col2", "dAUROC", "ARI col1 / col2"))
        dd = []
        for d2 in sorted(glob.glob(os.path.join(R, a.root2, "*", "seed*"))):
            ds, sd = d2.split("/")[-2], d2.split("/")[-1]; d1 = os.path.join(R, a.root, ds, sd)
            if not (os.path.exists(os.path.join(d2, "metrics.json")) and os.path.exists(os.path.join(d1, "metrics.json"))):
                continue
            z = load_dataset("experiments/data/%s.npz" % ds)
            x1 = auroc_auprc(np.load(os.path.join(d1, "W.npy")), z["A"]); x2 = auroc_auprc(np.load(os.path.join(d2, "W.npy")), z["A"])
            ari = ""
            if ds.startswith("a1v2") or ds.startswith("allier_control"):
                ari = "%.3f / %.3f" % (adjusted_rand_score(z["classes"], np.load(os.path.join(d1, "labels.npy"))),
                                       adjusted_rand_score(z["classes"], np.load(os.path.join(d2, "labels.npy"))))
            dd.append(x2[0] - x1[0])
            print("%-34s %.3f / %.3f    %.3f / %.3f    %+7.3f %s" % (ds + "/" + sd, *x1, *x2, dd[-1], ari))
        if dd:
            print("largest |dAUROC| %.3f, mean dAUROC %+.3f over %d cells" % (np.max(np.abs(dd)), np.mean(dd), len(dd)))


if __name__ == "__main__":
    main()
