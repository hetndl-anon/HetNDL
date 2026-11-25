"""Positive control of the Allier et al. 2026 baseline (experiments/launch/allier_control.py: data of their own model class
from their simulator, at the size of our benchmarks). For every training setting (runs/allier/control/<train>_ep<E>) and
control dataset and seed (allier_control_N100_seed<k>: 500 frames per run; _T2000_: the data-size control, 2000 frames
per run, same graph and types): graph AUROC / AUPRC of |W| (hetndl.metrics.auroc_auprc), sign accuracy on the true edges, Pearson r between
the learned and the true weights over all ordered pairs (their paper scores weights by slope and R^2), neuron-type ARI of
their read-out (K-means on the latents, K by silhouette) with the K it chose, final training loss. The calibration rule is
in the docstring of allier_control.py. Run from the project root: PYTHONPATH=. python experiments/analysis/allier_control_table.py"""
import glob, json, os, re
import numpy as np
from sklearn.metrics import adjusted_rand_score
from hetndl.metrics import auroc_auprc, sign_accuracy


def main():
    print(__doc__.split("\n")[0])
    print("%-26s %4s | %-15s %6s %7s %6s %3s | %s" % ("setting", "seed", "AUROC / AUPRC", "sign", "r(W)", "ARI", "K", "loss first -> last epoch"))
    for d in sorted(glob.glob("experiments/runs/allier/control/*_ep*/allier_control_N100_*/seed*")):
        tr = d.split("/")[4]; s = int(d.rsplit("seed", 1)[1]); ds = d.split("/")[5]
        tr = tr + ("" if "_T" not in ds else " " + ds.split("_")[3])   # data-size control: frames per run
        if not os.path.exists(os.path.join(d, "metrics.json")):
            print("%-26s %4d | running" % (tr, s)); continue
        z = np.load("experiments/data/%s.npz" % ds)
        W = np.load(os.path.join(d, "W.npy")); lab = np.load(os.path.join(d, "labels.npy")); m = json.load(open(os.path.join(d, "metrics.json")))
        off = ~np.eye(len(W), dtype=bool)
        au, ap = auroc_auprc(W, z["A"]); sg = sign_accuracy(W, z["S"]); r = np.corrcoef(W[off], z["W_true"][off])[0, 1]
        ari = adjusted_rand_score(z["classes"], lab)
        f = os.path.join(d, "epochs.log") if os.path.exists(os.path.join(d, "epochs.log")) else os.path.join(d, "stdout.log")
        ep = [float(x) for x in re.findall(r"^epoch \d+\. loss: ([0-9.]+)", open(f, errors="replace").read().replace("\r", "\n"), re.M)]
        print("%-26s %4d | %.3f / %.3f  %6.3f %7.3f %6.3f %3d | %s" % (tr, s, au, ap, sg, r, ari, m["K"], "%.1f -> %.1f" % (ep[0], ep[-1]) if ep else "?"))


if __name__ == "__main__":
    main()
