"""What the node latents of the Allier et al. 2026 runs encode on MixRelax (protocol of the runs: experiments/launch/allier_run.py).
With a coupling x_j - x_i their model, whose inputs read the sender's state only, must put -x_i times the incoming weights
of node i into the node function phi(a_i, x_i), so the latent a_i of a node should follow its in-degree (all MixRelax
weights are equal) besides its node law. Truth enters only here. Per run:
  * R2 of the in-degree on the two latent dimensions, fitted with an intercept inside every true class and pooled over
    the classes (1 - residual / total sum of squares, both within classes);
  * the same R2 after permuting the in-degrees inside every class (2000 permutations, generator seed 0): its 99th
    percentile;
  * on the directed graphs the same R2 for the out-degree, which the argument does not involve (the BA graphs are
    symmetric, so the two degrees coincide there);
  * type ARI of K-means on the latents at the true number of classes (n_init 10, random_state = data seed, as in the
    runner), next to the ARI of their read-out (K = 2..10 by the silhouette score, labels.npy).
Receiver factors (no ground truth): a sender-only input sees a product coupling h(x_i) k(x_j) only through the mean of
the receiver's factor h(x_i). Per Kuramoto dataset of the runs: the root mean square over the nodes of the data means of
cos(theta_i) and sin(theta_i) (the factors of sin(theta_j - theta_i); 0.707 would be the size of the factors
themselves); per GLV dataset: the range over the nodes of the data mean of x_i (the factor of x_i x_j).
usage (project root): PYTHONPATH=. python experiments/analysis/allier_latents.py --root PDE_N5_supp8_fig2lr_ep20
       > experiments/results/review/allier_latents.txt"""
import argparse, glob, os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from hetndl.gen.kuramoto import load_dataset

R = "experiments/runs/allier"
CELLS = (("MixRelax (ER)", "a1v2_N30_seed%d"), ("MixRelax (BA)", "a1v2_BA30_m2_seed%d"))
SEEDS = (1, 2, 3)


def r2_within(a, y, cl):
    """R2 of y on the latents a (with an intercept), fitted inside every class and pooled over the classes."""
    res = tot = 0.0
    for c in np.unique(cl):
        m = cl == c
        X = np.c_[a[m], np.ones(m.sum())]
        b = np.linalg.lstsq(X, y[m], rcond=None)[0]
        res += ((y[m] - X @ b) ** 2).sum(); tot += ((y[m] - y[m].mean()) ** 2).sum()
    return 1.0 - res / tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--perm", type=int, default=2000)
    a = ap.parse_args()
    print("Latents of the Allier et al. 2026 runs on MixRelax (runs/allier/%s); truth only in this scoring." % a.root)
    print("R2 in / out: R2 of the in- / out-degree on the two latent dimensions inside the true classes; null 99%%: 99th "
          "percentile of R2 in over %d permutations of the in-degrees inside the classes." % a.perm)
    print("%-15s %-4s %-9s %-8s %-9s %-8s %-9s %-11s %s" % ("setting", "seed", "in-degree", "R2 in", "null 99%", "R2 out", "ARI", "ARI true K", "K"))
    for lab, pat in CELLS:
        rows = []
        for s in SEEDS:
            ds = pat % s; d = load_dataset("experiments/data/%s.npz" % ds)
            A = np.asarray(d["A"], float); cl = np.asarray(d["classes"])           # A[j, i] = 1: edge j -> i
            run = os.path.join(R, a.root, ds, "seed%d" % s)
            lat = np.load(os.path.join(run, "a.npy")); labels = np.load(os.path.join(run, "labels.npy"))
            indeg, outdeg = A.sum(0), A.sum(1)
            rin = r2_within(lat, indeg, cl)
            rng = np.random.default_rng(0); null = []
            for _ in range(a.perm):
                p = indeg.copy()
                for c in np.unique(cl):
                    m = np.where(cl == c)[0]; p[m] = indeg[rng.permutation(m)]
                null.append(r2_within(lat, p, cl))
            sym = np.allclose(A, A.T)
            rout = np.nan if sym else r2_within(lat, outdeg, cl)
            kt = len(np.unique(cl))
            ari_k = adjusted_rand_score(cl, KMeans(n_clusters=kt, n_init=10, random_state=s).fit_predict(lat))
            ari = adjusted_rand_score(cl, labels)
            rows.append((rin, np.quantile(null, 0.99), rout, ari, ari_k))
            print("%-15s %-4d %-9s %-8.3f %-9.3f %-8s %-9.3f %-11.3f %d -> %d" % (
                lab, s, "%d-%d" % (indeg.min(), indeg.max()), rin, rows[-1][1], "sym." if sym else "%.3f" % rout, ari, ari_k,
                len(np.unique(labels)), kt))
        v = np.array(rows)
        print("%-15s mean ARI %.3f, at the true K %.3f; R2 in %.3f-%.3f, largest null 99%% %.3f%s" % (
            lab, v[:, 3].mean(), v[:, 4].mean(), v[:, 0].min(), v[:, 0].max(), v[:, 1].max(),
            "" if np.isnan(v[:, 2]).all() else "; R2 out %.3f-%.3f" % (np.nanmin(v[:, 2]), np.nanmax(v[:, 2]))))
    print("\nReceiver factors (no ground truth): data means per node")
    rms = {}
    for run in sorted(glob.glob(os.path.join(R, a.root, "kura_*"))):
        ds = os.path.basename(run); X = np.asarray(load_dataset("experiments/data/%s.npz" % ds)["data"])[..., 0]
        c, s = np.cos(X).mean((0, 1)), np.sin(X).mean((0, 1))
        rms[ds] = float(np.sqrt((c ** 2 + s ** 2).mean() / 2))
        print("%-34s rms over nodes of mean cos, mean sin: %.3f" % (ds, rms[ds]))
    print("Kuramoto: %d datasets, rms %.3f-%.3f" % (len(rms), min(rms.values()), max(rms.values())))
    for run in sorted(glob.glob(os.path.join(R, a.root, "glv_*"))):
        ds = os.path.basename(run); X = np.asarray(load_dataset("experiments/data/%s.npz" % ds)["data"])[..., 0]
        m = X.mean((0, 1))
        print("%-34s mean x_i over nodes: %.3f-%.3f (smallest state %.3f)" % (ds, m.min(), m.max(), X.min()))


if __name__ == "__main__":
    main()
