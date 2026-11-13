"""GLV-± on Barabasi-Albert graphs: does the self-field read-out of GLV-± still find a single class when the in-degrees
are heterogeneous? Data glv_BA30_m2_n0.5_seed{1,2,3} (hetndl.gen.glv --graph BA --m 2: undirected, signs drawn per
undirected edge; everything else as glv_N30_n0.5_seed{k}: N 30, c 0.5, half of the edges inhibitory, 40 trajectories).
Runs glv_ba/<dataset>/{nonneg,nowu}/<run> of experiments/reproduce/jobs.json: the two candidates with the arguments of
the main-table members of glv_N30_n0.5_seed{k}, data replaced; the candidate with the lower validation error
(val_mse_best) is read, the selection rule of the paper (sel_tables.py).
Class read-out = the one used for GLV-± in the paper (glv_selffield.py): the learned curves F(x, z_i) on 64 points over
the 5-95 % range of the observed states, hetndl.readout.select_k(curves=True); the number of classes K is chosen from
the data. Graph AUROC and sign accuracy are reported alongside; per seed the spread of the true in-strength
s_i = sum_j W_true[j, i] and of the in-degree, with the ER benchmark glv_N30_n0.5_seed{k} for comparison (and, if K > 1,
the mean in-strength and in-degree of every read-out class). Truth (W_true, A) enters only the scores and this
description.
Run from the repository root with PYTHONPATH=. (-> results/supplement/glv_ba.txt)."""
import glob, json, os
import numpy as np, torch
from hetndl.edge_readout import load_run
from hetndl.gen.kuramoto import load_dataset
from hetndl.readout import select_k
from hetndl.snapshot import metrics_of
R = "experiments/runs/glv_ba"; DS = "glv_BA30_m2_n0.5_seed%d"


def pick(ds):
    c = []
    for member in ("nonneg", "nowu"):
        ps = glob.glob(os.path.join(R, ds, member, "*", "metrics.json"))
        if len(ps) != 1:
            return None
        m = json.load(open(ps[0]))
        if m.get("failed"):
            continue
        c.append((m["val_mse_best"], member, os.path.dirname(ps[0]), m))
    return min(c, key=lambda x: x[0]) if c else None


def classes(run):
    m, d, vf, W = load_run(run); N = W.shape[0]
    lo, hi = np.percentile(d["data"], [5, 95]); xs = np.linspace(lo, hi, 64)
    with torch.no_grad():
        h = vf.phi(torch.tensor(xs, dtype=torch.float32).view(-1, 1))
        Fc = np.stack([vf.F(torch.cat([h, vf.z_eff()[i].expand(64, -1)], -1))[:, 0].numpy() for i in range(N)])
    K, lab, _ = select_k(Fc, k_max=5, curves=True)                                      # truth-free
    return K, lab, d


def main():
    print("GLV-± on BA graphs (m = 2, N = 30); selected candidate by validation error; class read-out of glv_selffield.py")
    Ks = []
    for s in (1, 2, 3):
        ds = DS % s; p = pick(ds)
        if p is None:
            print("%s: missing or failed" % ds); Ks.append(None); continue
        val, member, run, m = p; b = metrics_of(m)
        K, lab, d = classes(run); Ks.append(K)
        si = d["W_true"].sum(0); deg = (d["A"] > 0).sum(0)                               # scoring / description only
        er = load_dataset("experiments/data/glv_N30_n0.5_seed%d.npz" % s); si_er = er["W_true"].sum(0)
        print("%s  pick %-6s val %.3g  AUROC %.3f AUPRC %.3f sign %.3f  K = %d   in-strength s_i: BA range [%.2f, %.2f] sd %.3f "
              "(ER seed %d: [%.2f, %.2f] sd %.3f); in-degree BA %d-%d" % (
                  ds, member, val, b["auroc"], b["auprc"], b.get("sign_acc", float("nan")), K, si.min(), si.max(), si.std(),
                  s, si_er.min(), si_er.max(), si_er.std(), deg.min(), deg.max()))
        if K > 1:
            for k in range(K):
                idx = lab == k
                print("    class %d: %2d nodes, mean in-strength %+.3f, mean in-degree %.1f" % (k, idx.sum(), si[idx].mean(), deg[idx].mean()))
    if all(k is not None for k in Ks):
        print("\nnumber of self-field classes K per seed: %s%s" % (
            " ".join(str(k) for k in Ks), " (a single class on every seed, as on the ER graphs)" if all(k == 1 for k in Ks) else ""))


if __name__ == "__main__":
    main()
