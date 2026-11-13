"""What the learned self-field is on GLV-±. With the diffusive coupling x_i x_j - x_i^2 the self-field that
the model can identify is r_i x - (1 - s_i) x^2 with s_i = sum_j W_true[j, i] (the summed signed in-weight moves the
-x_i^2 part of every edge into the self-field), not the intrinsic law x (r_i - x). For the three reported runs (data seeds
1-3, experiments/runs/sel/selected.json): R^2 of the learned per-node curves F(x, z_i), on 64 points over the 5-95 %
range of the observed states (the grid of a1_fcurves.py), against both targets, pooled over nodes and grid points with no
fitted scale or shift (1 - mean squared error / variance of the target), plus the per-node median; the number of
self-field groups read out by hetndl.readout.select_k(curves=True); and the per-node quadratic fit c0 + a x + b x^2 of
the learned curves (Pearson of a with r_i and of b with -(1 - s_i)). Truth (r_i = omega, W_true) only in the scores.
Run from the project root with PYTHONPATH=."""
import numpy as np, torch
from scipy.stats import pearsonr
from hetndl.edge_readout import load_run
from hetndl.readout import select_k
from experiments.analysis.baseline_table import B, sel, sel_key


def r2(F, T):
    return 1 - ((F - T) ** 2).mean() / T.var()


def main():
    print("%-6s %4s %4s  %-14s %2s  %10s %10s  %10s %10s  %8s %8s %9s %9s" % (
        "", "seed", "pick", "x range", "K", "R2 canon.", "R2 intr.", "med canon.", "med intr.", "quad R2", "a vs r", "b vs -q", "mean s_i"))
    out = []
    for lab, cells in B:
        if not lab.startswith("GLV"):
            continue
        for ds, s in cells:
            g = sel_key(lab, ds, s)
            m, d, vf, W = load_run(sel[g]["run"]); N = W.shape[0]
            lo, hi = np.percentile(d["data"], [5, 95]); xs = np.linspace(lo, hi, 64)
            with torch.no_grad():
                h = vf.phi(torch.tensor(xs, dtype=torch.float32).view(-1, 1))
                Fc = np.stack([vf.F(torch.cat([h, vf.z_eff()[i].expand(64, -1)], -1))[:, 0].numpy() for i in range(N)])
            K, _, _ = select_k(Fc, k_max=5, curves=True)                                    # truth-free
            r = d["omega"]; si = d["W_true"].sum(0)                                         # scoring only from here
            canon = r[:, None] * xs[None] - (1 - si)[:, None] * xs[None] ** 2; intr = r[:, None] * xs[None] - xs[None] ** 2
            X = np.stack([np.ones_like(xs), xs, xs ** 2], 1); coef = np.linalg.lstsq(X, Fc.T, rcond=None)[0].T
            node = lambda T: float(np.median([r2(Fc[i], T[i]) for i in range(N)]))
            row = (r2(Fc, canon), r2(Fc, intr), node(canon), node(intr), 1 - (Fc - coef @ X.T).var() / Fc.var(),
                   pearsonr(coef[:, 1], r)[0], pearsonr(coef[:, 2], -(1 - si))[0], float(si.mean()))
            out.append((K,) + row)
            print("%-6s %4d %4s  [%.2f, %.2f]   %2d  %10.3f %10.3f  %10.3f %10.3f  %8.4f %8.3f %9.3f %+9.3f" % (
                ("GLV-±", s, "n" if sel[g]["variant"] == "nonneg" else "s", lo, hi, K) + row))
    v = np.array(out)
    print("\nmean over %d seeds: R2 against r_i x - (1 - s_i) x^2 %.3f, against x (r_i - x) %.3f; self-field groups K = %s" % (
        len(v), v[:, 1].mean(), v[:, 2].mean(), " ".join("%d" % k for k in v[:, 0])))


if __name__ == "__main__":
    main()
