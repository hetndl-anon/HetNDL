"""Numeric accuracy of the Kuramoto node constants: for the fifteen reported Kuramoto runs (four ER settings
and the BA setting, data seeds 1-3, experiments/runs/sel/selected.json) the node constant c_i of the learned self-field
(hetndl.readout.node_constant on the uniform phase grid of omega_readout.py; no trajectory enters the read-out) against
the true natural frequency omega_i: signed Spearman and Pearson correlation, slope and intercept of the least-squares
line c = slope * omega + intercept, and the RMSE of c_i - omega_i (no fitted scale or shift). Truth (omega) only in the
scores. Run from the project root with PYTHONPATH=."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from hetndl.edge_readout import load_run
from hetndl.readout import node_constant
from experiments.analysis.baseline_table import B, PAPER_NAME, sel, sel_key
from experiments.analysis.omega_readout import GRID


def main():
    print("%-20s %4s %4s  %9s %8s %7s %9s %7s %9s %8s" % ("setting", "seed", "pick", "spearman", "pearson", "slope", "intercept", "RMSE", "sd(omega)", "max|err|"))
    rows = []
    for lab, cells in B:
        if "Kuramoto" not in lab:
            continue
        for ds, s in cells:
            g = sel_key(lab, ds, s)
            m, d, vf, W = load_run(sel[g]["run"]); om = d["omega"]; c = node_constant(vf, GRID)[:, 0]
            slope, icpt = np.polyfit(om, c, 1)
            r = (spearmanr(c, om).correlation, pearsonr(c, om)[0], slope, icpt, float(np.sqrt(((c - om) ** 2).mean())), float(om.std()),
                 float(np.abs(c - om).max()))
            rows.append(r)
            print("%-20s %4d %4s  %+9.3f %+8.3f %7.3f %+9.3f %7.4f %9.3f %8.4f" % ((PAPER_NAME[lab], s, "n" if sel[g]["variant"] == "nonneg" else "s") + r))
    v = np.array(rows)
    print("\n%d runs: signed Spearman %.3f-%.3f, Pearson %.4f-%.4f, slope %.3f-%.3f, |intercept| <= %.4f, RMSE %.4f-%.4f (max %.3f of sd(omega))" % (
        len(v), v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(), v[:, 2].min(), v[:, 2].max(), np.abs(v[:, 3]).max(),
        v[:, 4].min(), v[:, 4].max(), (v[:, 4] / v[:, 5]).max()))


if __name__ == "__main__":
    main()
