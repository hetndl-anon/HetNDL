"""Node-parameter read-out on Kuramoto: |Spearman(omega_hat, omega)| with omega_hat_i = the node constant of the
learned self-field (hetndl.readout.node_constant: F(phi(theta), z_i) averaged over theta), replacing the old
metrics.spearman_z, which projects z on its first principal component and is not meaningful once d_z = 2 (one main
cell gave 0.482 although the field is right). Phase observations are periodic, so theta is a uniform grid on [0, 2 pi)
(an observable-property rule; no noisy or clean trajectory enters the read-out). Truth (omega) only in the score.
Rows: every Kuramoto group of experiments/runs/sel/selected.json (main, robustness, noise, BA) and, with --runs, any
run directories (ablations). Run from the project root with PYTHONPATH=."""
import json, os, re, sys, numpy as np
from scipy.stats import spearmanr
from hetndl.edge_readout import load_run
from hetndl.metrics import spearman_z
from hetndl.readout import node_constant
GRID = np.linspace(0, 2 * np.pi, 64, endpoint=False)[:, None]


def score(run):
    m, d, vf, W = load_run(run)
    if not vf.additive:
        return None
    om = node_constant(vf, GRID)[:, 0]
    rho = spearmanr(om, d["omega"]).correlation
    old = spearman_z(vf.z_eff().detach().numpy(), d["omega"]) if vf.use_z else float("nan")
    return abs(rho), old


if __name__ == "__main__":
    if "--runs" in sys.argv:
        for r in sys.argv[sys.argv.index("--runs") + 1:]:
            s = score(r); print("%s  node-constant |rho| %s  (old PCA-z %s)" % (r, "%.3f" % s[0] if s else "-", "%.3f" % s[1] if s else "-"))
        sys.exit()
    sel = json.load(open("experiments/runs/sel/selected.json"))
    rows = {}
    for g, s in sorted(sel.items()):
        if "kura_" not in g:
            continue
        b = re.sub(r"_g\d", "", g.split("/")[-2] if g.startswith("ba/") else g.split("/")[0])
        tag = re.search(r"_(n\d+|sg51_noise[0-9.]+|noise[0-9.]+)_seed", g)
        key = b + (" [%s]" % tag.group(1) if tag else "")
        rows.setdefault(key, []).append(score(s["run"]) + ("n" if s["variant"] == "nonneg" else "s",))
    print("%-48s %-28s %-28s picks" % ("benchmark", "node-constant |rho| per seed", "old PCA-z |rho| per seed"))
    for k, v in rows.items():
        print("%-48s %-28s %-28s %s" % (k, " ".join("%.3f" % x[0] for x in v), " ".join("%.3f" % x[1] for x in v), "".join(x[2] for x in v)))
