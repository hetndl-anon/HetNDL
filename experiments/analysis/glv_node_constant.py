"""Node constant on GLV-± against the growth rates r_i (supplement, "Node parameters that are not additive").
Computes the |rho| quoted in the supplement with the one read-out path
(hetndl.readout.node_constant on the observed states of all trajectories, as omega_readout.py does for Kuramoto) for the
reported runs of data seeds 1-3 (runs/sel/selected.json). r_i is stored as d["omega"] by hetndl/gen/glv.py. Truth only in
the score. Run from the project root with PYTHONPATH=. (hetndl env)."""
import json
import numpy as np
from scipy.stats import spearmanr
from experiments.analysis.omega_readout import load_run
from hetndl.readout import node_constant

sel = json.load(open("experiments/runs/sel/selected.json"))
rho = []
for k in sorted(sel):
    if not k.startswith("glv_N30_n0.5_seed"):
        continue
    m, d, vf, W = load_run(sel[k]["run"])
    c = node_constant(vf, d["data"].reshape(-1, d["data"].shape[-1]))[:, 0]
    rho.append(abs(spearmanr(c, d["omega"]).correlation))
    print("%-50s node-constant |rho| with r_i %.3f" % (k, rho[-1]))
print("range over the seeds %.3f-%.3f, mean %.3f" % (min(rho), max(rho), np.mean(rho)))
