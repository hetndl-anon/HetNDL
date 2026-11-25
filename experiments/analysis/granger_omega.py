"""Two-stage node-parameter baseline for Table 3: does Granger-lasso also read the natural frequencies?
For every Kuramoto main-table run of runs/granger, refit the lasso VAR(1) at the validation-selected alpha (exactly as
granger_mechanism.py does), take each node's constant term in raw units (the intercept of its own equation; for a phase
oscillator theta_{t+1} = theta_t + dt (omega_i + coupling) it is the counterpart of ARNI's constant coefficient) and
report |Spearman rho| against omega. Paper seeds only (paper_seeds.PAPER_SEEDS). Run from the project root with
PYTHONPATH=."""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso
from experiments.analysis.paper_seeds import PAPER_SEEDS
from hetndl.baselines.granger import lagged_design
from hetndl.data import add_noise, select_trajectories, trajectory_split
from hetndl.gen.kuramoto import load_dataset

SETTINGS = {"Kuramoto-omega": ["kura_ER100_K2_s0.3_n0", "kura_ER100_K2_s2_n0"],
            "Kuramoto-pm": ["kura_ER100_K2_s0.3_n0.2", "kura_ER100_K2_s0.3_n0.5", "kura_BA100_m2_K2_s0.3_n0.5"]}


def ds_name(base, s):
    if base.startswith("kura_BA"):
        return "%s_g%d" % (base, s)
    return base if s == 1 else "%s_g%d" % (base, s)


def rho(run):
    m = json.load(open(os.path.join(run, "metrics.json"))); a = m["args"]; d = load_dataset(a["data"])
    keep = select_trajectories(d["data"].shape[0], a.get("n_traj"), a["seed"])
    obs = add_noise(d["data"][keep], np.random.default_rng(a["seed"] + 12345), a.get("noise_snr"), a.get("noise_std"))
    tr, _, _ = trajectory_split(len(keep), a["seed"]); X, Y = lagged_design(obs, tr, m["lags"])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    las = Lasso(alpha=m["alpha"], max_iter=20000, tol=1e-3).fit((X - mu) / sd, Y)
    const = las.intercept_ - (las.coef_ / sd[None, :]) @ mu                      # raw-unit constant per output (node)
    return abs(spearmanr(const, d["omega"]).correlation)


out = {}
for lab, bases in SETTINGS.items():
    vals = []
    for b in bases:
        for s in PAPER_SEEDS:
            runs = glob.glob(os.path.join("experiments/runs/granger", ds_name(b, s), "seed%d" % s))
            if runs:
                vals.append(rho(runs[0])); print("  %-40s seed %d  |rho| %.3f" % (ds_name(b, s), s, vals[-1]), flush=True)
    out[lab] = vals
    print("%-16s n %d  mean %.3f  min %.3f" % (lab, len(vals), np.mean(vals), np.min(vals)))
