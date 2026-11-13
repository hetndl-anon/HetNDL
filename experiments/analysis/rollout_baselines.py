"""Roll-out (dynamics fidelity) comparison for the paper: 100-step autoregressive roll-out NRMSE on the
TEST trajectories of every main-table seed, for HetNDL (default procedure, runs/sel/selected.json), NOGI (same pairs as
sel_vs_nogi.py), ARNI and Granger-lasso. Same definition as hetndl.metrics.rollout_nrmse: start at t = 0 of each test
trajectory (trajectory_split(n_traj, seed)[2], clean data), predict `steps` samples autoregressively, RMSE over all
predicted states divided by the std of the true states in the same window.

Baselines are refitted with the arguments stored in their main-table metrics.json (no new choices, no truth):
  * Granger: lasso VAR(1) on standardised lagged states at the validation-selected alpha (as granger_mechanism.py);
    roll-out = iterate the fitted one-step map x_{t+1} = f(x_t).
  * ARNI: the joint ("full" mode) least-squares fit of hetndl.baselines.arni (same basis rule, rows, ridge, smoothing),
    which is a complete model of dx/dt; roll-out = RK4 integration with 4 sub-steps per sampling step.
    On Kuramoto the main table reports the better of ARNI's full / greedy variants by AUROC; only the full fit is a
    complete dynamical model, so the roll-out always uses it.
Writes experiments/runs/rollout/rollout_summary.json. Run from the project root with PYTHONPATH=. (hetndl env)."""
import json, os, sys, time
import numpy as np
from sklearn.linear_model import Lasso

from hetndl.baselines.arni import self_basis, pair_basis, _full_fit, _rows
from hetndl.baselines.granger import lagged_design
from hetndl.data import trajectory_split, select_trajectories, add_noise
from hetndl.gen.kuramoto import load_dataset
from hetndl.snapshot import metrics_of

R = "experiments/runs"
STEPS = 100
OUT = os.path.join(R, "rollout", "rollout_summary.json")


def nrmse(pred, true):
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse / (float(np.std(true)) + 1e-12)


# ---------------------------------------------------------------- Granger
def granger_rollout(run, d):
    m = json.load(open(os.path.join(run, "metrics.json"))); a = m["args"]
    data = d["data"]; keep = select_trajectories(data.shape[0], a.get("n_traj"), a["seed"])
    obs = add_noise(data[keep], np.random.default_rng(a["seed"] + 12345), a.get("noise_snr"), a.get("noise_std"))
    tr, _, te = trajectory_split(len(keep), a["seed"])
    X, Y = lagged_design(obs, tr, m["lags"])
    if len(X) > a.get("max_rows", 50000):
        sel = np.random.default_rng(a["seed"]).choice(len(X), a["max_rows"], replace=False); X, Y = X[sel], Y[sel]
    mu, sd = X.mean(0), X.std(0) + 1e-12
    las = Lasso(alpha=m["alpha"], max_iter=20000, tol=1e-3).fit((X - mu) / sd, Y)
    S, T, N, dd = data.shape
    x = data[keep][te, 0].reshape(len(te), -1)
    preds = []
    for _ in range(min(STEPS, T - 1)):
        x = las.predict((x - mu) / sd)
        if not np.isfinite(x).all():
            return float("inf")
        preds.append(x)
    pred = np.stack(preds, 1).reshape(len(te), -1, N, dd)
    return nrmse(pred, data[keep][te, 1:pred.shape[1] + 1])


# ---------------------------------------------------------------- ARNI
def arni_fit(run, d):
    m = json.load(open(os.path.join(run, "metrics.json"))); a = m["args"]
    basis = m["basis"]; ridge = float(m.get("ridge", 0.0) or 0.0); smooth = int(m.get("smooth", 0) or 0)
    data = d["data"]; dt = float(d["meta"]["dt"])
    keep = select_trajectories(data.shape[0], a.get("n_traj"), a["seed"])
    obs = add_noise(data[keep], np.random.default_rng(a["seed"] + 12345), a.get("noise_snr"), a.get("noise_std"))
    tr, _, te = trajectory_split(obs.shape[0], a["seed"])
    X, Y = _rows(obs, dt, tr, a.get("max_rows", 50000), np.random.default_rng(a["seed"]), smooth)
    N = X.shape[1]
    fits = []
    for i in range(N):
        cand = [j for j in range(N) if j != i]
        S = self_basis(X[:, i], basis)
        beta, mu, _ = _full_fit(S, [pair_basis(X[:, i], X[:, j], basis) for j in cand], Y[:, i], ridge)
        fits.append((cand, beta, mu @ beta))
    return basis, fits, keep, te, dt


def arni_rhs(x, basis, fits):
    """x: (n, N, d) -> dx/dt (n, N, d)."""
    out = np.empty_like(x)
    for i, (cand, beta, off) in enumerate(fits):
        xi = x[:, i]
        B = np.concatenate([self_basis(xi, basis)] + [pair_basis(xi, x[:, j], basis) for j in cand], 1)
        out[:, i] = B @ beta - off
    return out


def arni_rollout(run, d, sub=4):
    basis, fits, keep, te, dt = arni_fit(run, d)
    data = d["data"][keep]
    x = data[te, 0].astype(np.float64); h = dt / sub; preds = []
    with np.errstate(all="ignore"):
        for _ in range(min(STEPS, data.shape[1] - 1)):
            for _ in range(sub):
                k1 = arni_rhs(x, basis, fits); k2 = arni_rhs(x + 0.5 * h * k1, basis, fits)
                k3 = arni_rhs(x + 0.5 * h * k2, basis, fits); k4 = arni_rhs(x + h * k3, basis, fits)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.isfinite(x).all() or np.abs(x).max() > 1e6:
                return float("inf")
            preds.append(x.copy())
    pred = np.stack(preds, 1)
    return nrmse(pred, data[te, 1:pred.shape[1] + 1])


# ---------------------------------------------------------------- benchmarks (same rows and pairs as sel_vs_nogi.py)
def bench():
    src = open("experiments/analysis/sel_vs_nogi.py").read()
    ns = {}; exec(src[:src.index('print("%-22s %6s')], ns)
    return ns["sel"], ns["B"], ns["met"], ns["nogi_grid"]


def dataset_of(sel_key):
    return sel_key.split("/")[-2] if sel_key.count("/") >= 2 else sel_key.split("/")[0]


def main():
    sel, B, met, nogi_grid = bench()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for lab, pairs in B:
        for g, nr in pairs:
            if g not in sel:
                continue
            ds = dataset_of(g); seed = int(g.rsplit("seed", 1)[1])
            key = "%s|%s" % (lab, ds)
            row = res.get(key, {})
            row["bench"], row["dataset"], row["seed"] = lab, ds, seed
            row["hetndl"] = sel[g].get("rollout")
            mn = nogi_grid(nr[1]) if isinstance(nr, tuple) else met(nr)
            row["nogi"] = metrics_of(mn).get("rollout_nrmse") if mn else None
            d = None
            for meth, fn, path in (("granger", granger_rollout, os.path.join(R, "granger", ds, "seed%d" % seed)),
                                   ("arni", arni_rollout, os.path.join(R, "arni", ds, "full_seed%d" % seed))):
                if meth in row or not os.path.exists(os.path.join(path, "metrics.json")):
                    continue
                if d is None:
                    d = load_dataset(os.path.join("experiments", "data", ds + ".npz"))
                t0 = time.time(); row[meth] = fn(path, d)
                print("  %-22s %-34s %-7s %.4g  (%.0f s)" % (lab, ds, meth, row[meth], time.time() - t0), flush=True)
            res[key] = row
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            json.dump(res, open(OUT, "w"), indent=1)
    print("\n%-22s %10s %10s %10s %10s   (mean over seeds; inf = diverged)" % ("benchmark", "HetNDL", "NOGI", "ARNI", "Granger"))
    for lab, _ in B:
        rows = [r for r in res.values() if r["bench"] == lab]
        f = lambda k: np.mean([r[k] for r in rows if r.get(k) is not None]) if rows else np.nan
        print("%-22s %10.4f %10.4f %10.4f %10.4f" % (lab, f("hetndl"), f("nogi"), f("arni"), f("granger")))


if __name__ == "__main__":
    main()
