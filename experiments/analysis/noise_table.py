"""Observation-noise table for the paper, written to answer the obvious questions:
  * how strong is the noise?  Reported on two scales: SNR in dB (signal variance / noise variance) and noise std
    relative to the clean ONE-STEP increment std (what one-step training actually sees), plus the residual noise left
    after the Savitzky-Golay smoothing (window 51, polyorder 3) relative to that increment -> smoothing does not
    remove the noise;
  * does the result come from the smoothing?  Rows without smoothing for every method (runs/robust, ARNI raw);
  * is the comparison fair?  Same smoothing for every neural method (runs/noise_sg, runs/sel) and for Granger
    (runs/granger_noise_sg); ARNI gets the best
    window per seed chosen with the truth (an oracle advantage for the baseline, runs/arni_robust_tuned).
Signed Kuramoto p- 0.5 with phase noise 0.05 / 0.1 / 0.2 rad (smoothed) and A2 N60 at SNR 30 / 20 / 10 dB, both
without smoothing (runs/sel, runs/robust) and with Savitzky-Golay window 11 (5 % of the 200-step trajectory, the same
a-priori rule as Kuramoto's 51 / 1000; runs/a2_sg, same smoothing for ours and NOGI; ours = the lower-validation-MSE
member of the pair, the default-procedure rule). Ours = the default procedure (lower validation MSE of
non-negative / plain signed, experiments/runs/sel/selected.json); a cell whose plain-signed member is still training
is marked "pending" and the non-negative member is shown. Mean (sd) over data seeds 1-3; protocol snapshot.
Run from the project root with PYTHONPATH=. (uses the hetndl env for smoothing)."""
import glob, json, os, numpy as np
from hetndl.snapshot import metrics_of
from hetndl.gen.kuramoto import load_dataset
from hetndl.data import smooth_series
from experiments.analysis.omega_readout import score as omega_score   # node-constant |rho| (the one read-out path)
R = "experiments/runs"; sel = json.load(open(os.path.join(R, "sel", "selected.json")))
man = json.load(open(os.path.join(R, "sel", "manifest.json")))


def met(p):
    p = p if p.startswith(R) else os.path.join(R, p)
    if not os.path.exists(os.path.join(p, "metrics.json")):
        return None
    m = json.load(open(os.path.join(p, "metrics.json")))
    return None if m.get("failed") else m


def ms(vals, k=0):
    v = np.array([x[k] for x in vals if x is not None], float)
    return "—" if len(v) == 0 else ("%.3f" % v.mean() + (" (%.3f)" % v.std() if len(v) > 1 else "") + ("" if len(v) == 3 else " n=%d" % len(v)))


def au(m):
    if m is None:
        return None
    b = metrics_of(m); return (b["auroc"], b["auprc"], b.get("sign_acc", np.nan), np.nan)   # rho: omega_score


def arni(path):
    p = os.path.join(R, path, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p)); return (m["auroc"], m["auprc"], m.get("sign_acc_jac", np.nan), np.nan)


def granger(path):
    """Granger metrics.json is flat (no snapshot blocks); unsigned scores, so no sign accuracy."""
    p = os.path.join(R, path, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p)); return (m["auroc"], m["auprc"], np.nan, np.nan)


rng = np.random.default_rng(0)
OUT = {"kuramoto": {}, "mixosc": {}}
print("=== Signed Kuramoto p- 0.5 (N 100, dt 0.01), phase noise in rad")
for sig in ("0.05", "0.1", "0.2"):
    scales, rows = [], {k: [] for k in ("ours", "ours_raw", "nogi", "nogi_raw", "arni_raw", "arni_best", "granger")}
    picks = ""
    for s in (1, 2, 3):
        ds = "kura_ER100_K2_s0.3_n0.5" + ("" if s == 1 else "_g%d" % s)
        X = load_dataset("experiments/data/%s.npz" % ds)["data"]; inc = np.diff(X, axis=1).std(); sg = float(sig)
        res = smooth_series(X + rng.normal(0, sg, X.shape), 51) - X
        scales.append((10 * np.log10(X.var() / sg ** 2), sg / inc, res.std() / inc))
        g = "%s/add_z_signed_diff_lz0_sg51_noise%s_seed%d" % (ds, sig, s)
        if g in sel:
            r_ = au(met(sel[g]["run"])); rows["ours"].append(r_[:3] + (omega_score(sel[g]["run"])[0],)); picks += "n" if sel[g]["variant"] == "nonneg" else "s"
        rows["ours_raw"].append(au(met("robust/%s/add_z_signed_diff_lz0_noise%s_seed%d" % (ds, sig, s))))
        rows["nogi"].append(au(met("noise_sg/%s/orig_sg51_noise%s_seed%d" % (ds, sig, s))))
        rows["nogi_raw"].append(au(met("robust/%s/orig_noise%s_seed%d" % (ds, sig, s))))
        rows["arni_raw"].append(arni("arni_robust/%s/noise%s_seed%d" % (ds, sig, s)))
        cands = [arni(os.path.relpath(os.path.dirname(p), R)) for p in glob.glob(os.path.join(R, "arni_robust_tuned", ds, "noise%s_w*_seed%d" % (sig, s), "metrics.json"))]
        cands = [c for c in cands + [rows["arni_raw"][-1]] if c is not None]
        rows["arni_best"].append(max(cands, key=lambda c: c[0]) if cands else None)
        rows["granger"].append(granger("granger_noise_sg/%s/sg51_noise%s_seed%d" % (ds, sig, s)))   # same SG51 as ours / NOGI
    sc = np.array(scales).mean(0); OUT["kuramoto"][sig] = dict(rows=rows, scales=sc.tolist())
    print("noise %s rad: SNR %.1f dB | noise / one-step increment %.0fx | residual after SG51 / increment %.1fx" % (sig, *sc))
    print("   ours (smoothed, picks %s)  AUROC %s  AUPRC %s  sign %s  node-constant |rho| %s" % (picks, ms(rows["ours"]), ms(rows["ours"], 1), ms(rows["ours"], 2), ms(rows["ours"], 3)))
    print("   ours signed member, NO smoothing  AUROC %s  AUPRC %s   per seed %s" % (ms(rows["ours_raw"]), ms(rows["ours_raw"], 1), " ".join("%.3f" % r[0] for r in rows["ours_raw"] if r)))
    print("   NOGI smoothed  AUROC %s  AUPRC %s | NOGI NO smoothing  AUROC %s  AUPRC %s" % (ms(rows["nogi"]), ms(rows["nogi"], 1), ms(rows["nogi_raw"]), ms(rows["nogi_raw"], 1)))
    print("   ARNI raw  AUROC %s  AUPRC %s | ARNI best window per seed (oracle)  AUROC %s  AUPRC %s" % (ms(rows["arni_raw"]), ms(rows["arni_raw"], 1), ms(rows["arni_best"]), ms(rows["arni_best"], 1)))
    print("   Granger smoothed  AUROC %s  AUPRC %s" % (ms(rows["granger"]), ms(rows["granger"], 1)))

print("\n=== A2 N60 (FHN / VdP, dt 0.05), additive noise at a given SNR; without smoothing and with SG window 11")
for snr in ("30", "20", "10"):
    scales, rows, picks, picks_sg = [], {k: [] for k in ("ours", "nonneg", "nowu", "nogi", "arni_raw", "arni_best", "ours_sg", "nogi_sg")}, "", ""
    for s in (1, 2, 3):
        ds = "a2_N60_seed%d" % s; X = load_dataset("experiments/data/%s.npz" % ds)["data"]
        inc = np.diff(X, axis=1).std(axis=(0, 1)).mean(); sd = np.sqrt(X.var(axis=(0, 1)) / 10 ** (float(snr) / 10)).mean()
        sdd = np.sqrt(X.var(axis=(0, 1)) / 10 ** (float(snr) / 10))
        res = smooth_series(X + rng.normal(0, 1, X.shape) * sdd, 11) - X
        scales.append((float(snr), sd / inc, (res.std(axis=(0, 1)) / np.diff(X, axis=1).std(axis=(0, 1))).mean()))
        cell = "%s/add_z_signed_diff_lz0_snr%s_seed%d" % (ds, snr, s)
        pair = [met("a2_sg/%s/%s_sg11_snr%s_seed%d" % (ds, n, snr, s)) for n in ("add_z_diff_lz0", "add_z_signed_diff_lz0")]
        if all(m is not None for m in pair):                                       # default-procedure pick by val MSE
            k = int(pair[1]["val_mse_best"] < pair[0]["val_mse_best"]); rows["ours_sg"].append(au(pair[k])); picks_sg += "ns"[k]
        rows["nogi_sg"].append(au(met("a2_sg/%s/orig_sg11_snr%s_seed%d" % (ds, snr, s))))
        rows["nonneg"].append(au(met(man[cell]["nonneg"]))); rows["nowu"].append(au(met(man[cell]["nowu"])))
        if cell in sel:
            rows["ours"].append(au(met(sel[cell]["run"]))); picks += "n" if sel[cell]["variant"] == "nonneg" else "s"
        rows["nogi"].append(au(met("robust/%s/orig_snr%s_seed%d" % (ds, snr, s))))
        rows["arni_raw"].append(arni("arni_robust/%s/snr%s_seed%d" % (ds, snr, s)))
        cands = [arni(os.path.relpath(os.path.dirname(p), R)) for p in glob.glob(os.path.join(R, "arni_robust_tuned", ds, "snr%s_w*_seed%d" % (snr, s), "metrics.json"))]
        cands = [c for c in cands + [rows["arni_raw"][-1]] if c is not None]
        rows["arni_best"].append(max(cands, key=lambda c: c[0]) if cands else None)
    sc = np.array(scales).mean(0); OUT["mixosc"][snr] = dict(rows=rows, scales=sc.tolist())
    print("SNR %s dB: noise / one-step increment %.1fx | residual after SG11 / increment %.1fx" % (snr, sc[1], sc[2]))
    print("   ours (picks %s)  AUROC %s  AUPRC %s   [non-negative member %s / %s; plain-signed member %s / %s]" % (
        picks or "pending", ms(rows["ours"]), ms(rows["ours"], 1), ms(rows["nonneg"]), ms(rows["nonneg"], 1), ms(rows["nowu"]), ms(rows["nowu"], 1)))
    print("   NOGI  AUROC %s  AUPRC %s | ARNI raw %s / %s | ARNI best window (oracle) %s / %s" % (
        ms(rows["nogi"]), ms(rows["nogi"], 1), ms(rows["arni_raw"]), ms(rows["arni_raw"], 1), ms(rows["arni_best"]), ms(rows["arni_best"], 1)))
    print("   SG11 smoothed: ours (picks %s)  AUROC %s  AUPRC %s | NOGI  AUROC %s  AUPRC %s" % (
        picks_sg or "pending", ms(rows["ours_sg"]), ms(rows["ours_sg"], 1), ms(rows["nogi_sg"]), ms(rows["nogi_sg"], 1)))

if "--json" in __import__("sys").argv:                                   # same numbers for the figure scripts
    _p = __import__("sys").argv[__import__("sys").argv.index("--json") + 1]
    os.makedirs(os.path.dirname(_p), exist_ok=True); json.dump(OUT, open(_p, "w"), indent=1, default=float)
