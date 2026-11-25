"""Mechanism read-out under observation noise and trajectory caps: MixOsc (A2) N60, the official
read-out of a2_fields.py (select_k on the learned self-fields, truth-free K; ARI against the true FHN / VdP classes,
per-class mean |F - f|). Runs, picked exactly as in noise_table.py / sel_tables.py (default procedure = lower validation
MSE of the two sign members):
  clean        runs/sel pick of a2_N60_seed<s>/add_z_signed_diff_lz0_seed<s>
  SNR x dB     runs/sel pick of .../add_z_signed_diff_lz0_snr<x>_seed<s> (no smoothing)
  SNR x + SG11 lower-validation-MSE member of runs/a2_sg/<ds>/{add_z_diff_lz0, add_z_signed_diff_lz0}_sg11_snr<x>_seed<s>
  n<k>         runs/sel pick of .../add_z_signed_diff_lz0_n<k>_seed<s> (k training+validation+test trajectories)
Run from the project root with PYTHONPATH=. (calls a2_fields.py with the hetndl interpreter)."""
import json, os, re, subprocess, sys, numpy as np
R = "experiments/runs"; sel = json.load(open(os.path.join(R, "sel", "selected.json")))
PY = sys.executable


def met(p):
    f = os.path.join(p, "metrics.json")
    if not os.path.exists(f):
        return None
    m = json.load(open(f)); return None if m.get("failed") else m


def readout(run):
    out = subprocess.run([PY, "experiments/analysis/a2_fields.py", run], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=".")).stdout
    k = re.search(r"K_sel (\d+) ARI\(K_sel\) ([0-9.\-]+)", out); e = re.search(r"per-class \|F−f\| \[([^\]]+)\]", out)
    if not k:
        return None
    return int(k.group(1)), float(k.group(2)), float(np.mean([float(x) for x in e.group(1).split()])) if e else float("nan")


rows = []
for lab, tag in [("clean", "")] + [("SNR %s dB" % x, "_snr%s" % x) for x in ("30", "20", "10")] + [("n%d" % k, "_n%d" % k) for k in (10, 5)]:
    res = []
    for s in (1, 2, 3):
        g = "a2_N60_seed%d/add_z_signed_diff_lz0%s_seed%d" % (s, tag, s)
        res.append(readout(sel[g]["run"]) if g in sel else None)
    rows.append((lab, res))
    if tag.startswith("_snr"):
        x = tag[4:]; res = []
        for s in (1, 2, 3):
            ds = "a2_N60_seed%d" % s
            pair = [(os.path.join(R, "a2_sg", ds, "%s_sg11_snr%s_seed%d" % (n, x, s))) for n in ("add_z_diff_lz0", "add_z_signed_diff_lz0")]
            ms = [met(p) for p in pair]
            res.append(readout(pair[int(ms[1]["val_mse_best"] < ms[0]["val_mse_best"])]) if all(ms) else None)
        rows.append(("SNR %s dB + SG11" % x, res))
print("%-20s %-28s %-24s %s" % ("MixOsc N60", "K_sel per seed", "ARI per seed (mean)", "mean |F - f| per seed"))
for lab, res in rows:
    ok = [r for r in res if r]
    print("%-20s %-28s %-24s %s" % (lab, " ".join(str(r[0]) if r else "-" for r in res),
                                   " ".join("%.3f" % r[1] if r else "-" for r in res) + (" (%.3f)" % np.mean([r[1] for r in ok]) if ok else ""),
                                   " ".join("%.3f" % r[2] if r else "-" for r in res)), flush=True)
