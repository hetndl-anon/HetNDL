"""Table 5 (ablation + sensitivity) from runs/ablation. Every cell = default procedure (lower validation MSE of the
two sign members, each a copy of the main-table member with one change, experiments/launch/ablation.py). Metrics, all
through the official read-outs (never metrics.json's spearman_z / cluster_ari):
  topology   AUROC / AUPRC of the protocol snapshot;
  Kuramoto   node-constant |rho| (experiments/analysis/omega_readout.py score -> hetndl.readout.node_constant);
  MixRelax   select_k ARI of the learned F curves (a1_fcurves.py);  MixOsc  select_k ARI of the learned fields (a2_fields.py).
Full model = the main-table pick (runs/sel/selected.json) on the same data seeds. "no z" has no node read-out. Window rows:
signed Kuramoto p- 0.5 with phase noise 0.1, smoothing window 21 / 101 against the main window 51. Single-sign-member rows:
experiments/analysis/sign_member_table.py. Run from the project root with PYTHONPATH=."""
import collections, glob, json, os, re, subprocess, sys, numpy as np
from hetndl.snapshot import metrics_of
from experiments.analysis.omega_readout import score as omega_score
R = "experiments/runs"; A = os.path.join(R, "ablation"); PY = sys.executable
sel = json.load(open(os.path.join(R, "sel", "selected.json")))
FAM = {"kura": ["kura_ER100_K2_s0.3_n0.5" + ("" if s == 1 else "_g%d" % s) for s in (1, 2, 3)],
       "a1": ["a1v2_N30_seed%d" % s for s in (1, 2, 3)], "a2": ["a2_N30_seed%d" % s for s in (1, 2, 3)]}


def seed(ds):
    m = re.search(r"_g(\d+)$", ds) or re.search(r"_seed(\d+)$", ds); return int(m.group(1)) if m else 1


def ari(script, run):
    out = subprocess.run([PY, script, run], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=".")).stdout
    k = re.search(r"ARI\(K_sel\) ([0-9.\-]+)", out)
    return float(k.group(1)) if k else None


def node_readout(fam, run, has_z=True):
    if not has_z:
        return None
    if fam == "kura":
        s = omega_score(run); return s[0] if s else None
    return ari("experiments/analysis/a1_fcurves.py" if fam == "a1" else "experiments/analysis/a2_fields.py", run)


def pick(group, variant, ds):
    c = []
    for mem in ("nonneg", "nowu"):
        for p in glob.glob(os.path.join(A, group, variant, mem, ds, "*", "metrics.json")):
            m = json.load(open(p))
            if not m.get("failed"):
                c.append((m["val_mse_best"], os.path.dirname(p), m))
    return min(c, key=lambda x: x[0]) if len(c) == 2 else None


def cell(fam, runs_metrics, has_z=True):
    if not runs_metrics or any(r is None for r in runs_metrics):
        return None
    top = np.array([[metrics_of(m)["auroc"], metrics_of(m)["auprc"]] for _, m in runs_metrics])
    rd = [node_readout(fam, run, has_z) for run, _ in runs_metrics]
    return top.mean(0), (None if any(x is None for x in rd) else (np.mean(rd), rd))


def fmt(c):
    if c is None:
        return "%-28s" % "pending"
    t, r = c
    return "%-28s" % ("%.3f/%.3f  %s" % (t[0], t[1], "—" if r is None else "%.3f" % r[0]))


ROWS = [("full model", None, None), ("no z", "core", "noz"), ("free coupling", "core", "free"), ("z penalty", "core", "lz"),
        ("d_z = 1", "sens", "dz1"), ("d_z = 4", "sens", "dz4"), ("width 8", "sens", "h8"), ("width 32", "sens", "h32"), ("λ_1 = 1e-6", "sens", "l1e6")]
print("%-16s %-28s %-28s %-28s" % ("variant", "Kuramoto-± AUROC/AUPRC |ρ|", "MixRelax AUROC/AUPRC ARI", "MixOsc AUROC/AUPRC ARI"))
for lab, g, v in ROWS:
    cols = []
    for fam in ("kura", "a1", "a2"):
        if g is None:
            rm = []
            for ds in FAM[fam]:
                run = sel["%s/add_z_signed_diff_lz0_seed%d" % (ds, seed(ds))]["run"]; rm.append((run, json.load(open(run + "/metrics.json"))))
        elif not os.path.isdir(os.path.join(A, g, v, "nonneg", FAM[fam][0])):
            cols.append("%-28s" % "—"); continue
        else:
            rm = [(p[1], p[2]) if p else None for p in (pick(g, v, ds) for ds in FAM[fam])]
        cols.append(fmt(cell(fam, rm, has_z=(v != "noz"))))
    print("%-16s %s" % (lab, " ".join(cols)), flush=True)

print("\nsmoothing window (signed Kuramoto p- 0.5, phase noise 0.1): AUROC/AUPRC |ρ|")
ref = [sel["%s/add_z_signed_diff_lz0_sg51_noise0.1_seed%d" % (ds, seed(ds))]["run"] for ds in FAM["kura"]]
print("  window 51 (main)  %s" % fmt(cell("kura", [(r, json.load(open(r + "/metrics.json"))) for r in ref])))
for w in ("sg21", "sg101"):
    rm = [(p[1], p[2]) if p else None for p in (pick("window", w, ds) for ds in FAM["kura"])]
    print("  window %-10s %s" % (w[2:], fmt(cell("kura", rm))))
