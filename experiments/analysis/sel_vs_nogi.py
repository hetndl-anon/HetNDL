"""Ours (new default procedure, experiments/runs/sel/selected.json from sel_tables.py) vs NOGI on every main benchmark:
mean AUROC / AUPRC over data seeds, per-seed wins / ties / losses on AUROC, and how many seeds of ours are still
pending. NOGI = the original NOGI model ("orig") with the same data, split and protocol snapshot; grid: best of its
3 restarts by validation MSE (grid_select.py rule). Kuramoto NOGI = runs/motiv_trig (seed 1) + runs/multiseed (seeds
2-3), as in finding 8. Netsim is summarised by sel_tables.py. Run from the project root with PYTHONPATH=."""
import glob, json, os, numpy as np
from hetndl.snapshot import metrics_of
from experiments.analysis.paper_seeds import PAPER_SEEDS
R = "experiments/runs"; sel = json.load(open(os.path.join(R, "sel", "selected.json")))


def met(run):
    p = os.path.join(R, run, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return None if m.get("failed") else m


def nogi_grid(s):
    ms = [m for m in (met("grid/grid118_seed%d/orig_seed%d%s" % (s, s, r)) for r in ("", "_r2", "_r3")) if m]
    return min(ms, key=lambda m: m["val_mse_best"]) if ms else None


B = []
kname = {"s0.3_n0": "Kuramoto σ/K 0.15", "s2_n0": "Kuramoto σ/K 1.0", "s0.3_n0.2": "Kuramoto p⁻ 0.2", "s0.3_n0.5": "Kuramoto p⁻ 0.5"}
for c, lab in kname.items():
    B.append((lab, [("kura_ER100_K2_%s%s/add_z_signed_diff_lz0_seed%d" % (c, "" if s == 1 else "_g%d" % s, s),
                     ("motiv_trig/kura_ER100_K2_%s/orig_seed1" % c) if s == 1 else ("multiseed/kura_ER100_K2_%s_g%d/orig_seed%d" % (c, s, s))) for s in (1, 2, 3)]))
B.append(("A1 (N30)", [("a1v2_N30_seed%d/add_z_signed_diff_lz0_seed%d" % (s, s), ("a1v2/a1v2_N30_seed1/orig_seed1" if s == 1 else "multiseed/a1v2_N30_seed%d/orig_seed%d" % (s, s))) for s in (1, 2, 3)]))
B.append(("A2 N30", [("a2_N30_seed%d/add_z_signed_diff_lz0_seed%d" % (s, s), "a2/a2_N30_seed%d/orig_seed%d" % (s, s)) for s in PAPER_SEEDS]))
B.append(("A2 N60", [("a2_N60_seed%d/add_z_signed_diff_lz0_seed%d" % (s, s), "a2/a2_N60_seed%d/orig_seed%d" % (s, s)) for s in (1, 2, 3)]))
B.append(("A3 (edge types)", [("a3_N30_seed%d/add_z_signed_diff_lz0_seed%d" % (s, s), "a3/a3_N30_seed%d/orig_seed%d" % (s, s)) for s in (1, 2, 3)]))
B.append(("GLV (signed)", [("glv_N30_n0.5_seed%d/add_z_signed_diff_lz0_seed%d" % (s, s), "glv/glv_N30_n0.5_seed%d/orig_seed%d" % (s, s)) for s in (1, 2, 3)]))
B.append(("IEEE-118 grid", [("grid118_seed%d/add_z_signed_diff_lz0_seed%d" % (s, s), ("GRID", s)) for s in (1, 2, 3)]))
B.append(("BA Kuramoto p⁻ 0.5", [("ba/kura_BA100_m2_K2_s0.3_n0.5_g%d/add_z_signed_diff_lz0_seed%d" % (s, s), "ba/kura_BA100_m2_K2_s0.3_n0.5_g%d/orig_seed%d" % (s, s)) for s in (1, 2, 3)]))
B.append(("BA A1", [("ba/a1v2_BA30_m2_seed%d/add_z_signed_diff_lz0_seed%d" % (s, s), "ba/a1v2_BA30_m2_seed%d/orig_seed%d" % (s, s)) for s in (1, 2, 3)]))
print("%-22s %6s %17s %17s %10s  picks" % ("benchmark", "seeds", "ours AUROC/AUPRC", "NOGI AUROC/AUPRC", "W/T/L"))
for lab, cells in B:
    o, n, picks, pend = [], [], "", 0
    for g, nr in cells:
        m = nogi_grid(nr[1]) if isinstance(nr, tuple) else met(nr)
        if g not in sel:
            pend += 1; continue
        if m is None:
            continue
        o.append((sel[g]["auroc"], sel[g]["auprc"])); b = metrics_of(m); n.append((b["auroc"], b["auprc"]))
        picks += "n" if sel[g]["variant"] == "nonneg" else "s"
    if not o:
        print("%-22s %6s %17s %17s %10s  (pending %d)" % (lab, "0", "—", "—", "—", pend)); continue
    o, n = np.array(o), np.array(n); d = o[:, 0] - n[:, 0]
    print("%-22s %6d    %.3f / %.3f    %.3f / %.3f   %2d/%d/%d  %s%s" % (lab, len(o), *o.mean(0), *n.mean(0), (d > 1e-3).sum(), (abs(d) <= 1e-3).sum(),
                                                                 (d < -1e-3).sum(), picks, "  (pending %d)" % pend if pend else ""))
