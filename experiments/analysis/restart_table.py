"""Training restarts on fixed data and a fixed split (--init_seed 101, 102; --seed and hence the split unchanged): data seed
1 of each of the twelve settings, both candidates; the signed candidate of the other four MixOsc cells (N = 30 and 60,
data seeds 2-3) and of the three a2ts cells (combined_table.py), where a stall of the signed candidate is the question.
Runs restart/... of experiments/reproduce/jobs.json. Written before any AUROC, selection or class count of these runs was
read (the a2ts restarts are also printed by combined_table.py; they are repeated here for completeness).

Reported per setting: the range of AUROC, whether the selected candidate is the same, whether the number of classes is
the same, and the number of stalls of C±. No pass / fail threshold is set; this script reports. Definitions (fixed before
reading):
  * restart 0 = the main-table runs of data seed 1 (manifest cell; Grid-118: restart 1 of the three, the cell that
    was restarted), restarts 1, 2 = _i101, _i102 of the same two members;
  * per restart the member with the lower val_mse_best is selected (the main-table rule); AUROC / AUPRC of its reported
    snapshot (metrics_of); sign accuracy on true edges where the truth has inhibitory edges;
  * classes: the paper's read-outs, on the selected run of each restart: readout_split_check.readout (Table 3) for
    MixRelax, MixRelax (BA), MixOsc N = 30 / 60 and Grid-118; a2_fields.py (Sect. 5.6) for MixOsc-E; K and ARI;
  * stall of C± (the rule of combined_table.py): val_mse(C±) > 10 x val_mse(C+) of the same restart (on all-excitatory
    settings the supplement's stalled pairs differ by 80-263x, non-stalled ones by < 7x); for the signed-only restarts
    of the other MixOsc cells the reference is that cell's main-table C+; for a2ts the reference is the best of the two
    a2t runs of the same data seed (combined_table.py).
A failed or missing run is listed and the restart left out (never rerun). Ground truth only in the scores.
Run from the project root with PYTHONPATH=. (-> results/supplement/restart.txt)."""
import json, os, re, subprocess, sys, numpy as np
from sklearn.metrics import adjusted_rand_score
from hetndl.edge_readout import load_run
from hetndl.snapshot import metrics_of
from experiments.analysis.baseline_table import B, PAPER_NAME, sel_key
from experiments.analysis.readout_split_check import readout
R = "experiments/runs"; PY = sys.executable; STALL = 10.0; INIT = (101, 102)
man = json.load(open(os.path.join(R, "sel", "manifest.json")))
MEMBERS = ("nonneg", "nowu")
CLASS_T3 = ("MixRelax", "MixRelax (BA)", "MixOsc N=30", "MixOsc N=60", "Grid-118")


def met(run):
    p = os.path.join(run, "metrics.json")
    if not os.path.exists(p):
        return None, "missing"
    m = json.load(open(p))
    return (None, "failed") if m.get("failed") else (m, "ok")


def fields(run):
    out = subprocess.run([PY, "experiments/analysis/a2_fields.py", run], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=".")).stdout
    g = re.search(r"K_sel (\d+) ARI\(K_sel\) (-?[0-9.]+)", out)
    return (int(g.group(1)), float(g.group(2))) if g else (None, float("nan"))


def classes(name, run):
    if name in CLASS_T3:
        m, d, vf, W = load_run(run); X = np.asarray(d["data"], np.float32)
        K, lab, _ = readout(vf, X, d["meta"]["system"]); return K, adjusted_rand_score(d["classes"], lab)
    if name.startswith("MixOsc-E"):
        return fields(run)
    return None, None


def restarts(cell, tag):
    """[(restart, {member: run})]: restart 0 = the manifest cell, 1-2 = _i101, _i102."""
    out = [(0, {v: cell[v] for v in MEMBERS})]
    for k, i in enumerate(INIT, 1):
        out.append((k, {v: os.path.join(R, "restart", v, tag, os.path.basename(cell[v]) + "_i%d" % i) for v in MEMBERS}))
    return out


def f3(x):
    return "  -  " if x is None or (isinstance(x, float) and not np.isfinite(x)) else "%.3f" % x


def main():
    stalls, judged = 0, 0
    print("Training restarts on data seed 1 (restart 0 = the main-table runs; 1, 2 = --init_seed 101, 102)\n")
    summary = []
    for lab, cs in B:
        name = PAPER_NAME[lab]; ds, s = cs[0]; cell = man[sel_key(lab, ds, s)]
        print("== %s (%s)" % (name, ds)); rows = []
        for k, runs in restarts(cell, ds):
            ms = {v: met(r) for v, r in runs.items()}
            bad = ["%s %s" % (v, st) for v, (m, st) in ms.items() if m is None]
            if bad:
                print("   restart %d PENDING %s" % (k, "; ".join(bad))); continue
            v = min(MEMBERS, key=lambda x: ms[x][0]["val_mse_best"]); m = ms[v][0]; b = metrics_of(m)
            ratio = ms["nowu"][0]["val_mse_best"] / ms["nonneg"][0]["val_mse_best"]; judged += 1; st = ratio > STALL; stalls += st
            K, ari = classes(name, runs[v])
            rows.append(dict(k=k, v=v, auroc=b["auroc"], auprc=b["auprc"], sign=b.get("sign_acc") if b.get("has_neg_edges") else None, K=K, ari=ari))
            print("   restart %d  picks %s  AUROC %s AUPRC %s sign %s  classes K %s ARI %s  val(C±)/val(C+) %.2f%s" % (
                k, "n" if v == "nonneg" else "s", f3(b["auroc"]), f3(b["auprc"]), f3(rows[-1]["sign"]), K if K is not None else "-", f3(ari),
                ratio, "  STALL" if st else ""))
        if rows:
            au = [r["auroc"] for r in rows]
            line = "%-22s n %d  AUROC %.3f-%.3f  picks %s  K %s" % (name, len(rows), min(au), max(au), "".join("n" if r["v"] == "nonneg" else "s" for r in rows),
                                                                 ",".join(str(r["K"]) for r in rows) if rows[0]["K"] is not None else "-")
            summary.append(line); print("   " + line)
    print("\n== signed-only restarts of the other MixOsc cells (reference: the cell's main-table C+)")
    for n in (30, 60):
        for s in (2, 3):
            ds = "a2_N%d_seed%d" % (n, s); cell = man["%s/add_z_signed_diff_lz0_seed%d" % (ds, s)]
            ref, _ = met(cell["nonneg"]); m0, _ = met(cell["nowu"])
            for k, i in enumerate((None,) + INIT):
                run = cell["nowu"] if i is None else os.path.join(R, "restart", "nowu", ds, os.path.basename(cell["nowu"]) + "_i%d" % i)
                m, st = met(run)
                if m is None or ref is None:
                    print("   %s restart %d PENDING (%s)" % (ds, k, st)); continue
                r = m["val_mse_best"] / ref["val_mse_best"]; judged += 1; stalls += r > STALL
                print("   %s restart %d  C± AUROC %s  val(C±)/val(C+) %.2f%s" % (ds, k, f3(metrics_of(m)["auroc"]), r, "  STALL" if r > STALL else ""))
    print("\n== a2ts (half of the edges inhibitory, tanh coupling; reference: best val of the two a2t runs, as in combined_table.py)")
    for s in (1, 2, 3):
        ds = "a2ts_N30_n0.5_seed%d" % s; base = os.path.join(R, "a2ts", ds); ctl = os.path.join(R, "a2t", "a2t_N30_seed%d" % s)
        refs = [met(os.path.join(ctl, x))[0] for x in ("add_z_diff_lz0_seed%d" % s, "add_z_signed_diff_lz0_seed%d" % s)]
        if any(r is None for r in refs):
            print("   %s control PENDING" % ds); continue
        ref = min(r["val_mse_best"] for r in refs)
        for k, run in enumerate([os.path.join(base, "add_z_signed_diff_lz0_seed%d" % s)] +
                                [os.path.join(R, "restart", "nowu", ds, "add_z_signed_diff_lz0_seed%d_i%d" % (s, i)) for i in INIT]):
            m, st = met(run)
            if m is None:
                print("   %s restart %d PENDING (%s)" % (ds, k, st)); continue
            r = m["val_mse_best"] / ref; judged += 1; stalls += r > STALL
            print("   %s restart %d  C± AUROC %s  val(C±, a2ts)/best val(a2t) %.2f%s" % (ds, k, f3(metrics_of(m)["auroc"]), r, "  STALL" if r > STALL else ""))
    print("\nSUMMARY (data seed 1, three restarts each)")
    for line in summary:
        print("  " + line)
    print("C± stalls: %d of %d judged restarts (rule: ratio > %g)" % (stalls, judged, STALL))


if __name__ == "__main__":
    main()
