"""Signed NOGI: the shared-function NOGI-style model without the non-negativity of W (variant orig_signed), trained with the
arguments of the NOGI run of every main-table cell and of every NetSim simulation (the runs snogi/... of
experiments/reproduce/jobs.json; Grid-118: the three restarts of each data seed, NetSim: three training seeds). The scores
and the reading rule below were fixed before any of these results was read.
Scores, exactly as for NOGI in Table 2:
  * AUROC / AUPRC of the reported snapshot (hetndl.snapshot.metrics_of), ranking by |W| (hetndl.metrics.auroc_auprc);
  * Grid-118: of the three restarts per data seed (orig_signed_seed{s}, _r2, _r3) the one with the lowest validation MSE
    (val_mse_best), as baseline_table.nogi_grid does for NOGI;
  * sign accuracy on the true edges under the better global sign, max(acc(W), acc(-W)), from the snapshot's W: the sign
    of W has no convention in a non-additive model; computed on the settings with inhibitory edges;
  * NetSim: per simulation the mean over training seeds 1-3, mean over the 27 simulations, two-sided Wilcoxon over the
    simulations of HetNDL against signed NOGI (scipy defaults, as baseline_table.py);
  * per data seed W/T/L of HetNDL against signed NOGI (|AUROC difference| <= 0.001 = tie, as Table 2);
  * a failed or missing run is listed and left out of its mean (never rerun, never replaced).
Reading (fixed before any result):
  (a) "close": on the four settings with inhibitory edges (Kuramoto-+- p- 0.2, p- 0.5, Kuramoto-+- (BA), GLV-+-) the
      mean AUROC of signed NOGI is within 0.01 of HetNDL's in at least three of the four. Then the graph-recovery gain
      over NOGI comes mainly from allowing negative weights, and what HetNDL adds is the node-level read-outs and the
      edge-conditioned coupling on top of a signed graph.
  (b) otherwise "clearly below": signed NOGI is reported next to Table 2 as a check that the sign alone does not explain
      HetNDL's graph recovery.
  Also printed: the number of the twelve settings with |mean AUROC difference| <= 0.01 and the number with HetNDL ahead
  by more than 0.01.
Ground truth only in the scores. Run from the project root with PYTHONPATH=. (-> results/supplement/snogi.txt)."""
import json, os, numpy as np
from scipy.stats import wilcoxon
from hetndl.gen.kuramoto import load_dataset
from hetndl.snapshot import array_path, metrics_of
from experiments.analysis.baseline_table import B, PAPER_NAME, R, met, nogi_grid, nogi_path, score, sel, sel_key
INHIB = ("Kuramoto-± p⁻ 0.2", "Kuramoto-± p⁻ 0.5", "Kuramoto-± (BA)", "GLV-±")


def load(path):
    p = os.path.join(R, path, "metrics.json")
    if not os.path.exists(p):
        return None, "missing"
    m = json.load(open(p))
    return (None, "failed: " + m.get("reason", "")[:80]) if m.get("failed") else (m, "")


def snogi_run(lab, ds, s):
    """(run path relative to runs/, metrics, note) of the signed NOGI run reported for one data seed."""
    if lab.startswith("IEEE"):
        c = []
        for r in ("", "_r2", "_r3"):
            p = "snogi/%s/orig_signed_seed%d%s" % (ds, s, r); m, note = load(p)
            if m:
                c.append((p, m))
        if not c:
            return None, None, "no finished restart"
        p, m = min(c, key=lambda x: x[1]["val_mse_best"])
        return p, m, "restart %s of %d" % (p.rsplit("_", 1)[-1] if p.endswith(("_r2", "_r3")) else "1", len(c))
    p = "snogi/%s/%s" % (ds, os.path.basename(nogi_path(lab, ds, s)).replace("orig_", "orig_signed_"))
    m, note = load(p)
    return p, m, note


def sign_acc(path, m, ds):
    S = load_dataset("experiments/data/%s.npz" % ds)["S"]; E = S != 0
    if not (S < 0).any():
        return None
    W = np.load(array_path(os.path.join(R, path), "W", m))
    a = float((np.sign(W[E]) == np.sign(S[E])).mean())
    return max(a, 1.0 - a)


def main():
    print("Signed NOGI (orig_signed) vs HetNDL and NOGI; AUROC / AUPRC, means over data seeds 1-3\n")
    print("%-20s %-15s %-15s %-15s %8s %7s  %s" % ("setting", "HetNDL", "NOGI", "signed NOGI", "W/T/L", "sign", "notes"))
    close = {}; lead = 0; level = 0; tot = np.zeros(3, int)
    for lab, cells in B:
        name = PAPER_NAME[lab]; o, n, sn, notes, sg = [], [], [], [], []
        for ds, s in cells:
            g = sel_key(lab, ds, s); o.append((sel[g]["auroc"], sel[g]["auprc"]))
            nm = nogi_grid(s) if lab.startswith("IEEE") else met(nogi_path(lab, ds, s)); n.append(score(nm))
            p, m, note = snogi_run(lab, ds, s)
            if m is None:
                sn.append(None); notes.append("seed %d %s" % (s, note)); continue
            b = metrics_of(m); sn.append((b["auroc"], b["auprc"]))
            if note:
                notes.append("seed %d %s" % (s, note))
            a = sign_acc(p, m, ds)
            if a is not None:
                sg.append(a)
        ok = [i for i, x in enumerate(sn) if x is not None]
        if not ok:
            print("%-20s no finished run (%s)" % (name, "; ".join(notes))); continue
        om, nm_, sm = np.mean(o, 0), np.mean([x for x in n if x], 0), np.mean([sn[i] for i in ok], 0)
        d = np.array([o[i][0] - sn[i][0] for i in ok]); wtl = ((d > 1e-3).sum(), (abs(d) <= 1e-3).sum(), (d < -1e-3).sum()); tot += wtl
        level += abs(om[0] - sm[0]) <= 0.01; lead += (om[0] - sm[0]) > 0.01
        if name in INHIB:
            close[name] = abs(om[0] - sm[0]) <= 0.01
        print("%-20s %.3f / %.3f  %.3f / %.3f  %.3f / %.3f  %d/%d/%d  %7s  %s%s" % (
            name, *om, *nm_, *sm, *wtl, ("%.3f" % np.mean(sg)) if sg else "-", "" if len(ok) == 3 else "(%d of 3 seeds) " % len(ok), "; ".join(notes)))
    print("\nper data seed, HetNDL vs signed NOGI: W/T/L %d/%d/%d" % tuple(tot))
    print("settings with |mean AUROC difference| <= 0.01: %d of 12; HetNDL ahead by more than 0.01: %d" % (level, lead))
    print("inhibitory settings within 0.01: %d of %d (%s)" % (sum(close.values()), len(close), ", ".join("%s %s" % (k, "yes" if v else "no") for k, v in close.items())))
    if len(close) == 4:
        print("READING: %s" % ("(a) close: the gain over NOGI comes mainly from signed weights" if sum(close.values()) >= 3
                                else "(b) clearly below: report next to Table 2"))
    else:
        print("READING: incomplete (%d of 4 inhibitory settings have results)" % len(close))
    # NetSim
    sims = [k for k in range(1, 29) if k != 13]; ho, so_, no_ = [], [], []
    for k in sims:
        ds = "netsim_sim%d" % k
        hs = [sel[g]["auroc"] for g in sel if g.startswith(ds + "/")]
        ss = [metrics_of(m)["auroc"] for m in (load("snogi/%s/orig_signed_seed%d" % (ds, t))[0] for t in (1, 2, 3)) if m]
        if hs and len(ss) == 3:
            ho.append(np.mean(hs)); so_.append(np.mean(ss))
    if so_:
        p = wilcoxon(ho, so_).pvalue if len(so_) > 5 else float("nan")
        print("\nNetSim (%d of 27 simulations with all three seeds): HetNDL %.3f, signed NOGI %.3f, W/T/L %d/%d/%d, Wilcoxon p %.3g" % (
            len(so_), np.mean(ho), np.mean(so_), *(lambda d: ((d > 1e-3).sum(), (abs(d) <= 1e-3).sum(), (d < -1e-3).sum()))(np.array(ho) - np.array(so_)), p))
    else:
        print("\nNetSim: no simulation with all three seeds yet")


if __name__ == "__main__":
    main()
