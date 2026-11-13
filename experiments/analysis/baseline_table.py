"""Ours (default procedure, experiments/runs/sel/selected.json from sel_tables.py) vs EVERY baseline in the harness on the
main benchmarks: NOGI (same paths as sel_vs_nogi.py), ARNI (runs/arni; on Kuramoto both the full basis and the
greedy search exist -> the better of the two per data seed by AUROC, i.e. a truth-picked advantage for the baseline),
Granger-lasso (runs/granger), NRI (runs/nri), AIDD (runs/aidd). Mean AUROC / AUPRC over data seeds; per seed ours is
compared with the BEST baseline on that seed (AUROC, |difference| <= 1e-3 = tie). Netsim: per-sim means of ours / NOGI
over the 3 training seeds, single runs for the other baselines; Wilcoxon over sims against each baseline.
BA benchmarks: ARNI / Granger / NRI / AIDD with the arguments of the matching ER benchmark (experiments/launch/
ba_baselines.py). COSINE (experiments/launch/cosine_run.py: no truth inside the run, round by its own
validation score) from runs/cosine/<dataset>/seed<k>; Netsim seed 1. A missing or failed baseline run (e.g. a COSINE crash)
is shown as "—" and left out of that seed's best baseline. MTGNN (experiments/launch/mtgnn_run.py: official code,
dense learned adjacency, W = A^T) from runs/mtgnn/<dataset>/seed<k>. SGODE (hetndl/baselines/sgode.py, our
re-implementation): per cell the variant with the lower validation loss (experiments/analysis/sgode_table.pick).
--per-seed: instead of the tables above print, for every benchmark and data seed, AUROC / AUPRC of ours and
of every baseline, with ARNI's two modes (full basis, greedy search) in separate columns, i.e. without the truth-picked
"better of the two". The default output is unchanged. run_dirs() / score_matrix() give the run directory and the edge
score matrix behind every number (used by binary_graph_table.py).
Grid-118: every neural method reports
the best of three training runs (train seeds s, 10 s + 2, 10 s + 3) by its own validation criterion, as HetNDL and NOGI
always did: NRI / AIDD val_nll_best, MTGNN val_mae_best, SGODE val_l1_best over both variants, COSINE val_score
(grid_pick(); the extra runs are in runs/grid_budget). GRID_BUDGET = False restores the single runs (for comparison).
Run from the project root with PYTHONPATH=."""
import glob, json, os, re, sys, numpy as np
from scipy.stats import wilcoxon
from hetndl.snapshot import array_path, metrics_of
from experiments.analysis.paper_seeds import PAPER_SEEDS
from experiments.analysis.sgode_table import pick as sgode_pick
R = "experiments/runs"; sel = json.load(open(os.path.join(R, "sel", "selected.json")))
BASE = ("NOGI", "ARNI", "Granger", "NRI", "AIDD", "COSINE", "MTGNN", "SGODE")


def met(path):
    p = os.path.join(R, path, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return None if m.get("failed") else m


def score(m):
    if m is None:
        return None
    if "auroc" in m and "val_mse_best" not in m:                       # baseline metrics.json (ARNI, Granger, NRI, AIDD)
        return (m["auroc"], m["auprc"])
    b = metrics_of(m); return (b["auroc"], b["auprc"])                  # NOGI: protocol snapshot


GRID_BUDGET = True
GRID_CRIT = {"nri": "val_nll_best", "aidd": "val_nll_best", "mtgnn": "val_mae_best", "sgode": "val_l1_best", "cosine": "val_score"}


def grid_candidates(method, s):
    """(run directory relative to experiments/runs, metrics or "missing" / "failed") of the three Grid-118 trainings."""
    ds = "grid118_seed%d" % s; ks = (s, 10 * s + 2, 10 * s + 3)
    if method == "sgode":
        paths = ["sgode/%s/%s/seed%d" % (v, ds, k) if k == s else "grid_budget/sgode/%s/%s/seed%d" % (v, ds, k)
                 for v in ("v1", "v3") for k in ks]
    else:
        paths = ["%s/%s/seed%d" % (method, ds, k) if k == s else "grid_budget/%s/%s/seed%d" % (method, ds, k) for k in ks]
    out = []
    for p in paths:
        f = os.path.join(R, p, "metrics.json")
        m = json.load(open(f)) if os.path.exists(f) else "missing"
        out.append((p, "failed" if isinstance(m, dict) and m.get("failed") else m))
    return out


def grid_pick(method, s):
    """Run directory of the reported Grid-118 run of a neural baseline: the lowest validation criterion over its three
    trainings (failed runs left out); with GRID_BUDGET = False the single run of seed s (SGODE: the sgode_table pick)."""
    if not GRID_BUDGET:
        if method == "sgode":
            p = sgode_pick("grid118_seed%d" % s, s)
            return p[1] if p else None
        return _ok("%s/grid118_seed%d/seed%d" % (method, s, s))
    c = grid_candidates(method, s)
    assert not any(m == "missing" for _, m in c), ("Grid-118 training missing", method, s, c)
    ok = [(m[GRID_CRIT[method]], p) for p, m in c if isinstance(m, dict) and m.get(GRID_CRIT[method]) is not None
          and np.isfinite(m[GRID_CRIT[method]])]
    return min(ok)[1] if ok else None


def nogi_grid(s):
    ms = [m for m in (met("grid/grid118_seed%d/orig_seed%d%s" % (s, s, r)) for r in ("", "_r2", "_r3")) if m]
    return min(ms, key=lambda m: m["val_mse_best"]) if ms else None


def one(base, ds):
    if ds.startswith("grid118_seed") and base in ("nri", "aidd", "mtgnn"):
        p = grid_pick(base, int(ds[len("grid118_seed"):]))
        return score(met(p)) if p else None
    ps = sorted(glob.glob(os.path.join(R, base, ds, "*", "metrics.json")))
    if base == "arni":
        c = [score(json.load(open(p))) for p in ps]
        c = [x for x in c if x]
        return max(c) if c else None
    return score(json.load(open(ps[0]))) if ps else None


def sgode(ds, s):
    if ds.startswith("grid118_seed"):
        p = grid_pick("sgode", s)
        return score(met(p)) if p else None
    p = sgode_pick(ds, s)
    return score(p[2]) if p else None


def cosine(ds, s):
    p = grid_pick("cosine", s) if ds.startswith("grid118_seed") else "cosine/%s/seed%d" % (ds, s)
    return score(met(p)) if p else None


def nogi_path(lab, ds, s):
    if lab.startswith("Kuramoto"):
        c = re.sub(r"_g\d+$", "", ds)
        return ("motiv_trig/%s/orig_seed1" % c) if s == 1 else ("multiseed/%s/orig_seed%d" % (ds, s))
    if lab.startswith("A1"):
        return "a1v2/a1v2_N30_seed1/orig_seed1" if s == 1 else "multiseed/a1v2_N30_seed%d/orig_seed%d" % (s, s)
    d = {"A2 N30": "a2", "A2 N60": "a2", "A3 (edge types)": "a3", "GLV (signed)": "glv"}.get(lab)
    if d:
        return "%s/%s/orig_seed%d" % (d, ds, s)
    if lab.startswith("BA"):
        return "ba/%s/orig_seed%d" % (ds, s)
    return None


B = []
for c, lab in (("s0.3_n0", "Kuramoto σ/K 0.15"), ("s2_n0", "Kuramoto σ/K 1.0"), ("s0.3_n0.2", "Kuramoto p⁻ 0.2"), ("s0.3_n0.5", "Kuramoto p⁻ 0.5")):
    B.append((lab, [("kura_ER100_K2_%s%s" % (c, "" if s == 1 else "_g%d" % s), s) for s in (1, 2, 3)]))
B += [("A1 (N30)", [("a1v2_N30_seed%d" % s, s) for s in (1, 2, 3)]), ("A2 N30", [("a2_N30_seed%d" % s, s) for s in PAPER_SEEDS]),
      ("A2 N60", [("a2_N60_seed%d" % s, s) for s in (1, 2, 3)]), ("A3 (edge types)", [("a3_N30_seed%d" % s, s) for s in (1, 2, 3)]),
      ("GLV (signed)", [("glv_N30_n0.5_seed%d" % s, s) for s in (1, 2, 3)]), ("IEEE-118 grid", [("grid118_seed%d" % s, s) for s in (1, 2, 3)]),
      ("BA Kuramoto p⁻ 0.5", [("kura_BA100_m2_K2_s0.3_n0.5_g%d" % s, s) for s in (1, 2, 3)]), ("BA A1", [("a1v2_BA30_m2_seed%d" % s, s) for s in (1, 2, 3)])]


# ---- per-seed numbers and the score matrix behind every column
PER_SEED = ("NOGI", "ARNI full", "ARNI greedy", "Granger", "NRI", "AIDD", "COSINE", "MTGNN", "SGODE")
PAPER_NAME = {"Kuramoto σ/K 0.15": "Kuramoto-ω σ/κ 0.15", "Kuramoto σ/K 1.0": "Kuramoto-ω σ/κ 1.0", "Kuramoto p⁻ 0.2": "Kuramoto-± p⁻ 0.2",
              "Kuramoto p⁻ 0.5": "Kuramoto-± p⁻ 0.5", "BA Kuramoto p⁻ 0.5": "Kuramoto-± (BA)", "A1 (N30)": "MixRelax", "BA A1": "MixRelax (BA)",
              "A2 N30": "MixOsc N=30", "A2 N60": "MixOsc N=60", "A3 (edge types)": "MixOsc-E N=30", "GLV (signed)": "GLV-±",
              "IEEE-118 grid": "Grid-118"}                                   # row names of the paper's main table


def sel_key(lab, ds, s):
    return ("ba/%s/add_z_signed_diff_lz0_seed%d" if lab.startswith("BA") else "%s/add_z_signed_diff_lz0_seed%d") % (ds, s)


def _ok(path):
    return path if path is not None and met(path) is not None else None


def run_dirs(lab, ds, s):
    """Run directory (relative to experiments/runs) of every baseline for one benchmark seed, None where the run is missing
    or failed: the same runs as the default table, except that ARNI's two modes are listed separately instead of the
    better one by AUROC."""
    if lab.startswith("IEEE"):
        c = [p for p in ("grid/grid118_seed%d/orig_seed%d%s" % (s, s, r) for r in ("", "_r2", "_r3")) if met(p)]
        nogi = min(c, key=lambda p: met(p)["val_mse_best"]) if c else None
    else:
        nogi = _ok(nogi_path(lab, ds, s))

    def first(pattern):
        ps = sorted(glob.glob(os.path.join(R, pattern, "metrics.json")))
        return _ok(os.path.relpath(os.path.dirname(ps[0]), R)) if ps else None
    return {"NOGI": nogi, "ARNI full": first("arni/%s/full_seed*" % ds), "ARNI greedy": first("arni/%s/greedy_seed*" % ds),
            "Granger": first("granger/%s/*" % ds), "NRI": first("nri/%s/*" % ds), "AIDD": first("aidd/%s/*" % ds),
            "COSINE": _ok("cosine/%s/seed%d" % (ds, s)), "MTGNN": first("mtgnn/%s/*" % ds),
            "SGODE": (sgode_pick(ds, s) or (None, None))[1]} if not lab.startswith("IEEE") else {
            "NOGI": nogi, "ARNI full": first("arni/%s/full_seed*" % ds), "ARNI greedy": first("arni/%s/greedy_seed*" % ds),
            "Granger": first("granger/%s/*" % ds), "NRI": grid_pick("nri", s), "AIDD": grid_pick("aidd", s),
            "COSINE": grid_pick("cosine", s), "MTGNN": grid_pick("mtgnn", s), "SGODE": grid_pick("sgode", s)}


def score_matrix(path):
    """Edge score matrix W[j, i] of a run directory given relative to experiments/runs, the matrix whose AUROC / AUPRC the
    tables report: W.npy for the baselines (MTGNN: W + W^T, the symmetrised scoring of mtgnn_rescore.py recorded in its
    metrics.json), the protocol snapshot (hetndl.snapshot) for NOGI and for our runs."""
    m = met(path); p = os.path.join(R, path)
    W = np.load(os.path.join(p, "W.npy") if "val_mse_best" not in m else array_path(p, "W", m))
    return W + W.T if str(m.get("scoring", "")).startswith("symmetrised") else W


def per_seed():
    f = lambda x: "%.3f / %.3f" % x if x is not None else "      —      "
    print("AUROC / AUPRC per data seed; ARNI full = full basis, ARNI greedy = greedy search (run on Kuramoto only); pick = the "
          "candidate chosen by validation MSE (n = non-negative, s = signed); — = no run or a failed run")
    print("%-19s %4s %4s  %-13s  %s" % ("benchmark", "seed", "pick", "ours", "  ".join("%-13s" % b for b in PER_SEED)))
    for lab, cells in B:
        rows = []
        for ds, s in cells:
            g = sel_key(lab, ds, s)
            if g not in sel:
                continue
            rd = run_dirs(lab, ds, s)
            o = (sel[g]["auroc"], sel[g]["auprc"]); bs = [score(met(rd[b])) if rd[b] else None for b in PER_SEED]
            rows.append((o, bs))
            print("%-19s %4d %4s  %s  %s" % (lab, s, "n" if sel[g]["variant"] == "nonneg" else "s", f(o), "  ".join(f(x) for x in bs)))
        if rows:
            cols = []
            for k in range(len(PER_SEED)):
                v = [r[1][k] for r in rows if r[1][k] is not None]
                cols.append(f(tuple(np.mean(v, 0))) if v else f(None))
            print("%-19s %4s %4s  %s  %s\n" % (lab, "mean", "", f(tuple(np.mean([r[0] for r in rows], 0))), "  ".join(cols)))


def main():
    print("%-19s %3s  %-13s %s  %-9s %s" % ("benchmark", "n", "ours", "  ".join("%-13s" % b for b in BASE), "W/T/L", "best baseline per seed"))
    tot = np.zeros(3, int)
    for lab, cells in B:
        rows = []
        for ds, s in cells:
            g = ("ba/%s/add_z_signed_diff_lz0_seed%d" if lab.startswith("BA") else "%s/add_z_signed_diff_lz0_seed%d") % (ds, s)
            if g not in sel:
                continue
            o = (sel[g]["auroc"], sel[g]["auprc"])
            nm = nogi_grid(s) if lab.startswith("IEEE") else met(nogi_path(lab, ds, s))
            bs = [score(nm)] + [one(b, ds) for b in ("arni", "granger", "nri", "aidd")] + [cosine(ds, s), one("mtgnn", ds), sgode(ds, s)]
            rows.append((o, bs))
        if not rows:
            continue
        o = np.array([r[0] for r in rows])
        cols = []
        for k in range(len(BASE)):
            v = np.array([r[1][k] for r in rows if r[1][k] is not None])
            cols.append("%.3f / %.3f" % tuple(v.mean(0)) if len(v) else "     —       ")
        best = [max((x for x in r[1] if x is not None), key=lambda x: x[0]) for r in rows]
        who = [BASE[[x for x in r[1]].index(b)] for r, b in zip(rows, best)]
        d = o[:, 0] - np.array([b[0] for b in best]); wtl = ((d > 1e-3).sum(), (abs(d) <= 1e-3).sum(), (d < -1e-3).sum()); tot += wtl
        print("%-19s %3d  %.3f / %.3f %s  %d/%d/%d     %s" % (lab, len(rows), *o.mean(0), "  ".join(cols), *wtl, ",".join(who)))
    print("synthetic total vs the best baseline per seed: W/T/L %d/%d/%d" % tuple(tot))

    # sign accuracy on the true edges (benchmarks with negative edges); only ours and ARNI output signs (NOGI is non-negative,
    # Granger / NRI / AIDD / COSINE give no edge sign). ARNI: the run picked as in its AUROC column; full mode reports the
    # Jacobian sign (sign_acc_jac, needed for GLV's multiplicative x_i x_j term), greedy mode the coefficient sign.
    def arni_sign(ds):
        c = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, "arni", ds, "*", "metrics.json")))]
        c = [m for m in c if not m.get("failed")]
        if not c:
            return None
        m = max(c, key=lambda m: m["auroc"]); j = m.get("sign_acc_jac")
        return j if j is not None and np.isfinite(j) else m.get("sign_acc")


    print("\nsign accuracy on true edges (signed benchmarks): ours | ARNI, per seed")
    for lab, cells in B:
        if not ("p⁻" in lab or lab.startswith("GLV")):
            continue
        o, a_ = [], []
        for ds, s in cells:
            g = ("ba/%s/add_z_signed_diff_lz0_seed%d" if lab.startswith("BA") else "%s/add_z_signed_diff_lz0_seed%d") % (ds, s)
            if g in sel and sel[g]["sign"] is not None:
                o.append(sel[g]["sign"])
            x = arni_sign(ds)
            if x is not None:
                a_.append(x)
        print("  %-19s ours %.3f (%s) | ARNI %.3f (%s)" % (lab, np.mean(o), " ".join("%.3f" % v for v in o), np.mean(a_), " ".join("%.3f" % v for v in a_)))

    # Netsim
    ours, nogi = {}, {}
    for g, s in sel.items():
        mm = re.match(r"netsim_sim(\d+)/.*_seed(\d+)$", g)
        if mm:
            k, sd = int(mm.group(1)), int(mm.group(2)); ours.setdefault(k, []).append((s["auroc"], s["auprc"]))
            x = score(met("netsim_rk4sub/netsim_sim%d/orig_seed%d" % (k, sd)))
            if x:
                nogi.setdefault(k, []).append(x)
    ks = [k for k in sorted(ours) if len(ours[k]) == 3]
    a = np.array([np.mean(ours[k], 0) for k in ks])
    print("\nNetsim (%d sims): ours %.3f / %.3f" % (len(ks), *a.mean(0)))
    for name, get in (("NOGI", lambda k: tuple(np.mean(nogi[k], 0)) if len(nogi.get(k, [])) == 3 else None),
                      ("ARNI", lambda k: one("arni", "netsim_sim%d" % k)), ("Granger", lambda k: one("granger", "netsim_sim%d" % k)),
                      ("NRI", lambda k: one("nri", "netsim_sim%d" % k)), ("AIDD", lambda k: one("aidd", "netsim_sim%d" % k)),
                      ("COSINE", lambda k: score(met("cosine/netsim_sim%d/seed1" % k))), ("MTGNN", lambda k: one("mtgnn", "netsim_sim%d" % k)),
                      ("SGODE", lambda k: sgode("netsim_sim%d" % k, 1))):
        pairs = [(a[i], get(k)) for i, k in enumerate(ks) if get(k) is not None]
        if not pairs:
            print("  vs %-8s no runs" % name); continue
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs]); d = x[:, 0] - y[:, 0]
        print("  vs %-8s n %2d  %.3f / %.3f   W/T/L %d/%d/%d   Wilcoxon p AUROC %.2g  AUPRC %.2g" % (
            name, len(pairs), *y.mean(0), (d > 1e-3).sum(), (abs(d) <= 1e-3).sum(), (d < -1e-3).sum(),
            wilcoxon(x[:, 0], y[:, 0]).pvalue, wilcoxon(x[:, 1], y[:, 1]).pvalue))


if __name__ == "__main__":
    per_seed() if "--per-seed" in sys.argv else main()
