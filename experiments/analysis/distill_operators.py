"""Symbolic distillation of the coupling laws with distractor operators.

Question. Table 4 counts how many distilled coupling laws have exactly the terms of the true law (13 of 18 after the
diagonal correction of distill_diag.py). The operator library of distill_pysr.py (+ - * square cube tanh sin) contains
the primitives of every true law. Does the count hold when the library also offers operators that no true law uses?

Equations. The 18 coupling laws of the paper seeds, exactly the set and order of skeleton_match.py --full
(distill_diag.records(): every experiments/distill_sel/*/pysr.json and experiments/distill_glv_1106/*/pysr.json whose
seed is in PAPER_SEEDS), fitted on the same coupling grid of grids.npz (24 x 24 points (c_grid_xi[:, 0], c_grid_xj[:, 0]))
and the same output component (distill_pysr.coupling_output_dim of the learned coupling; = coupling_eq.output_dim).

Fit. distill_pysr.fit with unary operators square, cube, tanh, sin PLUS exp, abs, inv (1/x); everything else as the
paper's run of distill_pysr.py: binary + - *, maxsize 20, niterations 400, populations 15, population size 40,
procs 8 (multithreading), random_state 0, and the choice of the simplest equation on the Pareto front with R^2 >= 0.999
against the learned coupling (PySR's own best if none reaches it). The fitted law g is then corrected to
g'(x_i, x_j) = g(x_i, x_j) - g(x_i, x_i) and scored exactly as in distill_diag.py (corrected / score).

Scores. exact = skeleton_match's 1 % rule (terms whose RMS on the grid is below 1 % of the expression's RMS are dropped,
the remaining set of term types must equal that of the true law; coefficients are not compared); strict = the same
without dropping terms; R^2 vs truth = distill_pysr.r2_scaled (one least-squares scale alpha, no offset; the R^2 of
Table 4). The original law and its scores (pysr.json, corrected the same way) are printed alongside.
PySR with procs > 1 (multithreading) is not bit-reproducible even with a fixed random_state, so a re-run can return
other equations; the comparison is about the count.

Usage (repository root):
  PYTHONPATH=. $SR_PYTHON experiments/analysis/distill_operators.py fit <k> <n> [--niterations 400]
      fits the laws with index % n == k (0-based, order of distill_diag.records()), one JSON per law in
      experiments/distill_operators/<distill dir name>/coupling.json (skipped if present)
  PYTHONPATH=. python experiments/analysis/distill_operators.py score > results/supplement/distill_operators.txt"""
import argparse, json, os, sys, time
import numpy as np

OUT = os.environ.get("DISTILL_OPS_OUT", "experiments/distill_operators")          # override only for smoke tests
EXTENDED = ("square", "cube", "tanh", "sin", "exp", "abs", "inv")
NAMES_X = {"kuramoto": ["theta"], "a1": ["x"], "glv": ["x"], "powergrid": ["omega"], "a2": ["u", "v"]}


def _inv():
    """sympy reads PySR's inv(x) as 1/x (exp and abs are sympy names already)."""
    from experiments.analysis import skeleton_match as sm
    sm.LOC.setdefault("inv", lambda a: 1 / a)


def fit_laws(k, n, niterations):
    from experiments.analysis.distill_diag import records
    from experiments.analysis.distill_pysr import fit, coupling_output_dim, r2
    for idx, (i, seed, sysn, r, z, p) in enumerate(records()):
        if idx % n != k:
            continue
        tag = os.path.basename(os.path.dirname(p)); od = os.path.join(OUT, tag); fn = os.path.join(od, "coupling.json")
        if os.path.exists(fn):
            print("skip", tag, flush=True); continue
        os.makedirs(od, exist_ok=True)
        Xc = np.stack([z["c_grid_xi"][:, 0], z["c_grid_xj"][:, 0]], -1)
        c1 = ["theta"] if sysn == "powergrid" else NAMES_X[sysn][:1]
        cn = [c + "_i" for c in c1] + [c + "_j" for c in c1]
        cd = coupling_output_dim(z["c_learned"]); assert cd == r["coupling_eq"].get("output_dim", 0), tag
        y = z["c_learned"][:, cd]; t0 = time.time()
        eqs, pred = fit(Xc, y, cn, niterations, 8, 0, maxsize=20, unary_operators=EXTENDED)
        res = dict(index=idx, source=p, system=sysn, seed=seed, names=cn, output_dim=cd, unary_operators=list(EXTENDED),
                   niterations=niterations, procs=8, random_state=0, maxsize=20, equation=eqs[0],
                   r2_vs_learned=r2(y, pred[:, 0]), elapsed_s=time.time() - t0)
        json.dump(res, open(fn, "w"), indent=1)
        print("%2d %s: %s | R2 vs learned %.4f | %.0f s" % (idx, tag, eqs[0], res["r2_vs_learned"], res["elapsed_s"]), flush=True)


def score_all():
    _inv()
    from experiments.analysis.distill_diag import records, corrected, score
    from experiments.analysis.skeleton_match import COUP, CNAMES, SETTING
    Y = lambda b: "yes" if b else "no"
    rows = []
    for idx, (i, seed, sysn, r, z, p) in enumerate(records()):
        tag = os.path.basename(os.path.dirname(p)); fn = os.path.join(OUT, tag, "coupling.json")
        name = SETTING[i][1]; cd = r["coupling_eq"].get("output_dim", 0)
        e0 = r["coupling_eq"]["equation"]; g0, _, d0, _ = corrected(e0, CNAMES[sysn]); s0 = score(g0, sysn, z, cd)
        if not os.path.exists(fn):
            rows.append(dict(name=name, seed=seed, sysn=sysn, missing=True, d0=d0, s0=s0)); continue
        x = json.load(open(fn)); assert x["index"] == idx and x["names"] == CNAMES[sysn], tag
        g1, _, d1, _ = corrected(x["equation"], CNAMES[sysn]); s1 = score(g1, sysn, z, cd)
        rows.append(dict(name=name, seed=seed, sysn=sysn, missing=False, d0=d0, s0=s0, eq1=x["equation"], d1=d1, s1=s1,
                         sec=x["elapsed_s"], rl=x["r2_vs_learned"]))
    print("Coupling laws distilled with the unary operators extended by exp, abs and inv (1/x); everything else as the")
    print("paper (distill_pysr.py; diagonal correction and scores of distill_diag.py). exact = 1 % rule of skeleton_match.py,")
    print("strict = without dropping terms; R2 true = against the true law after one least-squares scale (Table 4).\n")
    hdr = "%-18s %4s  %-28s | %-5s %-6s %7s | %-5s %-6s %7s %9s"
    print(hdr % ("setting", "seed", "true law", "orig", "strict", "R2", "ext", "strict", "R2", "R2 learn"))
    for w in rows:
        a = w["s0"]
        if w["missing"]:
            print(hdr % (w["name"], w["seed"], COUP[w["sysn"]], Y(a["exact"]), Y(a["strict"]), "%.3f" % a["rt"], "MISSING", "", "", ""))
            continue
        b = w["s1"]
        print(hdr % (w["name"], w["seed"], COUP[w["sysn"]], Y(a["exact"]), Y(a["strict"]), "%.3f" % a["rt"],
                     Y(b["exact"]), Y(b["strict"]), "%.3f" % b["rt"], "%.4f" % w["rl"]))
        print("      original (corrected): %s" % w["d0"])
        print("      extended (corrected): %s" % w["d1"])
        print("      extended as fitted:   %s" % w["eq1"])
    done = [w for w in rows if not w["missing"]]
    n0 = sum(w["s0"]["exact"] for w in rows); n1 = sum(w["s1"]["exact"] for w in done)
    used = {op: sum(op + "(" in w["eq1"] for w in done) for op in ("exp", "abs", "inv")}
    print("\nexact (1 %% rule): original library %d / %d; extended library %d / %d (%d of 18 fitted)" % (n0, len(rows), n1, len(rows), len(done)))
    print("strict: original %d / %d; extended %d / %d" % (sum(w["s0"]["strict"] for w in rows), len(rows), sum(w["s1"]["strict"] for w in done), len(rows)))
    print("laws whose fitted equation uses a distractor operator: %s" % ", ".join("%s %d" % kv for kv in used.items()))
    print("changed exact status: %s" % ("; ".join("%s seed %d %s -> %s" % (w["name"], w["seed"], Y(w["s0"]["exact"]), Y(w["s1"]["exact"]))
                                                  for w in done if w["s0"]["exact"] != w["s1"]["exact"]) or "none"))
    print("R2 against the truth, extended library: %.3f-%.3f" % (min(w["s1"]["rt"] for w in done), max(w["s1"]["rt"] for w in done)) if done else "")
    print("fit time per law: %.0f-%.0f s" % (min(w["sec"] for w in done), max(w["sec"] for w in done)) if done else "")
    if len(done) < len(rows):
        print("\nincomplete: %d of 18 fitted" % len(done))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["fit", "score"])
    ap.add_argument("k", nargs="?", type=int, default=0); ap.add_argument("n", nargs="?", type=int, default=1)
    ap.add_argument("--niterations", type=int, default=400)
    a = ap.parse_args()
    fit_laws(a.k, a.n, a.niterations) if a.mode == "fit" else score_all()
