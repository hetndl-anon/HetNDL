"""Diagonal-corrected distilled coupling laws. Settings and reading rules fixed before running.

Question. The learned coupling of a diffusive HetNDL model is c(x_i, x_j) = G(x_i, x_j) - G(x_i, x_i), which vanishes at
x_j = x_i by construction. The PySR expression g fitted to it (distill_pysr.py) need not: Table 4's GLV-+- coupling
x_i(dx + 0.055) - 0.055 gives 0.055 x - 0.055 at x_j = x_i. This script imposes the diffusive form on the distilled law,
    g'(x_i, x_j) = g(x_i, x_j) - g(x_i, x_i)          (so g'(x, x) = 0 identically),
and scores g and g' side by side.

Equations. The 18 coupling laws of the paper seeds, i.e. exactly the set and order of skeleton_match.py --full: every
experiments/distill_sel/*/pysr.json and experiments/distill_glv_1106/*/pysr.json whose path seed is in PAPER_SEEDS (1-3),
sorted by setting (skeleton_match.SETTING) and seed; g = coupling_eq.equation, variables skeleton_match.CNAMES.

Construction of g'. g is parsed with sympy (skeleton_match.LOC for square / cube); its float constants are converted to
exact rationals (sympy.nsimplify(rational=True)) so that g(x_i, x_i) (substitute x_j -> x_i) is subtracted in exact
arithmetic: subtracting in floats leaves residues such as 7e-18 that the strict term count would read as an extra
constant term. If g(x_i, x_i) simplifies to exactly 0, g' := g and the ORIGINAL equation string is scored (results
identical to g by construction). Otherwise g' is handed to the scorers as the sympy-expanded expression with the
rationals turned back into 15-digit floats.

Scores (the same code path as distill_pysr.py / skeleton_match.py; imported, not copied), on the coupling grid of
grids.npz next to pysr.json (24 x 24 points (c_grid_xi[:, 0], c_grid_xj[:, 0])), output component
cd = coupling_eq.output_dim:
  R2 learned   distill_pysr.r2(c_learned[:, cd], eq on the grid)        (the "R^2 vs learned" of distill_pysr.py)
  R2 true      distill_pysr.r2_scaled(c_true[:, cd], eq on the grid)    (one least-squares scale alpha, no offset; the
               R^2 of Table 4 / Table S12), alpha printed
  max|g(x,x)|  max |eq| over the 24 grid points with c_grid_xi = c_grid_xj (the grid diagonal); the same for the learned
               coupling c_learned is printed once per law as a check (0 for a diffusive model)
  exact        skeleton_match.skeleton_match(eq, COUP, CNAMES, grid) (1 % rule) and without grid (strict)
Equations are evaluated with skeleton_match._eval (sympy lambdify, numpy). As a reproduction check the R^2 values of g
recomputed here are compared with pysr.json (largest deviation printed; expected < 1e-6).

Printed form of g'. sympy.simplify of g' in exact arithmetic; if g' / (x_j - x_i) cancels to an expression without that
denominator, it is printed as (x_j - x_i)*h with h simplified; then every float constant is rounded to 3 significant
digits (Table S12 style). The printed form is display only; all scores use the unrounded g'.

Reading rules (fixed before running).
  * A law "changes" when g(x_i, x_i) is not identically 0 (then its printed expression changes).
  * Table 4 = the seed-1 coupling rows of the paper: Kuramoto-+- p- 0.5, MixRelax, MixOsc N=30 (Table 4's MixOsc rows are
    the N=30 run: its FHN du/dt constants 0.33 / 0.49 are those of a2_N30 seed 1), GLV-+-, Grid-118. A Table 4 entry
    changes when its expression changes, or its R^2 against the truth changes at 3 decimals.
  * Table S12 (tab:eq-nonexact) lists the coupling laws that are NOT exact by the 1 % rule. A row leaves S12 when g' is
    exact; a row enters S12 when g is exact and g' is not; a remaining row changes when its expression or its R^2 (3
    decimals) changes. The coupling count of the paper (12 / 18 exact) is recounted with g'.
  * Self-field equations are not touched (50 equations, unchanged).
"Table S12" / "S12", here and in the printed lines, is the supplement's table of the equations that are not exact
(tab:eq-nonexact).
Ground truth (c_true, COUP) only in the scores. Writes nothing but stdout; result file:
  PYTHONPATH=. python experiments/analysis/distill_diag.py > results/supplement/distill_diag.txt
Run from the project root."""
import glob, json, os, re
import numpy as np
import sympy as sp
from experiments.analysis.paper_seeds import PAPER_SEEDS
from experiments.analysis.skeleton_match import LOC, COUP, CNAMES, SETTING, skeleton_match, _eval
from experiments.analysis.distill_pysr import r2, r2_scaled

TABLE4 = {"MixRelax", "MixOsc N=30", "Grid-118", "Kuramoto-± p⁻ 0.5", "GLV-±"}       # seed-1 coupling rows of Table 4


def records():
    """The 18 coupling laws, selected and ordered exactly as skeleton_match.full()."""
    dirs = sorted(glob.glob("experiments/distill_sel/*/pysr.json")) + sorted(glob.glob("experiments/distill_glv_1106/*/pysr.json"))
    dirs = [p for p in dirs if int(re.search(r"_seed(\d+)", p).group(1)) in PAPER_SEEDS]
    recs = []
    for p in dirs:
        r = json.load(open(p)); z = np.load(os.path.join(os.path.dirname(p), "grids.npz")); sysn = r["system"]
        ds = r["run"].split("/")[-2]; seed = int(re.search(r"_seed(\d+)", os.path.basename(r["run"])).group(1))
        recs.append((next(i for i, (k, _) in enumerate(SETTING) if ds.startswith(k)), seed, sysn, r, z, p))
    recs.sort(key=lambda t: t[:2])
    return recs


def floatify(e):
    return e.xreplace({q: sp.Float(q, 15) for q in e.atoms(sp.Rational) if not q.is_Integer})


def round3(e):
    return e.xreplace({f: sp.Float("%.3g" % float(f), 3) for f in e.atoms(sp.Float)})


def corrected(eq, names):
    """-> (g'(string scored), zero_diag, printed form of g', exact sympy g' or None)."""
    a, b = sp.Symbol(names[0]), sp.Symbol(names[1])
    g = sp.nsimplify(sp.sympify(eq, locals={**LOC, names[0]: a, names[1]: b}), rational=True)
    gd = g.subs(b, a)
    if sp.simplify(gd) == 0:
        return eq, True, sp.sstr(round3(sp.simplify(sp.sympify(eq, locals={**LOC, names[0]: a, names[1]: b})))), None
    e = g - gd
    h = sp.cancel(sp.expand(e) / (b - a))
    if not sp.denom(sp.together(h)).has(a, b):
        disp = "(%s - %s)*(%s)" % (names[1], names[0], sp.sstr(round3(floatify(sp.simplify(h)))))
    else:
        disp = sp.sstr(round3(floatify(sp.simplify(e))))
    return str(floatify(sp.expand(e))), False, disp, e


def score(eq, sysn, z, cd):
    names = CNAMES[sysn]; cols = [z["c_grid_xi"][:, 0].astype(np.float64), z["c_grid_xj"][:, 0].astype(np.float64)]
    pred = _eval(eq, names, cols); dg = cols[0] == cols[1]
    rl = r2(z["c_learned"][:, cd], pred); rt, al = r2_scaled(z["c_true"][:, cd], pred)
    cg = np.stack(cols, -1)
    return dict(rl=rl, rt=rt, alpha=al, diag=float(np.abs(pred[dg]).max()), n_diag=int(dg.sum()),
                exact=bool(skeleton_match(eq, COUP[sysn], names, grid=cg)), strict=bool(skeleton_match(eq, COUP[sysn], names)))


def main():
    recs = records()
    Y = lambda b: "yes" if b else "no"
    rows, dev = [], 0.0
    for i, seed, sysn, r, z, p in recs:
        name = SETTING[i][1]; ce = r["coupling_eq"]; cd = ce.get("output_dim", 0); eq = ce["equation"]
        s0 = score(eq, sysn, z, cd)
        dev = max(dev, abs(s0["rl"] - ce["r2_vs_learned"]), abs(s0["rt"] - ce["r2_vs_true_scaled"]))
        eq1, zero, disp, _ = corrected(eq, CNAMES[sysn])
        s1 = score(eq1, sysn, z, cd)
        learned_diag = float(np.abs(z["c_learned"][:, cd][z["c_grid_xi"][:, 0] == z["c_grid_xj"][:, 0]]).max())
        rows.append(dict(name=name, seed=seed, sysn=sysn, eq=eq, eq1=eq1, zero=zero, disp=disp, s0=s0, s1=s1, ldiag=learned_diag))

    print("Distilled coupling laws g and their diagonal-corrected forms g'(x_i, x_j) = g(x_i, x_j) - g(x_i, x_i)")
    print("18 laws of the paper seeds (as skeleton_match.py --full). R2 learned: against the learned coupling grid; R2 true:")
    print("against the true law after one least-squares scale alpha (Table 4 / S12); max|g(x,x)|: over the 24 diagonal grid")
    print("points; exact: 1 % rule (strict in the next column).\n")
    hdr = "%-18s %4s  %-4s %9s %9s %8s %11s %-5s %-6s"
    print(hdr % ("setting", "seed", "form", "R2 learn", "R2 true", "alpha", "max|g(x,x)|", "exact", "strict"))
    for w in rows:
        for tag, s in (("g", w["s0"]), ("g'", w["s1"])):
            print(hdr % (w["name"] if tag == "g" else "", w["seed"] if tag == "g" else "", tag, "%.4f" % s["rl"], "%.4f" % s["rt"],
                         "%.3f" % s["alpha"], "%.2e" % s["diag"], Y(s["exact"]), Y(s["strict"])))
        print("      g  = %s" % w["eq"])
        print("      g' = %s%s" % (w["disp"], "   (g(x_i, x_i) = 0 identically: g' = g)" if w["zero"] else ""))
        print("      learned coupling on the diagonal: max |c| = %.1e over %d points" % (w["ldiag"], w["s0"]["n_diag"]))
    print("\nreproduction: largest |R2 recomputed here - R2 in pysr.json| over the 18 laws (g): %.1e" % dev)

    # ---- reading rules
    print("\nlaws with g(x_i, x_i) not identically 0 (their expression changes):")
    for w in rows:
        if not w["zero"]:
            a, b = w["s0"], w["s1"]
            print("  %-18s seed %d: R2 learned %.4f -> %.4f, R2 true %.4f -> %.4f (alpha %.3f -> %.3f), max|g(x,x)| %.3g -> %.1e, "
                  "exact %s -> %s, strict %s -> %s" % (w["name"], w["seed"], a["rl"], b["rl"], a["rt"], b["rt"], a["alpha"], b["alpha"],
                                                     a["diag"], b["diag"], Y(a["exact"]), Y(b["exact"]), Y(a["strict"]), Y(b["strict"])))
            print("  %-18s         corrected: %s" % ("", w["disp"]))
    ch3 = lambda w: "%.3f" % w["s0"]["rt"] != "%.3f" % w["s1"]["rt"]
    print("\nTable 4 (seed-1 coupling rows):")
    for w in rows:
        if w["seed"] == 1 and w["name"] in TABLE4:
            what = [k for k, c in (("expression", not w["zero"]), ("R2 %.3f -> %.3f" % (w["s0"]["rt"], w["s1"]["rt"]), ch3(w))) if c]
            print("  %-18s %s" % (w["name"], "changes: " + ", ".join(what) if what else "unchanged"))
    print("\nTable S12 (tab:eq-nonexact), coupling rows = laws not exact by the 1 % rule:")
    for w in rows:
        e0, e1 = w["s0"]["exact"], w["s1"]["exact"]
        if e0 and e1:
            continue
        if not e0 and e1:
            st = "LEAVES S12 (g' is exact)"
        elif e0 and not e1:
            st = "ENTERS S12 (g' is not exact)"
        else:
            what = [k for k, c in (("expression -> %s" % w["disp"], not w["zero"]), ("R2 %.3f -> %.3f" % (w["s0"]["rt"], w["s1"]["rt"]), ch3(w))) if c]
            st = "stays; " + ("changes: " + "; ".join(what) if what else "unchanged")
        print("  %-18s seed %d: %s" % (w["name"], w["seed"], st))
    n0 = sum(w["s0"]["exact"] for w in rows); n1 = sum(w["s1"]["exact"] for w in rows)
    s0 = sum(w["s0"]["strict"] for w in rows); s1 = sum(w["s1"]["strict"] for w in rows)
    print("\ncoupling laws exact (1 %% rule): g %d / %d, g' %d / %d   (strict: g %d / %d, g' %d / %d)" % (n0, len(rows), n1, len(rows), s0, len(rows), s1, len(rows)))
    per = {}
    for w in rows:
        per.setdefault(w["name"], [0, 0, 0]); per[w["name"]][0] += w["s0"]["exact"]; per[w["name"]][1] += w["s1"]["exact"]; per[w["name"]][2] += 1
    print("per setting (g -> g'): " + "; ".join("%s %d -> %d of %d" % (k, *v) for k, v in per.items()))
    nz = [w for w in rows if not w["zero"]]
    print("R2 against the truth over all 18 laws: g %.3f-%.3f, g' %.3f-%.3f; non-exact laws (g'): R2 true %s" % (
        min(w["s0"]["rt"] for w in rows), max(w["s0"]["rt"] for w in rows), min(w["s1"]["rt"] for w in rows), max(w["s1"]["rt"] for w in rows),
        ", ".join("%.3f" % w["s1"]["rt"] for w in rows if not w["s1"]["exact"])))
    print("laws whose g(x, x) is not identically 0: %d of %d" % (len(nz), len(rows)))


if __name__ == "__main__":
    main()
