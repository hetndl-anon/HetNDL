"""Table 4 column "skeleton = truth". A distilled equation matches the true one when, after sympy expansion,
it is a sum of exactly the true basis terms, with free constants (coefficients are not compared — they are scored by
R^2 in distill_pysr.py). Additive terms whose RMS on the sampled grid is below `tol` (1 %, fixed before scoring) of the
whole expression's RMS are treated as numerical noise; the strict count without that rule is printed as well. Symbolic
constants (c0, c1, ...) of a shared skeleton count as free constants. Non-polynomial terms must match exactly
(tanh(tanh(x)) is not tanh(x); tanh(2.3 x) is not tanh(x)).
Inputs: every experiments/distill_sel/*/pysr.json (A1, A2, Kuramoto, Grid-118) and experiments/distill_glv_1106/*/pysr.json
(GLV), with the grids of the same directory. Truth only in the scores. Run from the project root (hetndl env).

--full: instead of the summary, list every distilled equation of the paper seeds (18 coupling laws, 53 self-field
equations): setting, seed, read-out group and the true class it is scored against, coordinate, the equation, "exact",
and R^2 against the truth; then the ranges of the grids the equations were fitted and scored on, and the counts.
Self-field equations come from the per-node distillation (pernode_export.py -> pernode_distill.py,
experiments/distill_pernode/<tag>/equations.json): one form per read-out class and state coordinate, constants per node,
on every benchmark (Kuramoto-± and GLV-± included, one read-out class each). The default summary (no option) reads the
class-mean self-field equations of distill_pysr.py, which the paper does not report.
What the two scores are, exactly as computed:
  exact  the 1 % rule above: the equation is expanded with sympy and split into additive terms; terms whose RMS on the
         grid is below 1 % of the RMS of the whole expression are dropped; the remaining set of term types (terms
         without their numeric coefficients) must equal the set of the true law. Coefficient values are not compared.
         "strict" is the same test without dropping any term. Self-fields: tested on the form with every constant at
         its median over the nodes of the class (as stored in equations.json).
  R^2    Self-field: plain R^2 = 1 - SS_res / SS_tot of the per-node equations (each node with its own constants) against
         the TRUE per-node functions on the self-field grid, pooled over the nodes of the class, with NO fitted scale or
         offset (equations.json r2_true_nodes, recomputed here from the skeleton and the per-node constants). Coupling:
         R^2 AFTER fitting one scalar alpha by least squares, truth ~ alpha * equation (r2_vs_true_scaled), because the
         model fixes the coupling law only up to the scale it shares with W; alpha is printed. No offset is fitted."""
import glob, json, os, re, sys
import numpy as np
import sympy as sp
from experiments.analysis.paper_seeds import PAPER_SEEDS
LOC = {"square": lambda a: a ** 2, "cube": lambda a: a ** 3}


def _parse(s, names):
    syms = {n: sp.Symbol(n) for n in names}
    e = sp.sympify(s, locals={**LOC, **syms})
    e = e.subs({c: 1 for c in e.free_symbols if c.name not in names})          # symbolic constants are free constants
    return sp.expand(e), [syms[n] for n in names]


def term_types(s, names, grid=None, tol=0.01):
    """Set of additive term types (term without its numeric coefficient; 1 for a constant)."""
    e, syms = _parse(s, names)
    terms = list(sp.Add.make_args(e))
    if grid is not None:
        cols = [np.asarray(grid, float)[:, k] for k in range(len(syms))]
        val = lambda t: np.broadcast_to(np.asarray(sp.lambdify(syms, t, "numpy")(*cols), float), (len(cols[0]),))
        tot = np.sqrt(np.mean(val(e) ** 2))
        terms = [t for t in terms if np.sqrt(np.mean(val(t) ** 2)) >= tol * tot]
    return frozenset(sp.srepr(t.as_coeff_Mul()[1]) for t in terms)


def skeleton_match(found, truth, names, grid=None, tol=0.01):
    return term_types(found, names, grid, tol) == term_types(truth, names)


NAMES = {"a1": ["x"], "glv": ["x"], "a2": ["u", "v"], "kuramoto": ["theta"], "powergrid": ["omega"]}
CNAMES = {"a1": ["x_i", "x_j"], "glv": ["x_i", "x_j"], "a2": ["u_i", "u_j"], "kuramoto": ["theta_i", "theta_j"], "powergrid": ["theta_i", "theta_j"]}
SELF = {"a1": {0: ["-x"], 1: ["-x - 0.5*x**3"], 2: ["-1.2*x + 0.6*tanh(x)"]},
        "a2": {0: ["u - u**3/3 - v + 0.5", "0.08*(u + 0.7 - 0.8*v)"], 1: ["v", "1.0*(1 - u**2)*v - u"]},   # FHN, Van der Pol
        "powergrid": {0: ["omega", "0.3 - 2.5*omega"], 1: ["omega", "0.3 - 0.5*omega"]},                 # load, generator
        "glv": "0.9*x - 0.8*x**2"}                                                                  # r_i x - q_i x^2 (diffusive)
COUP = {"a1": "tanh(x_j) - tanh(x_i)", "a2": "u_j - u_i", "kuramoto": "sin(theta_j - theta_i)",
        "powergrid": "sin(theta_j - theta_i)", "glv": "x_i*x_j - x_i**2"}


def main():
    dirs = sorted(glob.glob("experiments/distill_sel/*/pysr.json")) + sorted(glob.glob("experiments/distill_glv_1106/*/pysr.json"))
    dirs = [p for p in dirs if int(re.search(r"_seed(\d+)", p).group(1)) in PAPER_SEEDS]          # paper seeds only
    tab, per = {}, {}
    print("per equation: [match with 1 % rule | strict] found  ->  true")
    for p in dirs:
        r = json.load(open(p)); z = np.load(os.path.join(os.path.dirname(p), "grids.npz")); sysn = r["system"]
        t = tab.setdefault(sysn, dict(self=[0, 0, 0], coup=[0, 0, 0], runs=0)); t["runs"] += 1
        tag = os.path.basename(os.path.dirname(p))
        if sysn in ("a1", "a2", "powergrid"):
            for st in r["self_terms"]:
                for dim, (eq, tr) in enumerate(zip(st["equations"], SELF[sysn][st["true_class"]])):
                    m = skeleton_match(eq, tr, NAMES[sysn], grid=z["grid"]); s = skeleton_match(eq, tr, NAMES[sysn])
                    t["self"][0] += m; t["self"][1] += s; t["self"][2] += 1
                    k = (sysn, st["true_class"], dim); per.setdefault(k, [0, 0]); per[k][0] += m; per[k][1] += 1
                    print("  %s %s | class %d dim %d [%s|%s] %s  ->  %s" % (sysn, tag[-24:], st["true_class"], dim, "Y" if m else "n", "Y" if s else "n", eq, tr))
        elif sysn == "glv":
            sk = r["self_terms"]["shared_skeleton"]; m = skeleton_match(sk, SELF["glv"], ["x"])
            t["self"][0] += m; t["self"][1] += m; t["self"][2] += 1
            print("  glv %s | shared skeleton [%s] %s  ->  %s" % (tag[-24:], "Y" if m else "n", sk, SELF["glv"]))
        cg = np.stack([z["c_grid_xi"][:, 0], z["c_grid_xj"][:, 0]], -1); eq = r["coupling_eq"]["equation"]
        m = skeleton_match(eq, COUP[sysn], CNAMES[sysn], grid=cg); s = skeleton_match(eq, COUP[sysn], CNAMES[sysn])
        t["coup"][0] += m; t["coup"][1] += s; t["coup"][2] += 1
        print("  %s %s | coupling [%s|%s] %s  ->  %s" % (sysn, tag[-24:], "Y" if m else "n", "Y" if s else "n", eq, COUP[sysn]))
    print("\nTable 4 column (1 %% rule; strict in brackets)\n%-10s %5s  %-22s %-22s" % ("system", "runs", "self-term equations", "coupling"))
    for sysn, t in tab.items():
        self_s = "constant per node" if sysn == "kuramoto" else "%d / %d  (%d)" % (t["self"][0], t["self"][2], t["self"][1])
        print("%-10s %5d  %-22s %d / %d  (%d)" % (sysn, t["runs"], self_s, t["coup"][0], t["coup"][2], t["coup"][1]))
    LAB = {("a1", 0): "f = -x", ("a1", 1): "f = -x - 0.5x^3", ("a1", 2): "f = -1.2x + 0.6 tanh x", ("a2", 0): "FHN", ("a2", 1): "Van der Pol",
           ("powergrid", 0): "load", ("powergrid", 1): "generator"}
    print("\nper class and equation (1 % rule; a class can be split into several learned clusters):")
    for (sysn, c, dim), (m, n) in sorted(per.items()):
        print("  %-10s %-24s dim %d  %d / %d" % (sysn, LAB[(sysn, c)], dim, m, n))


# ---- --full: every equation with its two scores, and the grid ranges
SETTING = (("a1v2_N30", "MixRelax"), ("a2_N30", "MixOsc N=30"), ("a2_N60", "MixOsc N=60"), ("grid118", "Grid-118"),
           ("kura_ER100_K2_s0.3_n0.5", "Kuramoto-± p⁻ 0.5"), ("glv_N30_n0.5", "GLV-±"))
CLASS = {("a1", 0): "f = -x", ("a1", 1): "f = -x - 0.5x^3", ("a1", 2): "f = -1.2x + 0.6 tanh x", ("a2", 0): "FHN", ("a2", 1): "Van der Pol",
         ("powergrid", 0): "load", ("powergrid", 1): "generator", ("kuramoto", 0): "omega_i", ("glv", 0): "r_i x - (1 - s_i) x^2"}
COORD = {"a1": ["dx/dt"], "glv": ["dx/dt"], "a2": ["du/dt", "dv/dt"], "powergrid": ["dtheta/dt", "domega/dt"], "kuramoto": ["dtheta/dt"]}
PERNODE = "experiments/distill_pernode"          # self-field equations of the paper (pernode_distill.py)


def _display(eq):
    """Median-constant equation -> a readable string (sympy's own simplification), with a time guard for display only."""
    import signal

    def h(*_):
        raise TimeoutError
    old = signal.signal(signal.SIGALRM, h); signal.alarm(20)
    try:
        return str(sp.sympify(eq))
    except TimeoutError:
        return eq
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)


def _pernode_r2(cl, names, grid, T):
    """Pooled R^2 of the per-node equations (skeleton + each node's constants) against T (n, P), recomputed."""
    syms = [sp.Symbol(n) for n in names]; P = np.asarray(cl["constants"], float).reshape(len(T), -1)
    cs = sp.symbols("c0:%d" % P.shape[1]) if P.shape[1] else ()
    sk = sp.sympify(cl["skeleton"], locals={**{n: s for n, s in zip(names, syms)}, **{c.name: c for c in cs}}, evaluate=False)
    f = sp.lambdify(syms + list(cs), sk, "numpy"); cols = [grid[:, k] for k in range(grid.shape[1])]
    Y = np.stack([np.broadcast_to(np.asarray(f(*cols, *p), float), (len(grid),)) for p in P])
    return _r2(T.ravel(), Y.ravel())


def _eval(eq, names, cols):
    e = sp.sympify(eq, locals={**LOC, **{n: sp.Symbol(n) for n in names}})
    return np.broadcast_to(np.asarray(sp.lambdify([sp.Symbol(n) for n in names], e, "numpy")(*cols), float), (len(cols[0]),))


def _r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


def full():
    dirs = sorted(glob.glob("experiments/distill_sel/*/pysr.json")) + sorted(glob.glob("experiments/distill_glv_1106/*/pysr.json"))
    dirs = [p for p in dirs if int(re.search(r"_seed(\d+)", p).group(1)) in PAPER_SEEDS]
    recs = []
    for p in dirs:
        r = json.load(open(p)); z = np.load(os.path.join(os.path.dirname(p), "grids.npz")); sysn = r["system"]
        ds = r["run"].split("/")[-2]; seed = int(re.search(r"_seed(\d+)", os.path.basename(r["run"])).group(1))
        recs.append((next(i for i, (k, _) in enumerate(SETTING) if ds.startswith(k)), seed, sysn, r, z))
        r["_tag"] = os.path.basename(os.path.dirname(p))
    recs.sort(key=lambda t: t[:2])
    Y = lambda b: "yes" if b else "no"
    cnt = {}; dev = 0.0; rng = {}
    # self-fields: the per-node distillation (pernode_distill.py): one form per read-out class and coordinate, constants
    # per node; exact / strict as stored there (term_types on the form with class-median constants)
    print("self-field equations (one form per read-out class and coordinate, constants per node; R^2: plain, against the true")
    print("per-node functions pooled over the class, each node with its own constants; equation shown with class-median constants)")
    print("%-18s %4s  %-5s %-22s %-10s %-5s %-6s %7s  %s" % ("setting", "seed", "group", "scored against", "coordinate", "exact", "strict", "R2", "distilled equation"))
    for i, seed, sysn, r, z in recs:
        name = SETTING[i][1]; c = cnt.setdefault(name, dict(self=[0, 0], coup=[0, 0])); g = z["grid"]
        fj = json.load(open(os.path.join(PERNODE, r["_tag"], "equations.json"))); zp = np.load(os.path.join(PERNODE, r["_tag"], "data.npz"))
        assert np.array_equal(zp["grid"], g), r["_tag"]
        for cl in fj["classes"]:
            mem = zp["labels"] == cl["cluster"]; dim = cl["coord"]
            rec = _pernode_r2(cl, NAMES[sysn], g, zp["T_node"][mem, :, dim])
            dev = max(dev, abs(rec - cl["r2_true_nodes"])); c["self"][0] += cl["exact"]; c["self"][1] += 1
            print("%-18s %4d  %-5d %-22s %-10s %-5s %-6s %7.3f  %s" % (
                name, seed, cl["cluster"], CLASS[(sysn, cl["true_class"])], COORD[sysn][dim], Y(cl["exact"]), Y(cl["strict"]),
                cl["r2_true_nodes"], _display(cl["median_equation"])))
        rr = rng.setdefault(name, dict(g=[], c=[], n=set())); rr["g"].append((g.min(0), g.max(0))); rr["n"].add(len(g))
        rr["c"].append((min(z["c_grid_xi"][:, 0].min(), z["c_grid_xj"][:, 0].min()), max(z["c_grid_xi"][:, 0].max(), z["c_grid_xj"][:, 0].max())))
    print("\ncoupling laws (R^2: after one fitted scalar alpha, truth ~ alpha * equation; no offset)")
    print("%-18s %4s  %-28s %-5s %-6s %7s %8s  %s" % ("setting", "seed", "true law", "exact", "strict", "R2", "alpha", "distilled equation"))
    for i, seed, sysn, r, z in recs:
        name = SETTING[i][1]; ce = r["coupling_eq"]; eq = ce["equation"]; cols = [z["c_grid_xi"][:, 0], z["c_grid_xj"][:, 0]]
        cg = np.stack(cols, -1); m = skeleton_match(eq, COUP[sysn], CNAMES[sysn], grid=cg); s_ = skeleton_match(eq, COUP[sysn], CNAMES[sysn])
        y = z["c_true"][:, ce.get("output_dim", 0)]; yh = _eval(eq, CNAMES[sysn], cols); al = float(yh @ y / max(yh @ yh, 1e-12))
        dev = max(dev, abs(_r2(y, al * yh) - ce["r2_vs_true_scaled"])); cnt[name]["coup"][0] += m; cnt[name]["coup"][1] += 1
        print("%-18s %4d  %-28s %-5s %-6s %7.3f %8.3f  %s" % (name, seed, COUP[sysn], Y(m), Y(s_), ce["r2_vs_true_scaled"], ce["scale_alpha"], eq))
    print("\nlargest |stored R^2 - R^2 recomputed| (self-fields: equations.json vs skeleton + per-node constants; coupling: pysr.json vs equation string and grids.npz): %.1e" % dev)
    print("\ngrid ranges (min and max over the listed seeds of the lower and upper grid ends)")
    for name, rr in rng.items():
        lo = np.array([a for a, _ in rr["g"]]); hi = np.array([b for _, b in rr["g"]]); cl = [a for a, _ in rr["c"]]; ch = [b for _, b in rr["c"]]
        sysn = next(s for i, _, s, _, _ in recs if SETTING[i][1] == name)
        var = {"a1": ["x"], "glv": ["x"], "a2": ["u", "v"], "kuramoto": ["theta"], "powergrid": ["omega"]}[sysn]
        sf = "; ".join("%s lower %.3f..%.3f upper %.3f..%.3f" % (v, lo[:, k].min(), lo[:, k].max(), hi[:, k].min(), hi[:, k].max()) for k, v in enumerate(var))
        note = {"kuramoto": " (uniform on the circle)",
                "powergrid": " (2-98 % of the observed omega; theta fixed at each node's mean phase)",
                "a2": " (5-95 % per coordinate)"}.get(sysn, " (5-95 % of the observed states)")
        print("  %-18s self-field grid, %s points: %s%s" % (name, "/".join(str(n) for n in sorted(rr["n"])), sf, note))
        print("  %-18s coupling grid, 24 x 24 points in (%s, %s): lower %.3f..%.3f upper %.3f..%.3f%s" % (
            "", CNAMES[sysn][0], CNAMES[sysn][1], min(cl), max(cl), min(ch), max(ch),
            "" if sysn == "kuramoto" else " (5-95 % of the first state coordinate; other coordinates at their median)"))
    print("\ncounts, exact by the 1 % rule")
    for name, c in cnt.items():
        print("  %-18s self-field %s   coupling %d / %d" % (name, "%d / %d" % tuple(c["self"]) if c["self"][1] else "(node constants)", *c["coup"]))
    fam = lambda keys: tuple(np.sum([cnt[k]["self"] for k in keys if k in cnt], 0))
    print("  total: self-field %d / %d (MixRelax %d / %d, MixOsc %d / %d, Grid-118 %d / %d, Kuramoto-± %d / %d, GLV-± %d / %d); coupling %d / %d" % (
        *np.sum([c["self"] for c in cnt.values()], 0), *fam(["MixRelax"]), *fam(["MixOsc N=30", "MixOsc N=60"]), *fam(["Grid-118"]), *fam(["Kuramoto-± p⁻ 0.5"]), *fam(["GLV-±"]),
        *np.sum([c["coup"] for c in cnt.values()], 0)))


if __name__ == "__main__":
    full() if "--full" in sys.argv else main()
