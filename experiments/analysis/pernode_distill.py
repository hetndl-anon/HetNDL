"""Per-node symbolic distillation of the self-fields, stage 2 of 2 (env with PySR 2.7.2); stage 1 is pernode_export.py.
Sect. 4.7 of the paper and Sect. S10 of the supplement; Table 4 (self-field rows), Table S12 and the count "40 of 53"
come from the files written here (through skeleton_match.py --full and supp_tables.py).

Per read-out class and state coordinate (the same procedure on every benchmark; truth only in the scores):
  1 candidate nodes: the class members ordered by their first latent coordinate z_0, at the quantiles 0, .25, .5, .75
    and 1 (all members if fewer than five);
  2 PySR on each candidate node's learned field (all coordinates in one fit): operators + - *, square, cube, tanh, sin;
    maxsize 25; 400 iterations; 15 populations of 40; deterministic (serial search, random_state 0). Every equation on
    the coordinate's Pareto front is a candidate form; a form ("skeleton") is the PySR expression with every numeric
    constant replaced by a free constant, built WITHOUT sympy's automatic evaluation (as PySR's own export does), so
    nothing else is rewritten and deeply nested expressions parse instantly; candidates are de-duplicated by skeleton;
  3 the constants of each skeleton are fitted by least squares on every node of the class (start at the candidate's
    values, at most 2000 evaluations);
  4 choice: the skeleton with the fewest operations (sympy count_ops; ties: the higher R^2) whose R^2 against the
    learned fields, pooled over the nodes of the class, is >= 0.999; if none reaches it, PySR's own rule for its choice
    (model_selection="best" in pysr/sr.py) applied to the pooled candidates: complexity of a skeleton = the smallest
    PySR complexity with which it appears on the five fronts, loss = pooled MSE after the refit, Pareto front over
    (complexity, loss), score = -log(L_k / L_{k-1}) / (c_k - c_{k-1}), highest score among losses <= 1.5 x the smallest.
    Skeletons are refitted in increasing count_ops and the search stops after the first group of equal count_ops that
    contains one reaching 0.999 (a later skeleton cannot be chosen); without one, every skeleton is refitted.
Scores: exact = skeleton_match.term_types with the 1 % rule, on the skeleton with each constant at its median over the
nodes of the class, against the true law of the matched true class (Kuramoto-±: a constant; GLV-±: terms x and x^2);
strict = the same without dropping terms; R^2 of the per-node equations against the learned and against the TRUE
per-node fields, pooled over the class, with no fitted scale or offset.
Usage, from the repository root:
  PYTHONPATH=. $SR_PYTHON experiments/analysis/pernode_distill.py "<tag glob>"     (e.g. "*" for all 18 runs)
  PYTHONPATH=. $SR_PYTHON experiments/analysis/pernode_distill.py --summary
One run takes 5-70 minutes on one CPU core (5 PySR searches per class); runs are independent and can be started in
parallel, one process per tag. Output: experiments/distill_pernode/<tag>/{fronts.json, equations.json}; fronts.json is
reused when present, so the choice can be recomputed without PySR."""
import glob, json, os, signal, sys, time, warnings, numpy as np
import sympy as sp
from scipy.optimize import least_squares
from experiments.analysis.distill_pysr import r2, UNARY
from experiments.analysis.skeleton_match import term_types, SELF

warnings.filterwarnings("ignore")
LOC = {"square": lambda a: a ** 2, "cube": lambda a: a ** 3}
TRUE = {"a1": lambda c, j: SELF["a1"][c][0], "a2": lambda c, j: SELF["a2"][c][j], "powergrid": lambda c, j: SELF["powergrid"][c][j],
        "kuramoto": lambda c, j: "1.0", "glv": lambda c, j: SELF["glv"]}
TOL = 0.999


def skeleton(eq, syms):
    """PySR expression -> (skeleton with free constants c0.., the constants, their PySR values); no automatic evaluation."""
    with sp.evaluate(False):
        expr = sp.sympify(eq, locals={**LOC, **{s.name: s for s in syms}}, evaluate=False)
        floats = sorted(expr.atoms(sp.Float), key=lambda f: float(f))
        cs = sp.symbols("c0:%d" % len(floats))
        sk = expr.xreplace(dict(zip(floats, cs)))
    return sk, cs, np.array([float(f) for f in floats])


def refit(sk, cs, p0, syms, cols, F):
    """Per-node constants of skeleton sk on the learned fields F (n, P); returns (fitted fields, constants) or None."""
    fn = sp.lambdify(tuple(syms) + tuple(cs), sk, "numpy")
    ev = lambda c: np.broadcast_to(np.asarray(fn(*cols, *c), float), (F.shape[1],))
    out, P = [], []
    for k in range(len(F)):
        if len(cs):
            sol = least_squares(lambda c: ev(c) - F[k], p0, max_nfev=2000); c = sol.x
        else:
            c = np.zeros(0)
        y = ev(c)
        if not np.all(np.isfinite(y)):
            return None
        out.append(y); P.append(c)
    return np.array(out), np.array(P)


def knee(items):
    """items: (complexity, loss, payload); PySR's model_selection='best' on their Pareto front -> ((score, -index), front)."""
    fr, best = [], np.inf
    for c, l, p in sorted(items, key=lambda t: (t[0], t[1])):
        if l < best and (not fr or fr[-1][0] != c):
            fr.append((c, l, p)); best = l
    sc = [0.0] + [(-np.log(fr[k][1] / fr[k - 1][1]) / (fr[k][0] - fr[k - 1][0])) if fr[k][1] > 0 else np.inf for k in range(1, len(fr))]
    thr = 1.5 * min(l for _, l, _ in fr)
    return max((s, -k) for k, ((c, l, p), s) in enumerate(zip(fr, sc)) if l <= thr), fr


def _pysr_export_guard():
    """After a search PySR converts every front equation to sympy (its sympy_format / lambda_format columns, not used
    here), and sympy's automatic evaluation can stall on deeply nested expressions. A 30 s guard returns a placeholder
    for that unused conversion only; equation strings, complexities and losses come from the search and are unaffected."""
    import pysr.export as ex
    if getattr(ex, "_guarded", False):
        return
    orig = ex.pysr2sympy

    def guarded(*a, **k):
        def h(*_):
            raise TimeoutError
        old = signal.signal(signal.SIGALRM, h); signal.alarm(30)
        try:
            return orig(*a, **k)
        except TimeoutError:
            return sp.Symbol("unconverted")
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old)
    ex.pysr2sympy = guarded; ex._guarded = True


def front(X, Y, names):
    """Every equation on PySR's Pareto front per output coordinate: [complexity, loss, equation]."""
    from pysr import PySRRegressor
    _pysr_export_guard()
    m = PySRRegressor(niterations=400, binary_operators=["+", "-", "*"], unary_operators=list(UNARY), maxsize=25,
                      populations=15, population_size=40, procs=0, parallelism="serial", deterministic=True,
                      progress=False, verbosity=0, random_state=0, temp_equation_file=True, model_selection="best")
    m.fit(X, Y, variable_names=names)
    eqs = m.equations_ if isinstance(m.equations_, list) else [m.equations_]
    return [[[int(c), float(l), str(e)] for c, l, e in zip(t["complexity"], t["loss"], t["equation"])] for t in eqs]


def fit_run(dd):
    info = json.load(open(dd + "info.json")); z = np.load(dd + "data.npz"); sysn = info["system"]; names = info["names"]
    grid, Fn, Z, lab, to_true, T = z["grid"], z["F_node"], z["Z"], z["labels"], z["to_true"], z["T_node"]
    syms = [sp.Symbol(n) for n in names]; cols = [grid[:, k] for k in range(grid.shape[1])]
    fpath = dd + "fronts.json"
    saved = json.load(open(fpath)) if os.path.exists(fpath) else {}
    res = dict(system=sysn, run=info["run"], classes=[]); t0 = time.time()
    for k in range(lab.max() + 1):
        mem = np.where(lab == k)[0]; order = mem[np.argsort(Z[mem, 0])]
        picks = sorted({int(order[int(q * (len(order) - 1))]) for q in (0, .25, .5, .75, 1)})
        if str(k) not in saved:
            saved[str(k)] = dict(picks=picks, fronts=[front(grid, Fn[i], names) for i in picks])
            json.dump(saved, open(fpath, "w"))
        fronts = saved[str(k)]["fronts"]
        for j in range(Fn.shape[2]):
            cands = {}
            for fr in fronts:
                for cx, _, eq in fr[j]:
                    try:
                        sk, cs, p0 = skeleton(eq, syms)
                    except Exception:
                        continue
                    key = sp.srepr(sk)
                    if key in cands:
                        cands[key][4] = min(cands[key][4], cx)
                    else:
                        cands[key] = [sk, cs, p0, eq, cx]
            F = Fn[mem, :, j]; Tt = T[mem, :, j]; allr = []; done = None
            for sk, cs, p0, eq, cx in sorted(cands.values(), key=lambda c: (int(sp.count_ops(c[0])), len(c[1]))):
                if done is not None and int(sp.count_ops(sk)) > done:
                    break
                try:
                    r = refit(sk, cs, p0, syms, cols, F)
                except Exception:
                    r = None
                if r is None:
                    continue
                fitF, P = r
                allr.append(dict(ops=int(sp.count_ops(sk)), cx=cx, mse=float(((F - fitF) ** 2).mean()), rl=r2(F.ravel(), fitF.ravel()),
                                 sk=sk, cs=cs, P=P, fitF=fitF))
                if allr[-1]["rl"] >= TOL and done is None:
                    done = allr[-1]["ops"]
            passing = [it for it in allr if it["rl"] >= TOL]
            if passing:
                ch = min(passing, key=lambda it: (it["ops"], -it["rl"]))
            else:
                (_, kk), fr = knee([(it["cx"], it["mse"], it) for it in allr]); ch = fr[-kk][2]
            truth = TRUE[sysn](int(to_true[k]), j)
            with sp.evaluate(False):
                med = ch["sk"].xreplace({c: sp.Float(v) for c, v in zip(ch["cs"], np.median(ch["P"], 0))})
                one = ch["sk"].xreplace({c: sp.Integer(1) for c in ch["cs"]})
            ex = term_types(str(med), names, grid=grid) == term_types(truth, names)
            st = term_types(str(one), names) == term_types(truth, names)
            c = dict(cluster=k, true_class=int(to_true[k]), coord=j, n=len(mem), picks=picks, n_candidates=len(cands), n_refitted=len(allr),
                     reached=bool(passing), skeleton=str(ch["sk"]), ops=ch["ops"], complexity=ch["cx"],
                     median_equation=str(med), median_constants=[float(v) for v in np.median(ch["P"], 0)] if len(ch["cs"]) else [],
                     constants=ch["P"].tolist(), r2_learned=ch["rl"], r2_true_nodes=r2(Tt.ravel(), ch["fitF"].ravel()),
                     exact=bool(ex), strict=bool(st), truth=truth)
            res["classes"].append(c)
            print("%-60s %-9s class %d->%d coord %d n %3d | %-5s | %-44s | R2 learned %.4f true(nodes) %7.4f | exact %-5s strict %s" % (
                os.path.basename(dd.rstrip("/"))[:60], sysn, k, c["true_class"], j, len(mem), "0.999" if passing else "pysr",
                c["skeleton"][:44], c["r2_learned"], c["r2_true_nodes"], c["exact"], c["strict"]), flush=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(dd + "equations.json", "w"), indent=1)


def summary():
    SET = (("a1v2_N30", "MixRelax"), ("a2_N30", "MixOsc N=30"), ("a2_N60", "MixOsc N=60"), ("grid118", "Grid-118"),
           ("kura_ER100", "Kuramoto-± p⁻ 0.5"), ("glv_N30", "GLV-±"))
    tot = np.zeros(4, int)
    print("\n%-18s %-10s %-10s %-10s %s" % ("setting", "exact", "strict", "PySR rule", "R2 vs true per-node (min - max)"))
    for key, name in SET:
        cl = [c for p in sorted(glob.glob("experiments/distill_pernode/*%s*/equations.json" % key)) for c in json.load(open(p))["classes"]]
        if not cl:
            continue
        q = np.array([sum(c["exact"] for c in cl), sum(c["strict"] for c in cl), sum(not c["reached"] for c in cl), len(cl)]); tot += q
        rt = [c["r2_true_nodes"] for c in cl]
        print("%-18s %2d / %-5d %2d / %-5d %2d / %-5d %.3f - %.3f" % (name, q[0], q[3], q[1], q[3], q[2], q[3], min(rt), max(rt)))
    print("%-18s %2d / %-5d %2d / %-5d %2d / %d" % ("total", tot[0], tot[3], tot[1], tot[3], tot[2], tot[3]))


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary()
    else:
        for dd in sorted(glob.glob("experiments/distill_pernode/%s/" % sys.argv[1])):
            fit_run(dd)
