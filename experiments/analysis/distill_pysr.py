"""Stage 2 of symbolic distillation (env `sr`, PySR): fit closed-form equations to the exported grids of
experiments/distill/<tag>/grids.npz and score them against the TRUE functions.

Per run: one PySR fit per learned class mean F (all output dims at once) and one for the coupling c(x_i, x_j)
(first output dim; the others are reported as max |c|). The same operator library for every system:
+ - * with square, cube, tanh, sin. Scores: R^2 of the equation against the learned grid (how faithful the
distillation is) and against the true function (how right it is); for the coupling the true function is known
only up to the (W, G) scale gauge, so it is scored after a least-squares scale.
The paper reports the coupling laws from here. Its self-field equations come from the per-node distillation
(pernode_export.py -> pernode_distill.py: one form per read-out class, constants per node); the class-mean self-field
equations written here are not reported.
Usage: python distill_pysr.py <distill dir> [...] [--niterations 400] [--procs 8]"""
import argparse, json, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")


def r2(y, yhat):
    y, yhat = np.asarray(y, float).ravel(), np.asarray(yhat, float).ravel()
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


def r2_scaled(y, yhat):
    """R^2 after the best scalar alpha: y ~ alpha * yhat (the W-G scale gauge)."""
    y, yhat = np.asarray(y, float).ravel(), np.asarray(yhat, float).ravel()
    alpha = float(yhat @ y / max(yhat @ yhat, 1e-12))
    return r2(y, alpha * yhat), alpha


UNARY = ("square", "cube", "tanh", "sin")


def fit(X, y, names, niterations, procs, seed, maxsize=25, tol=0.999, unary_operators=UNARY):
    """unary_operators: the library of the paper by default; distill_operators.py passes an extended one."""
    from pysr import PySRRegressor
    m = PySRRegressor(niterations=niterations, binary_operators=["+", "-", "*"],
                      unary_operators=list(unary_operators), maxsize=maxsize, populations=15,
                      population_size=40, procs=procs, parallelism="multithreading" if procs > 1 else "serial",
                      progress=False, verbosity=0, random_state=seed, temp_equation_file=True,
                      model_selection="best")
    m.fit(X, y, variable_names=names)
    # Selection: the SIMPLEST Pareto-front equation whose R^2 against the learned grid is >= tol (distillation at a
    # tolerance). PySR's own "best" chases the MLP's tiny wiggles and returns needlessly long expressions.
    eqs = m.equations_ if isinstance(m.equations_, list) else [m.equations_]
    Y = y.reshape(len(X), -1)
    out_eq, out_pred = [], []
    for j, tab in enumerate(eqs):
        tab = tab.sort_values("complexity")
        pick = None
        for _, row in tab.iterrows():
            yhat = np.asarray(row["lambda_format"](X), float).reshape(-1)
            if yhat.shape[0] == 1:
                yhat = np.full(len(X), yhat[0])
            if r2(Y[:, j], yhat) >= tol:
                pick, pred = row, yhat
                break
        if pick is None:  # nothing reaches tol: fall back to PySR's best
            pick = m.get_best()[j] if len(eqs) > 1 else m.get_best()
            pred = np.asarray(pick["lambda_format"](X), float).reshape(-1)
        out_eq.append(pick["equation"]); out_pred.append(pred)
    return out_eq, np.stack(out_pred, 1)


def coupling_output_dim(c):
    """Output component the learned coupling acts on (largest variance over the grid; truth-free). 0 for every
    one-dimensional system and for A2 / A3 (coupling in u); 1 for the swing model (coupling in d omega / dt)."""
    c = np.asarray(c, float).reshape(len(c), -1)
    return int(np.argmax(c.var(0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--niterations", type=int, default=400)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for dd in a.dirs:
        z = np.load(os.path.join(dd, "grids.npz")); info = json.load(open(os.path.join(dd, "info.json")))
        sysname = info["system"]; res = dict(system=sysname, run=info["run"], coupling=info["coupling"], self_terms=[], t0=time.time())
        names_x = ["theta"] if sysname == "kuramoto" else (["x"] if sysname in ("a1", "glv") else
                                                             (["omega"] if sysname == "powergrid" else ["u", "v"]))   # grid: F sampled on observed states, input omega
        print("== %s (%s, %s)" % (dd, sysname, info["coupling"]), flush=True)
        # ---- self terms
        if "F_cls" in z.files:
            grid, Fc, T, match = z["grid"], z["F_cls"], z["F_true"], z["cluster_to_true"]
            for k in range(len(Fc)):
                eqs, pred = fit(grid, Fc[k], names_x, a.niterations, a.procs, a.seed)
                tk = T[match[k]]
                r = dict(cluster=int(k), true_class=int(match[k]), equations=eqs,
                         r2_vs_learned=[r2(Fc[k][:, j], pred[:, j]) for j in range(pred.shape[1])],
                         r2_vs_true=[r2(tk[:, j], pred[:, j]) for j in range(pred.shape[1])],
                         learned_vs_true_r2=[r2(tk[:, j], Fc[k][:, j]) for j in range(pred.shape[1])])
                res["self_terms"].append(r)
                print("  F cluster %d (true class %d): %s | R2 vs learned %s | R2 vs TRUE %s" % (
                    k, match[k], eqs, np.round(r["r2_vs_learned"], 3), np.round(r["r2_vs_true"], 3)), flush=True)
        elif sysname == "glv":  # pooled F(x, z) over all nodes (parameter heterogeneity)
            grid, Fn, Ft, Zn = z["grid"], z["F_node"][:, :, 0], z["F_true_node"][:, :, 0], z["Z"]
            N_, P_ = Fn.shape
            Xp = np.concatenate([np.repeat(grid, N_, 0).reshape(P_, N_, 1).transpose(1, 0, 2).reshape(-1, 1),
                                 np.repeat(Zn, P_, 0)], 1)                                            # (N*P, 1 + d_z)
            eqs, pred = fit(Xp, Fn.reshape(-1), ["x"] + ["z%d" % k for k in range(Zn.shape[1])], a.niterations, a.procs, a.seed)
            res["self_terms"] = dict(kind="pooled F(x, z)", equation=eqs[0], r2_vs_learned=r2(Fn.reshape(-1), pred[:, 0]),
                                     r2_vs_true=r2(Ft.reshape(-1), pred[:, 0]), learned_vs_true_r2=r2(Ft.reshape(-1), Fn.reshape(-1)),
                                     r_from_linear_coef_corr=float(np.corrcoef(z["quad_coef"][:, 1], z["omega"])[0, 1]))
            print("  F(x, z) pooled: %s | R2 vs learned %.3f | R2 vs TRUE %.3f | learned vs true %.3f | corr(linear coef, r_i) %.3f" % (
                eqs[0], res["self_terms"]["r2_vs_learned"], res["self_terms"]["r2_vs_true"], res["self_terms"]["learned_vs_true_r2"],
                res["self_terms"]["r_from_linear_coef_corr"]), flush=True)
            # Class-SR style: discover candidate skeletons on five nodes (z quantiles), turn every numeric
            # constant of each candidate into a free per-node parameter, refit the constants on EVERY node by least
            # squares, and keep the simplest candidate whose pooled R^2 against the learned curves is >= 0.99 (else the
            # best). Truth-free selection; the truth is used only for scoring. If the chosen skeleton expands to a
            # polynomial in x, its per-node x^1 coefficient is the growth-rate read-out.
            import sympy as sp
            from scipy.optimize import least_squares
            xs = sp.Symbol("x"); order = np.argsort(Zn[:, 0])
            picks = sorted({int(order[int(q * (N_ - 1))]) for q in (0.0, 0.25, 0.5, 0.75, 1.0)})
            cands = []
            for i in picks:
                e_i, _ = fit(grid, Fn[i], ["x"], a.niterations, a.procs, a.seed, maxsize=15)
                cands.append((i, e_i[0]))
            res["self_terms"]["node_equations"] = {str(i): e for i, e in cands}
            print("  per-node skeletons (nodes %s): %s" % (picks, [e for _, e in cands]), flush=True)
            xg = grid[:, 0]; best = None
            for i, e in cands:
                try:
                    expr = sp.sympify(e, locals={"square": lambda u: u ** 2, "cube": lambda u: u ** 3})
                except Exception:
                    continue
                floats = sorted(expr.atoms(sp.Float), key=lambda f: float(f))
                cs = sp.symbols("c0:%d" % len(floats)); skel = expr.xreplace(dict(zip(floats, cs)))
                fn = sp.lambdify((xs,) + tuple(cs), skel, "numpy"); p0 = np.array([float(f) for f in floats])
                try:
                    P = []
                    for k in range(N_):
                        sol = least_squares(lambda c: np.asarray(fn(xg, *c), float) * np.ones_like(xg) - Fn[k], p0, max_nfev=2000)
                        P.append(sol.x)
                    P = np.array(P); fitF = np.stack([np.asarray(fn(xg, *P[k]), float) * np.ones_like(xg) for k in range(N_)])
                except Exception:
                    continue
                rl, rt = r2(Fn.reshape(-1), fitF.reshape(-1)), r2(Ft.reshape(-1), fitF.reshape(-1))
                cx = sp.count_ops(skel)
                print("    candidate %s (from node %d, %d ops): pooled R2 vs learned %.3f, vs TRUE %.3f" % (skel, i, cx, rl, rt), flush=True)
                key = (0 if rl >= 0.99 else 1, cx if rl >= 0.99 else -rl)
                if best is None or key < best[0]:
                    best = (key, skel, P, rl, rt)
            if best is not None:
                _, skel, P, rl, rt = best
                out_st = dict(shared_skeleton=str(skel), shared_r2_vs_learned=rl, shared_r2_vs_true=rt)
                try:
                    poly = sp.Poly(sp.expand(skel), xs)
                    c1 = poly.coeff_monomial(xs)
                    f1 = sp.lambdify(sp.symbols("c0:%d" % P.shape[1]), c1, "numpy")
                    rhat = np.array([float(f1(*P[k])) for k in range(N_)])
                    out_st["x1_coef_corr_r"] = float(np.corrcoef(rhat, z["omega"])[0, 1])
                    out_st["x1_coef_vs_r_max_abs_err"] = float(np.abs(rhat - z["omega"]).max())
                except Exception:
                    out_st["x1_coef_corr_r"] = float("nan")
                res["self_terms"].update(out_st)
                print("  SHARED skeleton %s | pooled R2 vs learned %.3f vs TRUE %.3f | corr(x^1 coef, r_i) %.3f" % (
                    skel, rl, rt, out_st["x1_coef_corr_r"]), flush=True)
        else:  # kuramoto: F should be constant = omega_i
            Fm, om = z["F_mean"], z["omega"]
            res["self_terms"] = dict(kind="per-node constant", rel_std=float(np.mean(z["F_std"] / (np.abs(Fm) + 1e-9))),
                                     rho_F_mean_omega=float(np.corrcoef(Fm, om)[0, 1]),
                                     slope=float(np.polyfit(om, Fm, 1)[0]), max_abs_err=float(np.abs(Fm - om).max()))
            print("  F: per-node constants, rel std %.3f, corr with omega %.4f, max |F - omega| %.3f" % (
                res["self_terms"]["rel_std"], res["self_terms"]["rho_F_mean_omega"], res["self_terms"]["max_abs_err"]), flush=True)
        # ---- coupling, on the output component it acts on (coupling_output_dim; dim 0 except the swing model)
        Xc = np.stack([z["c_grid_xi"][:, 0], z["c_grid_xj"][:, 0]], -1)
        c1 = ["theta"] if sysname == "powergrid" else names_x[:1]
        cn = [n + "_i" for n in c1] + [n + "_j" for n in c1]
        cd = coupling_output_dim(z["c_learned"]); y = z["c_learned"][:, cd]
        eqs, pred = fit(Xc, y, cn, a.niterations, a.procs, a.seed, maxsize=20)
        rt, alpha = r2_scaled(z["c_true"][:, cd], pred[:, 0])
        rl_true, alpha_l = r2_scaled(z["c_true"][:, cd], y)
        res["coupling_eq"] = dict(equation=eqs[0], r2_vs_learned=r2(y, pred[:, 0]), r2_vs_true_scaled=rt, scale_alpha=alpha,
                                  learned_vs_true_r2_scaled=rl_true, learned_scale=alpha_l,
                                  output_dim=cd, other_dims_max_abs=[float(np.abs(z["c_learned"][:, j]).max()) for j in range(z["c_learned"].shape[1]) if j != cd])
        print("  coupling: %s | R2 vs learned %.3f | R2 vs TRUE (scaled) %.3f, alpha %.3f | other dims max|c| %s" % (
            eqs[0], res["coupling_eq"]["r2_vs_learned"], rt, alpha, np.round(res["coupling_eq"]["other_dims_max_abs"], 3)), flush=True)
        # ---- edge weights in the true units: W_learned * c_learned ~ W_true * c_true and c_true ~ alpha * c_learned,
        # so W_eff = W_learned / alpha should equal W_true on the true edges
        A, W, Wt = z["A"], z["W"] * info["gauge_sign"], z["W_true"]
        e = A > 0
        W_eff = W / alpha if abs(alpha) > 1e-9 else W
        rel = float(np.abs(W_eff[e] - Wt[e]).mean() / np.abs(Wt[e]).mean())
        res["weights"] = dict(corr_on_edges=float(np.corrcoef(W_eff[e], Wt[e])[0, 1]), rel_abs_err_on_edges=rel,
                              sign_acc=float((np.sign(W_eff[e]) == np.sign(Wt[e])).mean()),
                              max_abs_offedge=float(np.abs(W_eff[~e & ~np.eye(len(A), dtype=bool)]).max()),
                              min_abs_onedge=float(np.abs(Wt[e]).min()))
        print("  weights W/alpha vs true on edges: corr %.4f, rel |err| %.3f, sign acc %.3f | max |W| off-edge %.3f vs min true %.3f" % (
            res["weights"]["corr_on_edges"], rel, res["weights"]["sign_acc"], res["weights"]["max_abs_offedge"], res["weights"]["min_abs_onedge"]), flush=True)
        res["elapsed_s"] = time.time() - res.pop("t0")
        json.dump(res, open(os.path.join(dd, "pysr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
