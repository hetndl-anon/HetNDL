"""Combined heterogeneity: node classes together with inhibitory edges (data sets a2t / a2ts), and node classes,
source-dependent coupling and inhibitory edges together (data sets a3u / a3s). Written after these runs had finished and
before any of their numbers was read; the stall rule below was fixed at the same time.

Benchmarks (experiments/reproduce/datasets.json; N = 30, data seeds 1-3; graph, node classes and sampling of MixOsc). With
the linear coupling of MixOsc an inhibitory edge makes the network unstable (the simulated trajectories diverge for 20-50 %
inhibitory edges), so these data sets use bounded coupling laws (hetndl.gen.a3_mixed --laws):
  a2t_N30_seed{k}        kappa tanh(2 (u_j - u_i)) on every edge, all edges excitatory (the control);
  a2ts_N30_n0.5_seed{k}  the same law, half of the edges inhibitory (--neg_frac 0.5);
  a3u_N30_seed{k}        kappa sin(u_j - u_i) from FitzHugh-Nagumo sources and kappa tanh(2 (u_j - u_i)) from Van der Pol
                         sources, all edges excitatory (the control);
  a3s_N30_n0.5_seed{k}   the same two laws, half of the edges inhibitory, drawn independently of the node classes.
Runs (experiments/reproduce/jobs.json, a2t/ a2ts/ a3u/ a3s/): HetNDL with the arguments of its MixOsc runs (a2t, a2ts:
both candidates) and of its MixOsc-E runs (a3u, a3s: the four candidates, shared and edge-embedding, non-negative and
signed); NOGI, signed NOGI, ARNI (full) and Granger with the arguments of their MixOsc / MixOsc-E runs.

Read-out code paths are those of Sect. 5.6 (experiments/analysis/edge_b_compare.py):
  * hetndl.snapshot.metrics_of: AUROC / AUPRC by |W| of the reported snapshot, roll-out NRMSE;
  * hetndl.edge_readout.oriented_sign_accuracy: sign on the true edges after the model's own gauge;
  * hetndl.edge_readout.edge_type_readout: learned edges by 2-means, truth-free K per edge and per source, ARI;
  * experiments/analysis/a2_fields.py: self-field classes, truth-free K, ARI (called as edge_b_compare.fields does).
Baselines: NOGI (orig) and signed NOGI (orig_signed) by metrics_of as in Table 2; sign of signed NOGI under the better
global sign (as snogi_table.py); ARNI (full) and Granger from their metrics.json, ARNI sign = sign_acc_jac (as
baseline_table.arni_sign).
Selection (truth-free, by val_mse_best): on a2t / a2ts between C+ (add_z) and C± (add_z_signed); on a3u / a3s the minimum
over the four candidates (shared / edge-embedding x non-negative / signed), as in Sect. 5.6.

Stall rule (fixed before reading). The supplement describes a stalled C± on all-excitatory MixOsc as a validation error
80-263 times that of C+, while non-stalled pairs differ by less than a factor of 7. On a2ts C+ cannot represent
inhibition, so the reference is the matched control a2t of the same data seed (generator arguments identical except the
edge signs and --x_max 20): C± stalls on a2ts seed k if
    val_mse(C±, a2ts_k) > 10 x min(val_mse(C+, a2t_k), val_mse(C±, a2t_k)).
Also printed: the same ratio for C+ on a2ts, the C±/C+ ratio on a2t (the known all-excitatory stall) and the training
restarts of C± on a2ts (runs/restart/nowu/a2ts_*/..._i101, _i102) when they exist.
Reading: a2ts - if C± stalls on a seed, both candidates fail there (C+ cannot express the inhibitory edges), a real
limitation of the method; otherwise C± fits where C+ cannot. a3s - does the four-way selection choose the signed
edge-embedding candidate (on a3u, the all-excitatory control, an edge-embedding candidate is expected); graph AUROC, sign
accuracy, node-class ARI and edge-type ARI against the baselines.
A failed or missing run is listed and left out (never rerun, never replaced). Ground truth only in the scores.
Run from the project root with PYTHONPATH=. (-> results/supplement/combined.txt)."""
import json, os, re, subprocess, sys, numpy as np
from hetndl.edge_readout import edge_type_readout, oriented_sign_accuracy
from hetndl.gen.kuramoto import load_dataset
from hetndl.snapshot import array_path, metrics_of
R = "experiments/runs"; PY = sys.executable; SEEDS = (1, 2, 3); STALL = 10.0
BASES = (("NOGI", "orig_seed%d"), ("sNOGI", "orig_signed_seed%d"), ("ARNI", "arni_full_seed%d"), ("Granger", "granger_seed%d"))


def met(run):
    p = os.path.join(run, "metrics.json")
    if not os.path.exists(p):
        return None, "missing"
    m = json.load(open(p))
    return (None, "failed: " + m.get("reason", "")[:60]) if m.get("failed") else (m, "")


def has_neg(ds):
    return bool((load_dataset("experiments/data/%s.npz" % ds)["S"] < 0).any())


def fields(run):
    out = subprocess.run([PY, "experiments/analysis/a2_fields.py", run], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=".")).stdout
    g = re.search(r"K_sel (\d+) ARI\(K_sel\) (-?[0-9.]+)", out)
    return (int(g.group(1)), float(g.group(2))) if g else (None, float("nan"))


def cand(run, ds, edge_types):
    m, note = met(run)
    if m is None:
        return dict(note=note)
    b = metrics_of(m); r = dict(run=run, val=m["val_mse_best"], auroc=b["auroc"], auprc=b["auprc"], roll=b["rollout_nrmse"])
    r["sign"] = oriented_sign_accuracy(run) if has_neg(ds) else None
    r["K"], r["ari"] = fields(run)
    if edge_types:
        e = edge_type_readout(run)
        r.update(ecorr="%d/%d" % (e["n_correct"], e["n_learned"]), Ke=e.get("K_edge"), ari_e=e.get("ari_edge"), Ks=e.get("K_source"),
                 ari_s=e.get("ari_source_edges"))
    return r


def base(dsdir, ds, s):
    out = {}
    for lab, sub in BASES:
        run = os.path.join(dsdir, sub % s); m, note = met(run)
        if m is None:
            out[lab] = dict(note=note); continue
        if "val_mse_best" in m:
            b = metrics_of(m); r = dict(auroc=b["auroc"], auprc=b["auprc"])
        else:
            r = dict(auroc=m["auroc"], auprc=m["auprc"])
        r["sign"] = None
        if has_neg(ds) and lab == "sNOGI":
            S = load_dataset("experiments/data/%s.npz" % ds)["S"]; E = S != 0
            W = np.load(array_path(run, "W", m)); a = float((np.sign(W[E]) == np.sign(S[E])).mean()); r["sign"] = max(a, 1.0 - a)
        elif has_neg(ds) and lab == "ARNI":
            j = m.get("sign_acc_jac"); r["sign"] = j if j is not None and np.isfinite(j) else m.get("sign_acc")
        out[lab] = r
    return out


def f3(x):
    return "  -  " if x is None or (isinstance(x, float) and not np.isfinite(x)) else "%.3f" % x


def line(name, r):
    if "val" not in r:
        return "    %-12s %s" % (name, r["note"])
    s = "    %-12s val %.2e  AUROC %s AUPRC %s  sign %s  classes K %s ARI %s  roll-out %s" % (
        name, r["val"], f3(r["auroc"]), f3(r["auprc"]), f3(r["sign"]), r["K"], f3(r["ari"]), f3(r["roll"]))
    if "ecorr" in r:
        s += "  | edges %s correct, edge types per-edge K %s ARI %s, per-source K %s ARI %s" % (r["ecorr"], r["Ke"], f3(r["ari_e"]), r["Ks"], f3(r["ari_s"]))
    return s


def bline(b):
    return "    baselines    " + "  ".join("%s %s" % (k, "%s/%s%s" % (f3(v["auroc"]), f3(v["auprc"]), (" sign " + f3(v["sign"])) if v["sign"] is not None else "")
                                           if "auroc" in v else v["note"]) for k, v in b.items())


def summary(title, sel, bases):
    print("  %s, mean over seeds (n): selected %s (%d)" % (title, "AUROC %s AUPRC %s sign %s class ARI %s" % tuple(
        f3(np.mean([r[k] for r in sel if r.get(k) is not None])) if any(r.get(k) is not None for r in sel) else "  -  "
        for k in ("auroc", "auprc", "sign", "ari")), len(sel)))
    for lab, _ in BASES:
        v = [b[lab] for b in bases if "auroc" in b[lab]]
        if v:
            sg = [x["sign"] for x in v if x["sign"] is not None]
            print("    %-8s AUROC %s AUPRC %s%s (%d)" % (lab, f3(np.mean([x["auroc"] for x in v])), f3(np.mean([x["auprc"] for x in v])),
                                                    (" sign " + f3(np.mean(sg))) if sg else "", len(v)))


def signed_edges():
    print("== MixOsc nodes, saturating coupling kappa tanh(2 (u_j - u_i)); a2t all excitatory (control), "
          "a2ts half of the edges inhibitory; N = 30")
    stalls, judged, sel_t, sel_s, b_t, b_s = 0, 0, [], [], [], []
    for s in SEEDS:
        ct, ts = "a2t_N30_seed%d" % s, "a2ts_N30_n0.5_seed%d" % s
        c = {}
        for ds, top in ((ct, "a2t"), (ts, "a2ts")):
            print("  seed %d  %s" % (s, ds))
            for mem, mo in (("C+", "add_z"), ("C±", "add_z_signed")):
                c[(ds, mem)] = cand(os.path.join(R, top, ds, "%s_diff_lz0_seed%d" % (mo, s)), ds, False); print(line(mem, c[(ds, mem)]))
            ok = [mem for mem in ("C+", "C±") if "val" in c[(ds, mem)]]
            if len(ok) == 2:
                w = min(ok, key=lambda k: c[(ds, k)]["val"])
                print("    selected %s;  val(C±) / val(C+) = %.2f" % (w, c[(ds, "C±")]["val"] / c[(ds, "C+")]["val"]))
                (sel_t if ds == ct else sel_s).append(c[(ds, w)])
            b = base(os.path.join(R, top, ds), ds, s); print(bline(b)); (b_t if ds == ct else b_s).append(b)
        okc = [c[(ct, k)]["val"] for k in ("C+", "C±") if "val" in c[(ct, k)]]
        if len(okc) == 2 and "val" in c[(ts, "C±")]:
            ref = min(okc); r_pm = c[(ts, "C±")]["val"] / ref; judged += 1; stalls += r_pm > STALL
            r_p = c[(ts, "C+")]["val"] / ref if "val" in c[(ts, "C+")] else float("nan")
            print("  seed %d stall test: val(C±, a2ts) / best val(a2t) = %.2f -> %s;  val(C+, a2ts) / best val(a2t) = %.2f" % (
                s, r_pm, "STALL" if r_pm > STALL else "no stall", r_p))
        rs = []
        for i in (101, 102):
            run = os.path.join(R, "restart", "nowu", ts, "add_z_signed_diff_lz0_seed%d_i%d" % (s, i)); m, note = met(run)
            if m is None:
                rs.append("i%d %s" % (i, note)); continue
            b = metrics_of(m); r = m["val_mse_best"] / min(okc) if len(okc) == 2 else float("nan")
            rs.append("i%d val %.2e (ratio %.2f, %s) AUROC %.3f" % (i, m["val_mse_best"], r, "STALL" if r > STALL else "no stall", b["auroc"]))
        print("  seed %d restarts of C± on a2ts: %s" % (s, "; ".join(rs)))
    print("\n  summary, a2t / a2ts")
    summary("a2t (control)", sel_t, b_t); summary("a2ts (half inhibitory)", sel_s, b_s)
    print("  READING a2t / a2ts: C± stalls on %d of %d a2ts seeds (rule: ratio > %g)%s" % (stalls, judged, STALL, "" if judged == 3 else " INCOMPLETE"))
    if judged == 3:
        print("    -> %s" % ("both candidates fail on the stalled seeds: a real limitation of the method" if stalls
                             else "C± fits with inhibitory edges where C+ cannot (training restarts: lines above)"))


def signed_edges_and_types():
    print("\n== node classes (FitzHugh-Nagumo / Van der Pol), source-dependent bounded coupling (sin from FHN sources, "
          "tanh from VdP sources), a3u all excitatory (control), a3s half of the edges inhibitory; N = 30")
    K4 = (("shared C+", "add_z_diff_lz0_seed%d"), ("shared C±", "add_z_signed_diff_lz0_seed%d"),
          ("edge C+", "add_z_diff_lz0_ee2_seed%d"), ("edge C±", "add_z_signed_diff_lz0_ee2_seed%d"))
    pick = {"a3u": [], "a3s": []}; sel = {"a3u": [], "a3s": []}; bs = {"a3u": [], "a3s": []}
    for s in SEEDS:
        for key, ds in (("a3u", "a3u_N30_seed%d" % s), ("a3s", "a3s_N30_n0.5_seed%d" % s)):
            print("  seed %d  %s" % (s, ds)); c = {}
            for name, tpl in K4:
                c[name] = cand(os.path.join(R, key, ds, tpl % s), ds, True); print(line(name, c[name]))
            ok = [k for k in c if "val" in c[k]]
            if len(ok) == 4:
                w = min(ok, key=lambda k: c[k]["val"]); pick[key].append(w); sel[key].append(c[w])
                print("    four-way selection: %s  (val %s)" % (w, ", ".join("%s %.2e" % (k, c[k]["val"]) for k in ok)))
            else:
                print("    four-way selection: incomplete")
            b = base(os.path.join(R, key, ds), ds, s); print(bline(b)); bs[key].append(b)
    print("\n  summary, a3u / a3s")
    for key, title in (("a3u", "a3u (control)"), ("a3s", "a3s (half inhibitory)")):
        print("  %s: selected %s" % (title, ", ".join(pick[key])))
        summary(title, sel[key], bs[key])
        e = [r for r in sel[key] if r.get("ari_s") is not None]
        if e:
            print("    selected: edge-type ARI per edge %s, per source %s; per-source K %s" % (
                f3(np.mean([r["ari_e"] for r in e])), f3(np.mean([r["ari_s"] for r in e])), [r["Ks"] for r in e]))
    n_ok = sum(p == "edge C±" for p in pick["a3s"])
    print("  READING a3u / a3s: the four-way selection chooses the signed edge-embedding candidate on %d of %d a3s seeds; "
          "an edge-embedding candidate on %d of %d a3u seeds" % (n_ok, len(pick["a3s"]), sum(p.startswith("edge") for p in pick["a3u"]), len(pick["a3u"])))


if __name__ == "__main__":
    signed_edges(); signed_edges_and_types()
