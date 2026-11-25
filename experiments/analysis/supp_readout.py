"""Read-out of three additional experiments of the supplement (x1 = runs/a1s, x2 = runs/l1zero, x5 = runs/l1sweep). It only
chains the existing read-out code paths (no new metric): hetndl.snapshot.metrics_of (validation-MSE snapshot),
hetndl.edge_readout.oriented_sign_accuracy, hetndl.readout.binarize_2means / select_k, and the scripts
experiments/analysis/{a1_fcurves, a2_fields, granger_mechanism}.py and omega_readout.score. Selection = the candidate
with the lower validation MSE (rule of sel_tables.py / ablation_table.py: both candidates must exist and not have
failed). Ground truth enters only the scores.
  x1  MixRelax-+/-: per data seed the pick, AUROC / AUPRC, edge-sign accuracy on true edges, K and ARI of the self-field
      read-out, precision / recall / F1 of the 2-means graph; NOGI / ARNI / Granger AUROC, AUPRC and two-stage ARI.
  x2  lambda_1 = 0 next to the default (main-table pick of runs/sel/selected.json): AUROC / AUPRC / node read-out
      (Kuramoto |rho|, otherwise ARI) per setting, and the differences of the three means.
  x5  lambda_1 in {0, 1e-9, 1e-8, 1e-7, 1e-6} on Kuramoto-+/- p- 0.5 (runs/l1zero, runs/l1sweep, the main-table cell of
      runs/sel/manifest.json, runs/ablation/sens/l1e6): the rows of x2 for the selected candidate, plus the AUROC and
      validation MSE of each candidate. The five values are all that were run.
Optional second argument: another root with the same layout (used to test this script on finished runs).
Run from the project root: PYTHONPATH=. python experiments/analysis/supp_readout.py x1|x2|x5 [root]"""
import glob, json, os, re, subprocess, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from hetndl.edge_readout import oriented_sign_accuracy
from hetndl.gen.kuramoto import load_dataset
from hetndl.readout import binarize_2means, select_k
from hetndl.snapshot import array_path, metrics_of
R = "experiments/runs"; PY = sys.executable; AN = "experiments/analysis"


def met(run):
    p = os.path.join(run, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return None if m.get("failed") else m


def script_line(script, run):
    return subprocess.run([PY, os.path.join(AN, script), run], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=".")).stdout.strip()


def k_ari(script, run):
    k = re.search(r"K_sel (\d+) ARI\(K_sel\) ([0-9.\-]+)", script_line(script, run))
    return (int(k.group(1)), float(k.group(2))) if k else (None, None)


def prf(Bn, A):                                                        # as experiments/analysis/binary_graph_table.py
    off = ~np.eye(len(A), dtype=bool); tp = int((Bn & A)[off].sum()); n_pred, n_true = int(Bn[off].sum()), int(A[off].sum())
    P, Rc = tp / max(n_pred, 1), tp / max(n_true, 1)
    return P, Rc, 2 * P * Rc / max(P + Rc, 1e-12), n_pred, n_true


def pick(cands):
    """cands: {variant: run dir}. (variant, run, metrics) of the lower validation MSE, or None if one is missing / failed."""
    ms = {v: met(r) for v, r in cands.items()}
    if any(m is None for m in ms.values()):
        return None
    v = min(ms, key=lambda v: ms[v]["val_mse_best"])
    return v, cands[v], ms[v], {u: ms[u]["val_mse_best"] for u in ms}


def x1(root):
    print("MixRelax-+/- (a1v2s_N30_n0.5, data seeds 1-3); ours = lower validation MSE of {W >= 0, W real}")
    rows = {k: [] for k in ("ours", "NOGI", "ARNI", "Granger")}
    for s in (1, 2, 3):
        ds = "a1v2s_N30_n0.5_seed%d" % s; b = os.path.join(root, ds); d = load_dataset("experiments/data/%s.npz" % ds)
        A = d["A"] > 0; c = d["classes"]
        print("== %s: %d edges, %d inhibitory" % (ds, int(A.sum()), int((d["S"] < 0).sum())))
        p = pick({"nonneg": os.path.join(b, "add_z_diff_lz0_seed%d" % s), "signed": os.path.join(b, "add_z_signed_diff_lz0_seed%d" % s)})
        if p is None:
            print("  ours     pending (a candidate is missing or failed)")
        else:
            v, run, m, vals = p; bm = metrics_of(m); K, ari = k_ari("a1_fcurves.py", run)
            P, Rc, F1, n_pred, n_true = prf(binarize_2means(np.load(array_path(run, "W", m))), A)
            sg = oriented_sign_accuracy(run)
            print("  ours     pick %s (val MSE nonneg %.3e, signed %.3e; best epoch %d) | AUROC %.3f AUPRC %.3f | sign acc %.3f "
                  "(metrics.json %.3f) | read-out K %s ARI %s | 2-means P %.3f R %.3f F1 %.3f (%d predicted, %d true)" % (
                      v, vals["nonneg"], vals["signed"], m["best_epoch"], bm["auroc"], bm["auprc"], sg, bm["sign_acc"], K,
                      "%.3f" % ari if ari is not None else "-", P, Rc, F1, n_pred, n_true))
            rows["ours"].append((bm["auroc"], bm["auprc"], sg, ari, P, Rc, F1))
        m = met(os.path.join(b, "orig_seed%d" % s))
        if m is None:
            print("  NOGI     pending")
        else:
            bm = metrics_of(m); print("  NOGI     AUROC %.3f AUPRC %.3f (no node read-out: one shared function)" % (bm["auroc"], bm["auprc"]))
            rows["NOGI"].append((bm["auroc"], bm["auprc"]))
        for run in sorted(glob.glob(os.path.join(b, "arni_*"))):
            m = met(run)
            if m is None:
                print("  ARNI     pending"); continue
            C = np.load(os.path.join(run, "self_coef.npy")); K, l, _ = select_k(C, k_max=5)      # as arni_mechanism.py
            Cs = (C - C.mean(0)) / (C.std(0) + 1e-12)
            ref = adjusted_rand_score(c, KMeans(len(np.unique(c)), n_init=20, random_state=0).fit_predict(Cs))
            print("  ARNI %-4s AUROC %.3f AUPRC %.3f | sign acc %.3f (coefficient) %.3f (Jacobian) | two-stage K %d ARI %.3f (known-K %.3f)" % (
                m["mode"], m["auroc"], m["auprc"], m.get("sign_acc", float("nan")), m.get("sign_acc_jac", float("nan")), K, adjusted_rand_score(c, l), ref))
            rows["ARNI"].append((m["auroc"], m["auprc"], adjusted_rand_score(c, l)))
        run = os.path.join(b, "granger_seed%d" % s); m = met(run)
        if m is None:
            print("  Granger  pending")
        else:
            line = script_line("granger_mechanism.py", run); k = re.search(r"K_sel (\d+) ARI ([0-9.\-]+)", line)
            t = re.search(r"true K ([0-9.\-]+)", line)
            print("  Granger  AUROC %.3f AUPRC %.3f | two-stage K %s ARI %s (known-K %s)" % (m["auroc"], m["auprc"], k.group(1) if k else "-",
                                                                                    k.group(2) if k else "-", t.group(1) if t else "-"))
            rows["Granger"].append((m["auroc"], m["auprc"], float(k.group(2)) if k else float("nan")))
    print("== means over the finished data seeds")
    names = {"ours": ("AUROC", "AUPRC", "sign acc", "ARI", "P", "R", "F1"), "NOGI": ("AUROC", "AUPRC"),
             "ARNI": ("AUROC", "AUPRC", "ARI"), "Granger": ("AUROC", "AUPRC", "ARI")}
    for k, r in rows.items():
        if r:
            mu = np.array([[np.nan if x is None else x for x in row] for row in r], float).mean(0)
            print("  %-8s n %d  %s" % (k, len(r), "  ".join("%s %.3f" % (n, x) for n, x in zip(names[k], mu))))


def node(fam, run):
    """(node read-out, number of classes K or None)"""
    if fam == "kura":
        from experiments.analysis.omega_readout import score as omega_score
        return omega_score(run)[0], None
    K, ari = k_ari("a1_fcurves.py" if fam == "a1" else "a2_fields.py", run)
    return ari, K


def weights(run, m, ds):
    """Share of the summed |W| outside the true edges, precision / recall of the 2-means cut, last training loss."""
    import csv
    W = np.abs(np.load(array_path(run, "W", m))); A = load_dataset("experiments/data/%s.npz" % ds)["A"] > 0
    off = ~np.eye(len(A), dtype=bool); P, Rc = prf(binarize_2means(W), A)[:2]
    last = list(csv.DictReader(open(os.path.join(run, "log.csv"))))[-1]
    return float(W[off & ~A].sum() / W[off].sum()), P, Rc, float(last["train_loss"]), float(last["val_mse"])


ROW = ("  %-20s seed %d %-8s AUROC %.4f AUPRC %.4f node %.4f K %s | val MSE %.3e (best epoch %d) | non-edge share of |W| %.3f, 2-means P %.3f R %.3f | "
       "last epoch: training loss %.2e, val MSE %.2e | pick %s | %s")


def row(lab, fam, ds, s, tag, run, m):
    """Print one read-out row; return (AUROC, AUPRC, node read-out)."""
    bm = metrics_of(m); nr, K = node(fam, run); w = weights(run, m, ds)
    print(ROW % (lab, s, tag, bm["auroc"], bm["auprc"], nr, "-" if K is None else K, m["val_mse_best"], m["best_epoch"], w[0], w[1], w[2], w[3], w[4],
                 "signed" if "signed" in run else "nonneg", run.split("runs/")[1]))
    return bm["auroc"], bm["auprc"], nr


def x2(root):
    sel = json.load(open(os.path.join(R, "sel", "selected.json")))
    FAM = [("Kuramoto-+/- p- 0.5", "kura", ["kura_ER100_K2_s0.3_n0.5" + ("" if s == 1 else "_g%d" % s) for s in (1, 2, 3)]),
           ("MixRelax", "a1", ["a1v2_N30_seed%d" % s for s in (1, 2, 3)]), ("MixOsc N30", "a2", ["a2_N30_seed%d" % s for s in (1, 2, 3)])]

    print("lambda_1 = 0 against the default (lambda_1 = 1e-8), both by the lower validation MSE of the two candidates; node read-out = "
          "|rho| (Kuramoto) or ARI of the truth-free read-out")
    worst = 0.0
    for lab, fam, dss in FAM:
        res = {"default": [], "l1zero": []}
        for i, ds in enumerate(dss):
            s = i + 1; drun = sel["%s/add_z_signed_diff_lz0_seed%d" % (ds, s)]["run"]; dm = met(drun)
            c = {}
            for mem in ("nonneg", "nowu"):
                g = sorted(glob.glob(os.path.join(root, mem, ds, "*")))
                c[mem] = g[0] if g else os.path.join(root, mem, ds, "missing")
            p = pick(c)
            for tag, run, m in (("default", drun, dm), ("l1zero", p[1] if p else None, p[2] if p else None)):
                if m is None:
                    print("  %-20s seed %d %-8s pending" % (lab, s, tag)); continue
                res[tag].append(row(lab, fam, ds, s, tag, run, m))
        if len(res["default"]) == len(res["l1zero"]) == len(dss):
            a, b = np.array(res["default"], float).mean(0), np.array(res["l1zero"], float).mean(0); worst = max(worst, float(np.abs(a - b).max()))
            print("  %-20s MEAN default  AUROC %.4f AUPRC %.4f node %.4f" % (lab, *a))
            print("  %-20s MEAN l1zero   AUROC %.4f AUPRC %.4f node %.4f   | differences %+.4f %+.4f %+.4f" % (lab, *b, *(b - a)))
        else:
            worst = float("nan"); print("  %-20s incomplete" % lab)
    print("largest |difference of means| over the three settings and three metrics: %.4f (compared with 0.005)" % worst)
    print("size of the penalty at initialisation, lambda_1 * E||W||_1 with W ~ U(0, 1) and lambda_1 = 1e-8: N = 100: %.2e, N = 30: %.2e "
          "(the data term is a mean; compare with the validation MSE columns)" % (1e-8 * 100 * 99 / 2, 1e-8 * 30 * 29 / 2))


def x5(root):
    man = json.load(open(os.path.join(R, "sel", "manifest.json"))); lab = "Kuramoto-+/- p- 0.5"
    SRC = [("0", lambda mem, ds: os.path.join(R, "l1zero", mem, ds, "*")), ("1e-9", lambda mem, ds: os.path.join(root, "1e-9", mem, ds, "*")),
           ("1e-8", None), ("1e-7", lambda mem, ds: os.path.join(root, "1e-7", mem, ds, "*")),
           ("1e-6", lambda mem, ds: os.path.join(R, "ablation", "sens", "l1e6", mem, ds, "*"))]
    print("lambda_1 on Kuramoto-+/- p- 0.5 (N = 100), data seeds 1-3; selected = lower validation MSE of the two candidates; node = |rho|")
    for val, where in SRC:
        res = []
        for s in (1, 2, 3):
            ds = "kura_ER100_K2_s0.3_n0.5" + ("" if s == 1 else "_g%d" % s); c = {}
            for mem in ("nonneg", "nowu"):
                if where is None:
                    c[mem] = man["%s/add_z_signed_diff_lz0_seed%d" % (ds, s)][mem]
                else:
                    g = sorted(glob.glob(where(mem, ds))); assert len(g) <= 1, g
                    c[mem] = g[0] if g else os.path.join(root, val, mem, ds, "missing")
            p = pick(c)
            if p is None:
                print("  %-20s seed %d %-8s pending" % (lab, s, val)); continue
            a = json.load(open(os.path.join(p[1], "metrics.json")))["args"]; assert float(a["lambda_l1"]) == float(val) and a["seed"] == s, (p[1], a["lambda_l1"])
            res.append(row(lab, "kura", ds, s, val, p[1], p[2]))
            print("  %-20s seed %d %-8s candidates: %s" % (lab, s, val, "; ".join("%s AUROC %.4f val MSE %.3e" % (
                mem, metrics_of(met(c[mem]))["auroc"], met(c[mem])["val_mse_best"]) for mem in ("nonneg", "nowu"))))
        if len(res) == 3:
            print("  %-20s MEAN %-8s AUROC %.4f AUPRC %.4f node %.4f" % (lab, val, *np.array(res, float).mean(0)))


if __name__ == "__main__":
    if sys.argv[1] == "x1":
        x1(sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "a1s"))
    elif sys.argv[1] == "x5":
        x5(sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "l1sweep"))
    else:
        assert sys.argv[1] == "x2", sys.argv[1]
        x2(sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "l1zero"))
