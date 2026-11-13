"""The scale study (Kuramoto-± p- 0.5, N = 200 and 400, data seeds 1-3) retrained with the l1 penalty normalised by the
number of ordered node pairs, lambda_1 = 1e-8 * 9900 / (N (N - 1)), i.e. 2.4874e-09 at N = 200 and 6.2030e-10 at N = 400
(9900 = pairs at N = 100, where 1e-8 is the default). Runs scale_l1n/... of experiments/reproduce/jobs.json: both
candidates of the six scale cells, all other arguments as in runs/scale. Written before any AUROC of these runs was read.

Selection and scores as the scale table (experiments/analysis/scale_table.py): per data seed the member (add_z /
add_z_signed) with the lower val_mse_best, AUROC / AUPRC of its reported snapshot (metrics_of), sign accuracy on the true
edges. Compared with the unnormalised runs (runs/scale, lambda_1 = 1e-8; "published" below) of the same data seeds.
Reading (fixed before any run): if the mean AUROC at N = 400 returns to 0.98 or more, the loss of accuracy came from a
penalty not adjusted to N and the normalisation rule is given with the method; otherwise the penalty strength does not
explain it. "0.98 or more" is read on the mean over the three data seeds, printed to three decimals; a pending seed makes
the reading incomplete.
A failed or missing run is listed and its seed left pending (never rerun). Ground truth only in the scores.
Run from the project root with PYTHONPATH=. (-> results/supplement/scale_penalty.txt; experiments/figures/fig_robust.py
reads that file for the dashed line of its panel (b))."""
import glob, json, os, numpy as np
from hetndl.snapshot import metrics_of
R = "experiments/runs"; THRESH = 0.98


def ds(N, s):
    return "kura_ER%d_K2_s0.3_n0.5%s" % (N, "" if s == 1 else "_g%d" % s)


def pick(root, N, s, suffix):
    runs = [os.path.join(R, root, ds(N, s), "%s_diff_lz0_seed%d%s" % (mo, s, suffix)) for mo in ("add_z", "add_z_signed")]
    c, bad = [], []
    for r in runs:
        p = os.path.join(r, "metrics.json")
        if not os.path.exists(p):
            bad.append("%s missing" % os.path.basename(r)); continue
        m = json.load(open(p))
        if m.get("failed"):
            bad.append("%s failed (%s)" % (os.path.basename(r), m.get("reason", "")[:40])); continue
        c.append((m["val_mse_best"], r, m))
    if bad:
        return None, bad
    val, r, m = min(c, key=lambda x: x[0]); b = metrics_of(m)
    return dict(member="n" if m["args"]["model"] == "add_z" else "s", auroc=b["auroc"], auprc=b["auprc"], sign=b.get("sign_acc"),
                lam=m["args"]["lambda_l1"], epochs=m["epochs_run"]), None


def main():
    print("Penalty normalised by N (N - 1) vs the published scale runs (lambda_1 = 1e-8); Kuramoto-± p- 0.5, ER, "
          "mean degree as N = 100\n")
    reading = None
    for N in (200, 400):
        print("== N = %d" % N); cols = {"published": [], "scaled": []}
        for s in (1, 2, 3):
            a, abad = pick("scale", N, s, ""); b, bbad = pick("scale_l1n", N, s, "_l1n")
            fa = "%s AUROC %.3f AUPRC %.3f sign %.3f" % (a["member"], a["auroc"], a["auprc"], a["sign"]) if a else "PENDING " + "; ".join(abad)
            fb = "%s AUROC %.3f AUPRC %.3f sign %.3f (lambda_1 %.4e)" % (b["member"], b["auroc"], b["auprc"], b["sign"], b["lam"]) if b else "PENDING " + "; ".join(bbad)
            print("   seed %d  published %s | scaled %s" % (s, fa, fb))
            if a:
                cols["published"].append(a)
            if b:
                cols["scaled"].append(b)
        for k, v in cols.items():
            if v:
                print("   mean %-9s (n %d)  AUROC %.3f  AUPRC %.3f  sign %.3f" % (k, len(v), np.mean([x["auroc"] for x in v]),
                                                                       np.mean([x["auprc"] for x in v]), np.mean([x["sign"] for x in v])))
        if N == 400:
            v = cols["scaled"]
            if len(v) == 3:
                mu = round(float(np.mean([x["auroc"] for x in v])), 3)
                reading = ("(a) recovered: mean AUROC %.3f >= %.2f -> Sect. 5.4: the loss came from a penalty not adjusted to N; "
                           "Sect. 4.4 gives the normalisation rule" % (mu, THRESH)) if mu >= THRESH else \
                          ("(b) not recovered: mean AUROC %.3f < %.2f -> the paper says the penalty strength does not explain the loss" % (mu, THRESH))
            else:
                reading = "INCOMPLETE (%d of 3 seeds at N = 400)" % len(v)
    print("\nREADING: %s" % reading)


if __name__ == "__main__":
    main()
