"""New default procedure: per cell, the lower-validation-MSE run of {non-negative add_z, plain signed
add_z_signed without warm-up}; grid: the lowest validation MSE over both variants x 3 restarts (grid_select.py rule —
blow-ups, i.e. last-epoch val MSE > 1, are counted, not excluded). Runs per cell come from
experiments/runs/sel/manifest.json (experiments/launch/sel_default.py). Cells with a missing or failed member are
listed as pending and left out of the means.
Writes experiments/runs/sel/selected.json {group: {"run", "variant", ...}} for the read-out scripts and prints per
benchmark: n, mean AUROC / AUPRC, edge-sign accuracy where the truth has negative edges, picks (n = non-negative,
s = signed), plus Netsim (27 sims x 3 training seeds, per-sim seed means) vs NOGI with a Wilcoxon test.
Run from the project root with PYTHONPATH=."""
import collections, json, os, re, numpy as np
from scipy.stats import wilcoxon
from hetndl.snapshot import metrics_of
R = "experiments/runs"; MAN = os.path.join(R, "sel", "manifest.json")
ROBUST = ("_n3_", "_n5_", "_n10_", "_n20_", "noise", "snr")


def load(run):
    p = os.path.join(run, "metrics.json")
    if not os.path.exists(p):
        return None, "missing"
    m = json.load(open(p))
    return (None, "failed") if m.get("failed") else (m, "ok")


def bench(group):
    ds = group.split("/")[-2] if group.startswith("ba/") else group.split("/")[0]
    name = group.split("/")[-1]
    b = ("BA " if group.startswith("ba/") else "") + re.sub(r"_seed\d+$|_g\d+$", "", ds)
    b = re.sub(r"netsim_sim\d+", "Netsim", b)
    rob = [t.strip("_") for t in ROBUST if t in name]
    return b + (" [%s]" % re.search(r"(n\d+|sg51_noise[0-9.]+|snr\d+)", name).group(1) if rob else "")


man = json.load(open(MAN))
groups = collections.defaultdict(list)                      # grid restarts share a group
for cell, e in man.items():
    groups[re.sub(r"_r\d+$", "", cell)].append(e)
sel, pending = {}, []
for g, entries in sorted(groups.items()):
    cands, bad = [], []
    for e in entries:
        for v in ("nonneg", "nowu"):
            m, st = load(e[v])
            if m is None:
                bad.append("%s %s (%s)" % (v, e[v], st)); continue
            cands.append((m["val_mse_best"], v, e[v], m))
    if bad:
        pending.append((g, bad)); continue
    val, v, run, m = min(cands, key=lambda c: c[0]); b = metrics_of(m)
    sel[g] = dict(run=run, variant=v, val=val, auroc=b["auroc"], auprc=b["auprc"],
                  sign=b.get("sign_acc") if b.get("has_neg_edges") else None, rho=None,   # node constants via omega_readout.py
                  rollout=b.get("rollout_nrmse"), blowups=sum(c[3]["last"]["val_mse"] > 1 for c in cands), n_cand=len(cands))
json.dump(sel, open(os.path.join(R, "sel", "selected.json"), "w"), indent=1, default=float)

rows = collections.defaultdict(list)
for g, s in sel.items():
    rows[bench(g)].append(s)
print("%-42s %4s %15s %8s  picks" % ("benchmark", "n", "AUROC / AUPRC", "sign"))
for b in sorted(rows):
    xs = rows[b]; sg = [x["sign"] for x in xs if x["sign"] is not None]
    print("%-42s %4d   %.3f / %.3f %8s  %s" % (b, len(xs), np.mean([x["auroc"] for x in xs]), np.mean([x["auprc"] for x in xs]),
                                             "%.3f" % np.mean(sg) if sg else "—", "".join("n" if x["variant"] == "nonneg" else "s" for x in xs)))

# Netsim vs NOGI (per-sim means over the 3 training seeds)
ours, nogi = collections.defaultdict(list), collections.defaultdict(list)
for g, s in sel.items():
    mm = re.match(r"netsim_sim(\d+)/.*_seed(\d+)$", g)
    if mm:
        k, sd = int(mm.group(1)), int(mm.group(2)); ours[k].append((s["auroc"], s["auprc"]))
        m, st = load(os.path.join(R, "netsim_rk4sub", "netsim_sim%d" % k, "orig_seed%d" % sd))
        if m is not None:
            nogi[k].append((metrics_of(m)["auroc"], metrics_of(m)["auprc"]))
ks = [k for k in sorted(ours) if len(ours[k]) == 3 and len(nogi[k]) == 3]
if ks:
    a = np.array([np.mean(ours[k], 0) for k in ks]); b = np.array([np.mean(nogi[k], 0) for k in ks])
    print("\nNetsim (%d sims x 3 seeds): ours %.3f / %.3f vs NOGI %.3f / %.3f, wins %d/%d/%d, Wilcoxon p %.3f / %.3f" % (
        len(ks), *a.mean(0), *b.mean(0), (a[:, 0] > b[:, 0] + 1e-9).sum(), (abs(a[:, 0] - b[:, 0]) <= 1e-9).sum(),
        (a[:, 0] < b[:, 0] - 1e-9).sum(), wilcoxon(a[:, 0], b[:, 0]).pvalue, wilcoxon(a[:, 1], b[:, 1]).pvalue))
print("\n%d groups selected, %d pending" % (len(sel), len(pending)))
for g, bad in pending:
    print("  PENDING %-55s %s" % (g, "; ".join(x.split(" ")[0] + " " + x.split("(")[-1].rstrip(")") for x in bad)))
