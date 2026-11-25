"""The edge-embedding candidates of the six MixOsc-E networks (N = 30 and 60, data seeds 1-3) retrained with a normalisation
that is constant within an epoch (--edge_scale epoch: the per-edge RMS on 32 fixed training states, refreshed at the start
of every epoch and used in training and in evaluation). Runs edge_scale/... of experiments/reproduce/jobs.json, all other
arguments as the edge-embedding runs of runs/edge_b. Written before any AUROC, ARI, roll-out error or selection of these
runs was read; seen before writing: the run states and, while timing a slow run, the validation MSE in log.csv of two
non-negative runs (a3_N60, seeds 2 and 3).

The four conclusions of Sect. 5.6 are compared item by item (graph AUROC, edge-type ARI, node classes, roll-out error),
and the four-way selection. Each sentence of Sect. 5.6 is turned into a check (fixed before reading). Same code paths as
Sect. 5.6 (experiments/analysis/edge_b_compare.py):
"option B" = the edge-embedding member (non-negative / signed) with the lower val_mse_best, here the two `_es` runs;
"shared G" = the two members of the manifest cell (a3_N30) or of edge_b (a3_N60), unchanged.
  S1 graph      "With edge embeddings the graph stays at AUROC 1.000": option B's AUROC rounds to 1.000 on all six.
  S2 edge types "the edge types are recovered on all six networks (ARI 1.000)": per-source edge-type ARI
                (edge_type_readout, ari_source_edges) of option B rounds to 1.000 on all six.
  S3 classes    "the node classes on five": number of networks where option B's self-field ARI (a2_fields.py) rounds
                to 1.000; the sentence holds as written if it is 5 (printed either way).
  S4 roll-out   "the roll-out error falls from 0.083-0.104 to 0.029-0.046": the largest roll-out NRMSE of option B is
                below the smallest of shared G (the fall holds); the new range is printed.
  S5 selection  "the best shared-coupling candidate has 3.5-6.7 times the validation error of the best
                edge-conditioned one, which is chosen on all six networks": the minimum over the four candidates is an
                `_es` run on all six; the ratio range is printed.
Verdict: "unchanged" if S1, S2, S4, S5 hold and S3 counts 5 (Sect. 5.6 stands as written up to the printed ranges);
otherwise "changed" with the failing sentence(s).
The same quantities under the default --edge_scale batch (runs/edge_b ... _ee2_) are printed next to each line.
A failed or missing run is listed and its network left pending (never rerun). Ground truth only in the scores.
Run from the project root with PYTHONPATH=. (-> results/supplement/edge_scale.txt)."""
import json, os, re, subprocess, sys, numpy as np
from hetndl.edge_readout import edge_type_readout
from hetndl.snapshot import metrics_of
R = "experiments/runs"; PY = sys.executable
man = json.load(open(os.path.join(R, "sel", "manifest.json")))
VM = (("nonneg", "add_z"), ("nowu", "add_z_signed"))
NETS = ["a3_N%d_seed%d" % (n, s) for n in (30, 60) for s in (1, 2, 3)]


def met(run):
    p = os.path.join(run, "metrics.json")
    if not os.path.exists(p):
        return None, "missing"
    m = json.load(open(p))
    return (None, "failed") if m.get("failed") else (m, "ok")


def fields(run):
    out = subprocess.run([PY, "experiments/analysis/a2_fields.py", run], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=".")).stdout
    g = re.search(r"K_sel (\d+) ARI\(K_sel\) (-?[0-9.]+)", out)
    return (int(g.group(1)), float(g.group(2))) if g else (None, float("nan"))


def runs_of(ds):
    s = int(ds[-1]); cell = "%s/add_z_signed_diff_lz0_seed%d" % (ds, s)
    shared = [man[cell][v] for v, _ in VM] if cell in man else [os.path.join(R, "edge_b", v, ds, "%s_diff_lz0_seed%d" % (mo, s)) for v, mo in VM]
    batch = [os.path.join(R, "edge_b", v, ds, "%s_diff_lz0_ee2_seed%d" % (mo, s)) for v, mo in VM]
    epoch = [os.path.join(R, "edge_scale", v, ds, "%s_diff_lz0_ee2_seed%d_es" % (mo, s)) for v, mo in VM]
    return {"shared G": shared, "B batch": batch, "B epoch": epoch}


def pick(runs):
    c, bad = [], []
    for r in runs:
        m, st = met(r)
        if m is None:
            bad.append("%s %s" % (r, st))
        else:
            c.append((m["val_mse_best"], r, m))
    if bad:
        return None, bad
    val, r, m = min(c, key=lambda x: x[0]); return (r, m, val), None


def describe(r, m, edge):
    b = metrics_of(m); d = dict(run=r, val=m["val_mse_best"], auroc=b["auroc"], roll=b["rollout_nrmse"])
    d["K"], d["ari"] = fields(r)
    if edge:
        e = edge_type_readout(r); d.update(Ks=e.get("K_source"), ari_s=e.get("ari_source_edges"), Ke=e.get("K_edge"), ari_e=e.get("ari_edge"))
    return d


def r3(x):
    return x is not None and np.isfinite(x) and round(x, 3) == 1.0


def main():
    res = {}
    for ds in NETS:
        C = runs_of(ds); out = {}
        for kind, runs in C.items():
            p, bad = pick(runs)
            out[kind] = describe(p[0], p[1], kind != "shared G") if p else dict(pending=bad)
            if p:
                out[kind]["member"] = "nonneg" if p[1]["args"]["model"] == "add_z" else "signed"
        for kind in ("B batch", "B epoch"):                                 # four-way minimum (Sect. 5.6), per normalisation
            four = {}
            for k in ("shared G", kind):
                for r in C[k]:
                    m, st = met(r)
                    if m is not None:
                        four[r] = m["val_mse_best"]
            if len(four) == 4:
                w = min(four, key=four.get); sh = min(four[r] for r in C["shared G"]); eb = min(four[r] for r in C[kind])
                out[kind + " four"] = dict(choice=w, edge=w in C[kind], ratio=sh / eb)
        res[ds] = out
        print("== %s" % ds)
        for kind, d in out.items():
            if kind.endswith(" four"):
                print("   %-14s four-way choice %s (%s), val(best shared) / val(best edge) = %.2f" % (
                    kind, os.path.relpath(d["choice"], R), "edge" if d["edge"] else "shared", d["ratio"]))
            elif "pending" in d:
                print("   %-14s PENDING %s" % (kind, "; ".join(d["pending"])))
            else:
                print("   %-14s %-6s val %.2e AUROC %.3f roll-out %.3f fields K %s ARI %.3f%s" % (
                    kind, d["member"], d["val"], d["auroc"], d["roll"], d["K"], d["ari"],
                    "  edge types per-source K %s ARI %.3f, per-edge K %s ARI %.3f" % (d["Ks"], d["ari_s"], d["Ke"], d["ari_e"]) if "Ks" in d else ""))
    print()
    for kind in ("B batch", "B epoch"):
        ok = [ds for ds in NETS if "pending" not in res[ds][kind] and "pending" not in res[ds]["shared G"] and kind + " four" in res[ds]]
        print("== %s (%d of 6 networks complete)%s" % (kind, len(ok), "  [Sect. 5.6 as published]" if kind == "B batch" else "  [--edge_scale epoch]"))
        if not ok:
            continue
        B_ = [res[ds][kind] for ds in ok]; S_ = [res[ds]["shared G"] for ds in ok]; F_ = [res[ds][kind + " four"] for ds in ok]
        s1 = all(r3(d["auroc"]) for d in B_); s2 = all(r3(d["ari_s"]) for d in B_); s3 = sum(r3(d["ari"]) for d in B_)
        s4 = max(d["roll"] for d in B_) < min(d["roll"] for d in S_); s5 = all(f["edge"] for f in F_)
        print("  S1 graph AUROC rounds to 1.000 on all: %s (range %.3f-%.3f)" % (s1, min(d["auroc"] for d in B_), max(d["auroc"] for d in B_)))
        print("  S2 edge types per-source ARI rounds to 1.000 on all: %s (range %.3f-%.3f)" % (s2, min(d["ari_s"] for d in B_), max(d["ari_s"] for d in B_)))
        print("  S3 node classes ARI 1.000 on %d networks (Sect. 5.6: five)" % s3)
        print("  S4 roll-out falls: %s (shared G %.3f-%.3f, option B %.3f-%.3f)" % (
            s4, min(d["roll"] for d in S_), max(d["roll"] for d in S_), min(d["roll"] for d in B_), max(d["roll"] for d in B_)))
        print("  S5 edge-conditioned chosen on all: %s (%d of %d); val(best shared) / val(best edge) %.2f-%.2f" % (
            s5, sum(f["edge"] for f in F_), len(F_), min(f["ratio"] for f in F_), max(f["ratio"] for f in F_)))
        if kind == "B epoch":
            fails = [n for n, ok_ in (("S1", s1), ("S2", s2), ("S3", s3 == 5), ("S4", s4), ("S5", s5)) if not ok_]
            print("  READING: %s%s" % ("unchanged" if not fails else "changed (" + ", ".join(fails) + ")",
                                         "" if len(ok) == 6 else " - INCOMPLETE, %d of 6 networks" % len(ok)))


if __name__ == "__main__":
    main()
