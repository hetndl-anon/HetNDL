"""Settings / leakage audit: no training setting may differ between benchmarks without a rule over observable
properties, and no ground truth may enter any choice. Run before reporting any number. Exit code 1 if a check fails.

Check 1 — uniform settings. For the runs that are REPORTED (experiments/runs/sel/selected.json from sel_tables.py, i.e.
the default procedure), every training argument that differs between benchmarks must be listed in ALLOWED with a
written rule over OBSERVABLE properties fixed before the results (state type, trajectory length, compute budget), the
same for every method. Anything derived from the generating process (number / kind of node classes, edge signs, edge
density, which parameter is heterogeneous) may not set a model or training choice — that is how d_z = 1 / 2 slipped in.

Check 2 — truth in the read-out code. Lines that fix a number of clusters or read true labels in the analysis and
read-out code must be on the allowlist below (scoring-only uses, each with the reason).
Run from the project root with PYTHONPATH=."""
import collections, glob, json, os, re, sys
R = "experiments/runs"
CELL_KEYS = {"out", "device", "data", "seed", "noise_std", "noise_snr", "n_traj", "smooth_window", "num_workers",
             "log_w_stats", "eval_every", "model", "sign_warmup"}        # cell identity / logging / the two selection members
ALLOWED = {
    "epochs": "fixed per-benchmark compute budget, identical for every method (Kuramoto 600 as in the NOGI paper, Netsim 200 "
              "because of cost, otherwise 1000); the validation-MSE snapshot makes longer budgets harmless",
    "feat": "periodic observables (phases) are fed as (cos, sin) — a property of the observed variable known to the user, "
            "applied to every method",
    "method": "solver rule: dopri5 unless it is impractically slow on the data (stiff or noisy: grid, Netsim, Kuramoto, GLV), "
              "then fixed-step rk4; identical for every method",
    "step_size": "rk4 sub-steps for Netsim (4 per TR), identical for every method",
    "cluster_k": "scoring only: the z-cluster ARI reference metric with the true K (reported as a known-K reference, never "
                 "used for selection or read-out)",
}
CODE_ALLOW = {  # file: reason for every fixed-K / truth-label line that may stay
    "experiments/analysis/a1_fcurves.py": "true curves and per-class errors for scoring; the reported K is select_k",
    "experiments/analysis/a2_fields.py": "true fields and per-class errors for scoring; the reported K is select_k",
    "experiments/analysis/free_coupling_split.py": "true curves / fields and per-class errors for scoring (as a1_fcurves / a2_fields); the reported K is select_k",
    "experiments/analysis/a3_edges.py": "KMeans(2) printed as the known-K reference next to select_k",
    "hetndl/edge_readout.py": "KMeans(2) inside the scoring block (true edge types present) for the known-K reference "
                              "columns ref_knownK_edge / ref_knownK_source; the reported K is select_k",
    "experiments/analysis/granger_mechanism.py": "KMeans(true K) printed as the known-K reference next to select_k",
    "experiments/analysis/grid_readout.py": "KMeans(2) on the damping slope printed as the known-K reference next to select_k",
    "experiments/analysis/distill_export.py": "true curves for scoring only; classes from select_k",
    "hetndl/train.py": "cluster_k for the reference z-ARI metric only",
    "hetndl/metrics.py": "GMM with the given K for the reference z-ARI metric only",
    "hetndl/readout.py": "select_k itself and the 2-means edge binarisation (2 = edge / non-edge, not a class count)",
}
fail = False
sel = json.load(open(os.path.join(R, "sel", "selected.json")))
vals = collections.defaultdict(lambda: collections.defaultdict(set))
for g, s in sel.items():
    m = json.load(open(os.path.join(s["run"], "metrics.json")))
    b = re.sub(r"_seed\d+$|_g\d+$", "", g.split("/")[-2] if g.startswith("ba/") else g.split("/")[0])
    b = re.sub(r"netsim_sim\d+", "netsim", b)
    for k, v in m["args"].items():
        if k not in CELL_KEYS:
            vals[k][b].add(str(v))
print("== Check 1: settings that differ between benchmarks (reported runs, %d groups)" % len(sel))
for k in sorted(vals):
    distinct = {v for b in vals[k] for v in vals[k][b]}
    if len(distinct) > 1:
        ok = k in ALLOWED
        fail |= not ok
        print("  %s %-10s %s" % ("ok  " if ok else "FAIL", k, ALLOWED.get(k, "NOT ALLOWED — unify it or write a rule")))
        print("       " + " | ".join("%s=%s" % (b, "/".join(sorted(v))) for b, v in sorted(vals[k].items())))
print("== Check 2: fixed cluster counts / true labels in read-out code")
pat = re.compile(r"KMeans\(\s*\d|KMeans\(\s*K_true|n_cls\s*=\s*\d|range\(\s*[23]\s*\)|cluster_k")
for f in sorted(glob.glob("experiments/analysis/*.py") + glob.glob("hetndl/*.py")):
    if f.endswith("audit_settings.py") or f.endswith("sweep.py"):
        continue
    hits = [i + 1 for i, line in enumerate(open(f)) if pat.search(line)]
    if hits:
        ok = f in CODE_ALLOW
        fail |= not ok
        print("  %s %s lines %s — %s" % ("ok  " if ok else "FAIL", f, hits[:8], CODE_ALLOW.get(f, "NOT ALLOWED — use hetndl.readout.select_k")))
print("AUDIT", "FAILED" if fail else "PASSED")
sys.exit(1 if fail else 0)
