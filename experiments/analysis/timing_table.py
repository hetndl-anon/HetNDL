"""Runtime table from experiments/launch/timing_bench.py (runs/timing/timing.json): every method alone on one idle
RTX 3090 (ARNI / Granger on its CPU), estimated full protocol time; ours = both sign members (the default
procedure trains both; they are independent and can run in parallel). COSINE cannot run there (local LLM server): its
wall time is taken from its own logs (runs/cosine/<ds>/seed1, local A100 shared with other jobs) and marked as such.
Run from the project root."""
import json, os
R = "experiments/runs"; t = json.load(open(os.path.join(R, "timing", "timing.json")))
NAMES = {"a2_N30_seed1": "MixOsc N30 (N 30, 1000 ep)", "kura_ER100_K2_s0.3_n0.5": "Kuramoto-± (N 100, 600 ep)"}
COS = {"a2_N30_seed1": "a2_N30_seed1/seed1", "kura_ER100_K2_s0.3_n0.5": "kura_ER100_K2_s0.3_n0.5/seed1"}
fmt = lambda s: "%.0f s" % s if s < 120 else ("%.0f min" % (s / 60) if s < 7200 else "%.1f h" % (s / 3600))
print("%-28s %-22s %-10s %-10s %-10s %-10s %-14s %-10s %s" % ("benchmark", "ours (2 members / 1)", "NOGI", "NRI", "AIDD", "ARNI", "", "Granger", "COSINE (logs, shared A100)"))
for ds, r in t.items():
    o = r["ours_nonneg"]["est_total_sec"] + r["ours_signed"]["est_total_sec"]
    arni = " / ".join(fmt(r[k]["est_total_sec"]) for k in ("arni_full", "arni_greedy") if k in r)
    c = os.path.join(R, "cosine", COS[ds], "metrics.json"); cw = json.load(open(c))["wall_sec"] if os.path.exists(c) else None
    print("%-28s %-22s %-10s %-10s %-10s %-25s %-10s %s" % (NAMES[ds], "%s / %s" % (fmt(o), fmt(o / 2)), fmt(r["nogi"]["est_total_sec"]), fmt(r["nri"]["est_total_sec"]),
                                                       fmt(r["aidd"]["est_total_sec"]), arni, fmt(r["granger"]["est_total_sec"]), fmt(cw) if cw else "—"))
print("per epoch (s): " + "; ".join("%s ours %.2f / %.2f, NOGI %.2f, NRI %.2f, AIDD %.2f" % (ds, r["ours_nonneg"]["sec_per_epoch"], r["ours_signed"]["sec_per_epoch"],
                                                                                               r["nogi"]["sec_per_epoch"], r["nri"]["sec_per_epoch"], r["aidd"]["sec_per_epoch"]) for ds, r in t.items()))
