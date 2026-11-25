"""Controlled runtime measurement : the training logs come from several machines under shared load and
NRI / AIDD / ARNI / Granger record no time, so a runtime table needs its own measurement. Every method runs ALONE, one
after the other, on one otherwise idle GPU / its CPU, on two benchmarks (MixOsc N30 seed 1, N = 30;
Kuramoto-± p- 0.5 seed 1, N = 100), with the arguments of its main-table run except the epoch count:
  ours (both sign members) and NOGI: 20 epochs -> median seconds per epoch (log.csv, epochs >= 2) x the protocol epochs
      of the main-table run; ours = sum of the two members (the default procedure trains both);
  NRI, AIDD: 3 and 6 epochs -> per-epoch = (t6 - t3) / 3, overhead = t3 - 3 per-epoch; total = overhead + 600 per-epoch;
  ARNI (full; greedy on Kuramoto), Granger: complete runs, timed (CPU).
COSINE needs the local LLM server and is reported from its own logs (wall_sec, local A100 under shared load).
Output runs/timing/<ds>/<method>/ and runs/timing/timing.json. Usage (project root, hetndl env):
python experiments/launch/timing_bench.py <device, e.g. cuda:0>"""
import glob, json, os, subprocess, sys, time
import numpy as np
R = "experiments/runs"; O = os.path.join(R, "timing"); PY = sys.executable
man = json.load(open(os.path.join(R, "sel", "manifest.json")))
BENCH = {"a2_N30_seed1": dict(cell="a2_N30_seed1/add_z_signed_diff_lz0_seed1", nogi="a2/a2_N30_seed1/orig_seed1",
                              nri="nri/a2_N30_seed1/seed1", aidd="aidd/a2_N30_seed1/seed1", arni=["full"]),
         "kura_ER100_K2_s0.3_n0.5": dict(cell="kura_ER100_K2_s0.3_n0.5/add_z_signed_diff_lz0_seed1", nogi="motiv_trig/kura_ER100_K2_s0.3_n0.5/orig_seed1",
                                         nri="nri/kura_ER100_K2_s0.3_n0.5/seed1", aidd="aidd/kura_ER100_K2_s0.3_n0.5/seed1", arni=["full", "greedy"])}
DROP = ("out", "device", "release_reset", "release_lr_scale", "release_stable", "release_jaccard", "init_from", "w_support", "log_w_stats")


def args_of(run):
    p = run if run.startswith(R) else os.path.join(R, run)
    a = dict(json.load(open(os.path.join(p, "metrics.json")))["args"])
    a["data"] = os.path.join("experiments", "data", os.path.basename(a["data"]))   # some early runs stored absolute local paths
    return a


def cli(module, a, out, dev=None):
    cmd = [PY, "-m", module]
    for k, v in a.items():
        if k in DROP or v is None:
            continue
        cmd += ["--" + k, str(v)]
    cmd += ["--out", out] + (["--device", dev] if dev else [])
    return cmd


def timed(cmd, out):
    os.makedirs(out, exist_ok=True); t = time.time()
    with open(os.path.join(out, "stdout.log"), "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="8")).returncode
    return time.time() - t, rc


def sec_per_epoch(out):
    rows = [l.split(",") for l in open(os.path.join(out, "log.csv")).read().split("\n")[1:] if l]
    return float(np.median([float(r[-1]) for r in rows if int(r[0]) >= 2]))


def main(dev):
    res = json.load(open(os.path.join(O, "timing.json"))) if os.path.exists(os.path.join(O, "timing.json")) else {}
    for ds, b in BENCH.items():
        r = res.setdefault(ds, {})
        for who, run in (("ours_nonneg", man[b["cell"]]["nonneg"]), ("ours_signed", man[b["cell"]]["nowu"]), ("nogi", b["nogi"])):
            if who in r and r[who].get("rc") == 0:
                continue
            a = args_of(run); ep = int(a["epochs"]); a["epochs"] = 20; out = os.path.join(O, ds, who)
            wall, rc = timed(cli("hetndl.train", a, out, dev), out)
            r[who] = dict(rc=rc, wall_20=wall, sec_per_epoch=sec_per_epoch(out) if rc == 0 else None, protocol_epochs=ep)
            r[who]["est_total_sec"] = r[who]["sec_per_epoch"] * ep if rc == 0 else None
            json.dump(res, open(os.path.join(O, "timing.json"), "w"), indent=1); print(ds, who, r[who], flush=True)
        for who in ("nri", "aidd"):
            if who in r:
                continue
            a = args_of(b[who]); ep = int(a["epochs"]); t = {}
            for e in (3, 6):
                a["epochs"] = e; out = os.path.join(O, ds, "%s_e%d" % (who, e)); t[e], rc = timed(cli("hetndl.baselines." + who, a, out, dev), out)
            pe = (t[6] - t[3]) / 3; r[who] = dict(t3=t[3], t6=t[6], sec_per_epoch=pe, overhead=t[3] - 3 * pe, protocol_epochs=ep,
                                                  est_total_sec=t[3] - 3 * pe + ep * pe)
            json.dump(res, open(os.path.join(O, "timing.json"), "w"), indent=1); print(ds, who, r[who], flush=True)
        for mode in b["arni"]:
            who = "arni_" + mode
            if who in r:
                continue
            a = args_of(glob.glob(os.path.join(R, "arni", ds, "%s_seed1" % mode))[0]); out = os.path.join(O, ds, who)
            wall, rc = timed(cli("hetndl.baselines.arni", a, out), out); r[who] = dict(rc=rc, est_total_sec=wall)
            json.dump(res, open(os.path.join(O, "timing.json"), "w"), indent=1); print(ds, who, r[who], flush=True)
        if "granger" not in r:
            a = args_of(os.path.join("granger", ds, "seed1")); out = os.path.join(O, ds, "granger")
            wall, rc = timed(cli("hetndl.baselines.granger", a, out), out); r["granger"] = dict(rc=rc, est_total_sec=wall)
            json.dump(res, open(os.path.join(O, "timing.json"), "w"), indent=1); print(ds, "granger", r["granger"], flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
