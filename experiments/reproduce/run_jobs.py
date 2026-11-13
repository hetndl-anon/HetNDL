"""Run the training jobs behind the paper's numbers (experiments/reproduce/jobs.json).

jobs.json holds one entry per run directory that the analysis scripts of the paper and of its supplement read: 1902 runs
(HetNDL, its ablations and the NOGI-style shared-function model through hetndl.train; ARNI, Granger, NRI, AIDD, SGODE,
MTGNN, COSINE and the original NRI code through their runners). Each command was rebuilt from the arguments recorded by
the original run and checked by parsing it back with the entry point's own argument parser. "used_by" names the
analysis scripts that read the run, so a subset can be reproduced, e.g. only Table 2:  --used_by baseline_table.

A job is skipped when its out_dir already holds metrics.json (resume). GPU jobs are spread over --gpus with
--per_gpu jobs per device; CPU jobs (ARNI, Granger, and the runs we trained on a CPU: HetNDL and AIDD on the
five-oscillator Kuramoto data, NRI and AIDD on the spring network) run --cpu at a time. Each job writes
<out_dir>/job.log, and its wall time is appended to experiments/reproduce/run_jobs.log (the CPU times of the scale table
come from there). MTGNN, COSINE and original-NRI jobs need their external code (README.md); they are skipped with a
note when it is missing.

Usage (repository root):
  python experiments/reproduce/run_jobs.py --gpus 0,1 --per_gpu 3 --cpu 8 [--used_by baseline_table,scale_table]
                                           [--name REGEX] [--dry_run]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIME_LOG = "experiments/reproduce/run_jobs.log"
EXTERNAL = {"experiments/launch/mtgnn_run.py": "experiments/external/mtgnn/src",
            "experiments/launch/cosine_run.py": "experiments/external/cosine/src",
            "experiments/launch/nri_original_run.py": "experiments/external/nri/patched"}


def select(jobs, used_by=None, name=None):
    keep = []
    for j in jobs:
        if used_by and not set(used_by) & set(j["used_by"]):
            continue
        if name and not re.search(name, j["name"]):
            continue
        keep.append(j)
    return keep


def command(job, device):
    idx = device.split(":")[1] if device.startswith("cuda:") else "0"
    return [sys.executable] + [c.replace("{device}", device).replace("{device_index}", idx) for c in job["cmd"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpus", default="0", help="comma-separated CUDA device indices; empty string = run GPU jobs on CPU")
    ap.add_argument("--per_gpu", type=int, default=2)
    ap.add_argument("--cpu", type=int, default=4, help="parallel CPU jobs (ARNI, Granger)")
    ap.add_argument("--used_by", default=None, help="comma-separated analysis script names (see jobs.json)")
    ap.add_argument("--name", default=None, help="regular expression on the job name")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)
    jobs = json.load(open(os.path.join(ROOT, "experiments", "reproduce", "jobs.json")))
    jobs = select(jobs, a.used_by.split(",") if a.used_by else None, a.name)
    todo, missing_ext = [], set()
    for j in jobs:
        if os.path.exists(os.path.join(ROOT, j["out_dir"], "metrics.json")):
            continue
        if j["data"] and not os.path.exists(os.path.join(ROOT, j["data"])):     # data None: simulated by the job
            sys.exit("missing dataset %s: run experiments/reproduce/make_datasets.py first" % j["data"])
        ext = EXTERNAL.get(j["cmd"][0])
        if ext and not os.path.isdir(os.path.join(ROOT, ext)):
            missing_ext.add(ext)
            continue
        todo.append(j)
    print("%d selected, %d to run (%d already done)" % (len(jobs), len(todo), len(jobs) - len(todo) - sum(
        1 for j in jobs if EXTERNAL.get(j["cmd"][0]) in missing_ext)))
    for e in sorted(missing_ext):
        print("skipping jobs that need %s (see README.md)" % e)
    if a.dry_run:
        for j in todo:
            print(" ".join(command(j, "cuda:0" if j["gpu"] else "cpu")))
        return 0
    gpus = [g for g in a.gpus.split(",") if g != ""]
    slots = {("cuda:%s" % g, k): None for g in gpus for k in range(a.per_gpu)} or {("cpu", k): None for k in range(a.per_gpu)}
    cpu_slots = {("cpu", k): None for k in range(a.cpu)}
    pend_gpu = [j for j in todo if j["gpu"]]
    pend_cpu = [j for j in todo if not j["gpu"]]
    failed = []

    def fill(pool, pending):
        for key, cur in pool.items():
            if cur is not None:
                j, p, t0 = cur
                if p.poll() is None:
                    continue
                if p.returncode != 0 or not os.path.exists(os.path.join(ROOT, j["out_dir"], "metrics.json")):
                    failed.append(j["name"])
                print("[%s] %s" % ("done" if p.returncode == 0 else "FAILED rc=%d" % p.returncode, j["name"]), flush=True)
                with open(os.path.join(ROOT, TIME_LOG), "a") as f:          # wall times (read by scale_table.py)
                    f.write("[done] %s rc=%d %ds\n" % (j["out_dir"], p.returncode, round(time.time() - t0)))
                pool[key] = None
            if pool[key] is None and pending:
                j = pending.pop(0)
                os.makedirs(os.path.join(ROOT, j["out_dir"]), exist_ok=True)
                log = open(os.path.join(ROOT, j["out_dir"], "job.log"), "w")
                p = subprocess.Popen(command(j, key[0]), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, PYTHONPATH=ROOT))
                pool[key] = (j, p, time.time())

    while pend_gpu or pend_cpu or any(slots.values()) or any(cpu_slots.values()):
        fill(slots, pend_gpu)
        fill(cpu_slots, pend_cpu)
        time.sleep(2.0)
    print("finished; %d failed%s" % (len(failed), (": " + ", ".join(failed)) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
