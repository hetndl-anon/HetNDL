"""COSINE baseline (Liang, Zhang, Pan, Lü 2026, arXiv 2604.12806, "Interpretable Relational Inference with LLM-Guided
Symbolic Dynamics Modeling"; code = the anonymous repository COSINE-6D43, kept unmodified in
experiments/external/cosine/src and run through experiments/external/cosine/cosine_entry.py) on one HetNDL dataset.
Protocol:
  * same trajectory split as every method (hetndl.data.trajectory_split, seed = data seed) -> COSINE's x_tr / x_va /
    x_te; full trajectory length; raw states, as given to the other baselines.
  * the graph handed to COSINE is ALL ZEROS. COSINE uses the true graph only for its logged metrics, for its AUC-based
    early stop (convergence_auc) and for its post-hoc round choice (select_best_round wants AUC >= 0.9); with zeros the
    AUC is nan and all three are inactive, so no truth reaches the run. The round is chosen here by COSINE's own
    validation score (val NLL + beta_kl KL + lambda_W |W|, the _is_better rule of its discovery loop).
  * LLM: qwen3:32b on the local Ollama server (COSINE's default gpt-oss:20b needs a newer Ollama than the installed
    0.9.6). The LLM sees validation NLL and term weights only.
  * everything else = COSINE defaults (10 rounds x 800 epochs, <= 10 terms per stream, stop when val MSE <= 1e-5,
    variance 5e-7, beta_kl 0.1, lambda_W 0.1, batch 64, lr 5e-3 / 1e-1, Tstep 2; its update is x + phi W, dt unused).
  * score: AUROC / AUPRC of the learned edge probabilities with hetndl.metrics.auroc_auprc (COSINE's A_soft[i, j] is
    the weight of j -> i, transposed to our W[j, i]); COSINE learns no edge signs.
Output runs/cosine/<dataset>/seed<k>/{metrics.json, W.npy, rounds.json, cosine_stdout.log}.
Usage: python experiments/launch/cosine_run.py --data <npz> --seed k --device_id g [--extra "<COSINE args>"]
(run from the project root with the hetndl env; COSINE itself runs in its own environment: set COSINE_PYTHON to that interpreter)."""
import argparse, glob, json, os, pickle, shlex, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hetndl.data import select_trajectories, trajectory_split
from hetndl.gen.kuramoto import load_dataset
from hetndl.metrics import auroc_auprc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(ROOT, "experiments", "external", "cosine"); SRC = os.path.join(EXT, "src")
CPY = os.environ.get("COSINE_PYTHON", "python")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device_id", default="0"); ap.add_argument("--out", default=None)
    ap.add_argument("--n_traj", type=int, default=None, help="trajectory cap, same rule as NRI / AIDD on Netsim")
    ap.add_argument("--llm_model", default="qwen3:32b"); ap.add_argument("--llm_base_url", default="http://localhost:11434/v1/")
    ap.add_argument("--extra", default="", help="extra COSINE arguments (smoke tests only)")
    a = ap.parse_args(argv)
    a.device_id = str(a.device_id).split(":")[-1]           # also accepts "cuda:1" (job queues fill in a torch device)
    ds = os.path.splitext(os.path.basename(a.data))[0]
    out = a.out or os.path.join(ROOT, "experiments", "runs", "cosine", ds, "seed%d" % a.seed)
    os.makedirs(out, exist_ok=True)
    d = load_dataset(a.data); X = np.asarray(d["data"], np.float32)
    keep = select_trajectories(X.shape[0], a.n_traj, a.seed); X = X[keep]
    tr, va, te = trajectory_split(len(keep), a.seed)
    N, T = X.shape[2], X.shape[1]
    suffix = "HetNDL_%s_seed%d" % (ds, a.seed); run_name = "r%d" % int(time.time())   # fresh round directory per call
    with open(os.path.join(SRC, "data", suffix + ".pickle"), "wb") as f:
        pickle.dump((X[tr], X[va], X[te], np.zeros((N, N), np.float32)), f)          # zeros: no truth inside COSINE
    cmd = [CPY, os.path.join(EXT, "cosine_entry.py"), "--suffix", suffix, "--tr-num", str(len(tr)), "--va-num", str(len(va)),
           "--te-num", str(len(te)), "--trajr-length", str(T), "--dt", str(float(d["meta"]["dt"])), "--device-id", str(a.device_id),
           "--seed", str(a.seed), "--save-folder", os.path.join(out, "cosine_results"), "--run-name", run_name,
           "--llm-model", a.llm_model, "--llm-base-url", a.llm_base_url, "--convergence-auc", "2.0", "--skip-visual"] + shlex.split(a.extra)
    t0 = time.time()
    with open(os.path.join(out, "cosine_stdout.log"), "w") as lf:
        lf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n"); lf.flush()
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=SRC).returncode
    rounds = []
    for mp in sorted(glob.glob(os.path.join(SRC, "configs", "basis_specs", suffix + "_" + run_name, "rounds", "round_*", "metrics_best.json"))):
        m = json.load(open(mp)); rd = os.path.dirname(mp)
        spec = rd + ".json"
        rounds.append(dict(round=os.path.basename(rd), val_score=m.get("val_score"), val_nll=m.get("val_nll"), mse=m.get("mse"),
                           best_epoch=m.get("best_epoch"), dir=rd, spec=json.load(open(spec)) if os.path.exists(spec) else None))
    ok = [r for r in rounds if isinstance(r["val_score"], (int, float)) and np.isfinite(r["val_score"])
          and os.path.exists(os.path.join(r["dir"], "A_soft.npy"))]
    json.dump(rounds, open(os.path.join(out, "rounds.json"), "w"), indent=1, default=str)
    if not ok:
        json.dump(dict(method="cosine", failed=True, reason="no finished round (rc %d)" % rc, args=vars(a)),
                  open(os.path.join(out, "metrics.json"), "w"), indent=1)
        print("FAILED", ds, a.seed, rc); return
    best = min(ok, key=lambda r: r["val_score"])
    W = np.load(os.path.join(best["dir"], "A_soft.npy")).T                         # A_soft[i, j] = j -> i
    auroc, auprc = auroc_auprc(W, d["A"])
    np.save(os.path.join(out, "W.npy"), W)
    m = dict(method="cosine", auroc=auroc, auprc=auprc, sign_acc=float("nan"), chosen_round=best["round"], run_name=run_name,
             val_score=best["val_score"], n_rounds=len(rounds), rc=rc, wall_sec=time.time() - t0, llm=a.llm_model,
             n_traj=len(keep), args=vars(a))
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}, default=float))


if __name__ == "__main__":
    main()
