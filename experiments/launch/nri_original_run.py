"""The original NRI code (Kipf et al., ICML 2018, "Neural Relational Inference for Interacting Systems"; code
github.com/ethanfetaya/NRI, commit e63fcb0, MIT license) on our data and in the regime of the NRI paper. The code is
not part of this repository: README.md says how to fetch it and apply experiments/external/nri/patches.diff (current
PyTorch, a loader for our input files, the test posteriors saved as test_prob.npy / test_rel.npy). This runner writes
the input files into experiments/external/nri/patched/data/ and starts its train.py there.

Configuration of every run: CNN encoder, MLP decoder, --skip-first (edge type 0 = no interaction, so that the probability
of type 1 is an edge score that needs no truth), two edge types, hidden 256, learning rate 5e-4 (halved every 200
epochs, the code's default), output variance 5e-5, 10 prediction steps, windows of 49 steps (--timesteps); the code's
loader scales every feature to [-1, 1] with the training minimum and maximum; snapshot = the code's lowest validation
NLL. Three modes:
  regime  NRI's own Kuramoto setting (NRI paper, App. B.3): a new random graph for every simulation
          (hetndl.gen.kuramoto.simulate_kuramoto_nri_regime: N = 5, each pair connected with probability 0.5, k_ij = 1,
          omega ~ U[1, 10), initial phases ~ U[0, 2 pi), states every 0.1), --sims simulations of 100 states split
          5 : 1 : 1 into training, validation and test (70,000 = 50,000 / 10,000 / 10,000, as in the paper); the paper's
          inputs per node and step (d theta / dt by forward difference, sin theta, omega); training and validation on
          the first 50 states (49 steps), test on all 100. Scored per test simulation against its own graph
          (experiments/analysis/nri_regime.py; edge accuracy is the metric the NRI paper reports).
  fixed   the same inputs and configuration on one fixed graph per dataset (kura5_nri_seed<k>: 10,000 trajectories of
          50 states), split hetndl.data.trajectory_split(10,000, --seed); the test windows give edge probabilities only.
  bench   our benchmarks, with the protocol of our NRI runs wherever the code allows: the trajectories and the split of
          the NRI run of the same cell (hetndl.data.select_trajectories with --n_traj, trajectory_split, both with
          --seed), the observed states as input ((cos, sin) on Kuramoto, the phase rule of every neural method; raw
          elsewhere), cut into non-overlapping windows of 49 states. The edge labels handed to the code are all zero
          (it uses them only for its logged accuracy), so no run sees the true graph. Score
          (experiments/analysis/nri_official.py): the mean probability of edge type 1 over the test windows.
--batch is the batch size we used (the one of our NRI run of the cell; 16 at N = 60 and on NetSim sim4, where 32 did not
fit into the memory of our GPUs); if a run does not fit, it restarts from scratch with half the batch (down to 2), and
the batch actually used is written to batch.txt. The run directory receives the code's own exp<time stamp>/ folder
(model, log.txt, test_prob.npy, test_rel.npy), attempt_b<batch>.log, batch.txt, done and metrics.json (the arguments of
this runner).
Usage (repository root):
  python experiments/launch/nri_original_run.py --mode bench --data experiments/data/a1v2_N30_seed1.npz --seed 1 \
      --epochs 600 --batch 32 --device cuda:0 --out experiments/runs/nri_official/a1v2_N30_seed1/seed1
  add --prepare_only to write the input files without training."""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from hetndl.data import phase_features, select_trajectories, trajectory_split          # noqa: E402
from hetndl.gen.kuramoto import load_dataset, simulate_kuramoto_nri_regime              # noqa: E402

OFF = os.path.join(ROOT, "experiments", "external", "nri", "patched")
TS = 49                                                    # --timesteps of the original code


def nri_feats(theta, omega, dt):
    """(S, N, T-1, 3): forward-difference velocity, sine, omega; theta (S, T, N), omega (S, N) or (N,)."""
    v = np.diff(theta, axis=1) / dt; s = np.sin(theta[:, :-1])
    w = np.broadcast_to((omega if omega.ndim == 2 else omega[None])[:, None, :], v.shape)
    return np.stack([v, s, w], -1).transpose(0, 2, 1, 3).astype(np.float32)


def write_nri_inputs(suffix, theta, omega, A, dt, splits, test_len):
    """The NRI-paper inputs: training / validation on the first 50 states, test on the first test_len."""
    tr, va, te = splits
    for name, idx, L in (("train", tr, 50), ("valid", va, 50), ("test", te, test_len)):
        np.save(os.path.join(OFF, "data", "feat_%s%s.npy" % (name, suffix)),
                nri_feats(theta[idx, :L], omega[idx] if omega.ndim == 2 else omega, dt))
        np.save(os.path.join(OFF, "data", "edges_%s%s.npy" % (name, suffix)),
                (A[idx] if A.ndim == 3 else np.broadcast_to(A, (len(idx),) + A.shape)).astype(np.int64))
    return A.shape[-1], 3


def prepare_regime(sims, sim_seed):
    suffix = "_kuramoto5_graph_per_simulation_n%d_s%d" % (sims, sim_seed)
    d = simulate_kuramoto_nri_regime(sims, N=5, p=0.5, T=10.0, dt=0.1, seed=sim_seed)
    idx = np.arange(sims); n_tr = round(sims * 5 / 7); n_va = round(sims / 7)
    return suffix, write_nri_inputs(suffix, d["data"][..., 0].astype(np.float64), d["omega_all"], d["A_all"], 0.1,
                                    (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]), 100)


def prepare_fixed(data, seed):
    ds = os.path.splitext(os.path.basename(data))[0]; suffix = "_kuramoto5_fixed_%s_s%d" % (ds, seed)
    f = load_dataset(data); assert f["meta"]["system"] == "kuramoto" and f["data"].shape[1] == 50, data
    return suffix, write_nri_inputs(suffix, f["data"][..., 0].astype(np.float64), f["omega"], f["A"],
                                    float(f["meta"]["dt"]), trajectory_split(f["data"].shape[0], seed), 50)


def prepare_bench(data, seed, n_traj):
    ds = os.path.splitext(os.path.basename(data))[0]; suffix = "_tab2_%s_s%d" % (ds, seed)
    d = load_dataset(data)
    keep = select_trajectories(d["data"].shape[0], n_traj, seed)
    X = phase_features(np.asarray(d["data"][keep], np.float32), "trig" if d["meta"].get("system") == "kuramoto" else "raw")
    tr, va, te = trajectory_split(len(keep), seed); N = X.shape[2]
    for name, idx in (("train", tr), ("valid", va), ("test", te)):
        w = [X[i, a0:a0 + TS].transpose(1, 0, 2) for i in idx for a0 in range(0, X.shape[1] - TS + 1, TS)]
        np.save(os.path.join(OFF, "data", "feat_%s%s.npy" % (name, suffix)), np.stack(w).astype(np.float32))
        np.save(os.path.join(OFF, "data", "edges_%s%s.npy" % (name, suffix)), np.zeros((len(w), N, N), np.int64))
    return suffix, (N, X.shape[3])


def train_args(suffix, N, dims, epochs, batch, seed, out, cpu=False):
    """Arguments of the original train.py."""
    return (["--suffix", suffix, "--encoder", "cnn", "--decoder", "mlp", "--skip-first", "--edge-types", "2", "--dims",
             str(dims), "--timesteps", str(TS), "--num-atoms", str(N), "--epochs", str(epochs), "--batch-size", str(batch),
             "--lr", "0.0005", "--encoder-hidden", "256", "--decoder-hidden", "256", "--prediction-steps", "10", "--var",
             "5e-5", "--seed", str(seed), "--save-folder", out] + (["--no-cuda"] if cpu else []))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=["regime", "fixed", "bench"])
    ap.add_argument("--data", default=None, help="dataset (.npz) of the fixed and bench modes")
    ap.add_argument("--sims", type=int, default=70000, help="regime: number of simulations")
    ap.add_argument("--sim_seed", type=int, default=2001, help="regime: seed of the simulations")
    ap.add_argument("--seed", type=int, required=True, help="seed of the original code; fixed / bench: also the split")
    ap.add_argument("--n_traj", type=int, default=None, help="bench: trajectory cap of our NRI run of the cell")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True, help="batch size; halved on out-of-memory")
    ap.add_argument("--device", default="cuda:0", help="cuda:<k> or cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepare_only", action="store_true", help="write the input files, do not train")
    a = ap.parse_args(argv)
    assert (a.mode == "regime") == (a.data is None), "--data is needed by (and only by) the fixed and bench modes"
    return a


def main(argv=None):
    a = parse(argv)
    if not os.path.isfile(os.path.join(OFF, "train.py")):
        sys.exit("missing %s/train.py: fetch the original NRI code and apply the patch (README.md)" % OFF)
    os.makedirs(os.path.join(OFF, "data"), exist_ok=True)
    if a.mode == "regime":
        suffix, (N, dims) = prepare_regime(a.sims, a.sim_seed)
    elif a.mode == "fixed":
        suffix, (N, dims) = prepare_fixed(a.data, a.seed)
    else:
        suffix, (N, dims) = prepare_bench(a.data, a.seed, a.n_traj)
    print("input files", suffix, "N", N, "dims", dims, flush=True)
    if a.prepare_only:
        return 0
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    cpu = a.device == "cpu"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if cpu else a.device.split(":")[-1])
    for b in [b for b in (a.batch, a.batch // 2, a.batch // 4, a.batch // 8) if b >= 2]:
        log = os.path.join(out, "attempt_b%d.log" % b)
        with open(log, "w") as f:
            rc = subprocess.run([sys.executable, "train.py"] + train_args(suffix, N, dims, a.epochs, b, a.seed, out, cpu),
                                cwd=OFF, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0 and "out of memory" in open(log).read():
            print("out of memory at batch %d" % b, flush=True)
            continue
        if rc != 0:
            sys.exit("train.py failed (exit code %d), see %s" % (rc, log))
        exp = [p for p in glob.glob(os.path.join(out, "exp*")) if os.path.exists(os.path.join(p, "test_prob.npy"))]
        assert len(exp) == 1, exp
        open(os.path.join(out, "batch.txt"), "w").write("%d\n" % b)
        open(os.path.join(out, "done"), "w").close()
        json.dump(dict(args=vars(a), batch_used=b, suffix=suffix, exp_dir=os.path.basename(exp[0])),
                  open(os.path.join(out, "metrics.json"), "w"), indent=1)
        return 0
    sys.exit("out of memory down to batch %d" % max(2, a.batch // 8))


if __name__ == "__main__":
    sys.exit(main())
