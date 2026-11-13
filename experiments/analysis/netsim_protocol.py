"""NetSim protocol checks. Analysis only, no training, CPU only. Settings, outputs and the reading
rule below were fixed before any number of this script was computed.
Run from the project root:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=. python experiments/analysis/netsim_protocol.py \
      > results/supplement/netsim_protocol.txt
Raw data: experiments/real/netsim_raw/sim<k>.mat (ts = (S*T, N) subject-major, net = (S, N, N)), read exactly as
hetndl/gen/netsim.py reads it.

Part 1. Standardisation: training-subject statistics vs the all-subject statistics used now.
  hetndl/gen/netsim.py standardises each node with the mean and std (numpy, ddof 0, + 1e-12) over ALL 50 subjects and all
  times, test subjects included. For each of the 27 used simulations (1..28 without 13) and each of the three HetNDL
  training runs the paper reports (experiments/runs/sel/selected.json, keys netsim_sim<k>/add_z_signed_diff_lz0_seed<s>,
  s = 1, 2, 3; --seed and --n_traj read from the args of each selected run's metrics.json) the training subjects are
  keep[tr], keep = hetndl.data.select_trajectories(50, n_traj, seed), tr = hetndl.data.trajectory_split(len(keep), seed)[0],
  i.e. exactly what hetndl/train.py trains on. On the RAW ts (before any standardisation):
    m_tr, s_tr = per-node mean and std (ddof 0) over all time points of the training subjects,
    m_all, s_all = the same over all 50 subjects (the converter's statistics).
  Per simulation, over nodes and the three seeds: dmean = max |m_tr - m_all| / s_all, and the range [min, max] of
  s_tr / s_all; also the number of training subjects per seed and n_traj. Summary over the 27 sims: max and median of the
  per-sim dmean, overall range of the ratio. Meaning: the training inputs of the current pipeline equal a train-only
  standardisation up to a per-node offset of (m_tr - m_all) / s_all std units and a per-node scale factor s_tr / s_all.
  Descriptive context (not part of the rule): between-subject spread of the per-subject node means, max over nodes of
  std_s(subject mean) / s_all. Sanity checks: the npz data equal (ts - m_all) / s_all (max abs difference printed); the
  NOGI runs (netsim_rk4sub/netsim_sim<k>/orig_seed<s>) have the same seed and n_traj as the selected runs (count printed).
  Reading rule: NEGLIGIBLE if over all 27 x 3 (sim, seed) cells and all nodes dmean <= 0.10 and 0.90 <= ratio <= 1.10;
  otherwise NOT NEGLIGIBLE, offending sims listed. A tighter tier (dmean <= 0.05 and ratio in [0.95, 1.05]) is counted,
  descriptive only. No retraining: the check quantifies the difference only.

Part 2. Per-subject ground truth, every simulation 1..28 (13 included to show why it is skipped).
  For each subject s = 0..S-1: off-diagonal support supp_s = (net[s] != 0) off the diagonal (diagonal = self-decay, dropped
  as in the converter); sign = np.sign(net[s]). Per simulation:
    subjects whose support differs from net[0]'s; differing off-diagonal entries summed over subjects; number of distinct
    supports; subjects / entries whose sign differs on entries present in both supports; subjects whose full matrix
    (diagonal and weights included) equals net[0] exactly; for each true edge (off-diagonal support of net[0]) the
    coefficient of variation of its weight across the S subjects, CV = std_s / |mean_s| (ddof 0; a subject lacking the
    edge enters with weight 0): mean and max CV over edges; min and max true-edge weight over edges and subjects.
  Reading rule: "one shared graph = net[0] support" is EXACT for the 27 used sims iff in none of them any subject's
  off-diagonal support or sign differs from net[0]; otherwise NOT EXACT, sims listed. Weight variation is descriptive
  (the binary label and Table 2's scores do not depend on it)."""
import json
import os

import numpy as np
import scipy.io

from hetndl.data import select_trajectories, trajectory_split
from hetndl.gen.kuramoto import load_dataset

RAW = "experiments/real/netsim_raw/sim%d.mat"
SIMS = [k for k in range(1, 29) if k != 13]
SEL = json.load(open("experiments/runs/sel/selected.json"))
RULE = (0.10, 0.90, 1.10)
TIGHT = (0.05, 0.95, 1.05)


def load_raw(k):
    m = scipy.io.loadmat(RAW % k)
    N, S, T = (int(m[x][0, 0]) for x in ("Nnodes", "Nsubjects", "Ntimepoints"))
    ts = np.asarray(m["ts"], dtype=np.float64).reshape(S, T, N)
    return ts, np.asarray(m["net"], dtype=np.float64)


def run_args(k, s):
    r = SEL["netsim_sim%d/add_z_signed_diff_lz0_seed%d" % (k, s)]["run"]
    a = json.load(open(os.path.join(r, "metrics.json")))["args"]
    assert os.path.basename(a["data"]) == "netsim_sim%d.npz" % k and a["split"] == "traj", r
    assert a.get("noise_snr") is None and a.get("noise_std") is None and not a.get("smooth_window"), r
    return int(a["seed"]), a["n_traj"]


def part1():
    print("PART 1. Standardisation: per-node statistics of the TRAINING subjects vs the all-subject statistics of the converter")
    print("dmean = max over nodes and the 3 seeds of |m_tr - m_all| / s_all; ratio = s_tr / s_all (min..max over nodes and seeds);")
    print("subj.spread = max over nodes of std over subjects of the per-subject node mean / s_all (context only)")
    print("%-6s %3s %5s %7s %-9s %8s %-15s %11s %9s" % ("sim", "N", "T", "n_traj", "n_train", "dmean", "ratio", "subj.spread", "npz diff"))
    cells, per_sim, nogi_mismatch, nogi_checked = [], {}, 0, 0
    for k in SIMS:
        ts, _ = load_raw(k)
        S, T, N = ts.shape
        flat = ts.reshape(-1, N)
        m_all, s_all = flat.mean(0), flat.std(0) + 1e-12
        npz = load_dataset("experiments/data/netsim_sim%d.npz" % k)["data"][..., 0]
        npz_diff = float(np.abs(npz - (ts - m_all) / s_all).max())
        spread = float((ts.mean(1).std(0) / s_all).max())
        dms, rlo, rhi, ntr, nts = [], [], [], [], set()
        for s in (1, 2, 3):
            seed, nt = run_args(k, s)
            p = "experiments/runs/netsim_rk4sub/netsim_sim%d/orig_seed%d/metrics.json" % (k, s)
            if os.path.exists(p):
                a = json.load(open(p))["args"]; nogi_checked += 1
                nogi_mismatch += (int(a["seed"]), a["n_traj"]) != (seed, nt)
            keep = select_trajectories(S, nt, seed)
            tr, _, _ = trajectory_split(len(keep), seed)
            x = ts[keep[tr]].reshape(-1, N)
            dm = np.abs(x.mean(0) - m_all) / s_all
            ratio = x.std(0) / s_all
            cells.append((k, seed, float(dm.max()), float(ratio.min()), float(ratio.max())))
            dms.append(dm.max()); rlo.append(ratio.min()); rhi.append(ratio.max()); ntr.append(len(tr)); nts.add(nt)
        per_sim[k] = (max(dms), min(rlo), max(rhi))
        print("%-6s %3d %5d %7s %-9s %8.4f %6.4f..%6.4f %11.4f %9.1e" % (
            "sim%d" % k, N, T, "/".join(str(v) for v in sorted(nts, key=str)), "/".join(str(v) for v in ntr),
            max(dms), min(rlo), max(rhi), spread, npz_diff))
    d = np.array([v[0] for v in per_sim.values()])
    lo = min(v[1] for v in per_sim.values()); hi = max(v[2] for v in per_sim.values())
    print("\nsummary over %d sims x 3 seeds: per-sim dmean max %.4f, median %.4f; ratio range %.4f..%.4f" % (len(per_sim), d.max(), np.median(d), lo, hi))
    print("NOGI runs checked for the same seed / n_traj: %d, mismatches: %d" % (nogi_checked, nogi_mismatch))

    def ok(c, t):
        return c[2] <= t[0] and t[1] <= c[3] and c[4] <= t[2]
    bad = sorted({c[0] for c in cells if not ok(c, RULE)})
    tight = sum(ok(c, TIGHT) for c in cells)
    print("cells within the tight tier (dmean <= %.2f, ratio in [%.2f, %.2f]): %d of %d" % (*TIGHT, tight, len(cells)))
    if bad:
        print("VERDICT part 1: NOT NEGLIGIBLE by the pre-set rule (dmean <= %.2f and ratio in [%.2f, %.2f] in every cell); "
              "offending sims: %s" % (*RULE, ", ".join("sim%d" % k for k in bad)))
        for c in cells:
            if c[0] in bad and not ok(c, RULE):
                print("   sim%d seed %d: dmean %.4f, ratio %.4f..%.4f" % c)
    else:
        print("VERDICT part 1: NEGLIGIBLE by the pre-set rule (dmean <= %.2f and ratio in [%.2f, %.2f] in every cell)" % RULE)


def part2():
    print("\nPART 2. Per-subject ground truth net[s] vs net[0] (off-diagonal support and sign); weights of the net[0] edges across subjects")
    print("%-6s %3s %4s %4s %-12s %-13s %-9s %-11s %-12s %-9s %-9s %s" % (
        "sim", "S", "N", "E", "subj supp≠", "entries supp≠", "distinct", "subj sign≠", "entries sgn≠", "identical",
        "CV mean", "CV max   weight range"))
    bad = []
    for k in range(1, 29):
        _, net = load_raw(k)
        S, N = net.shape[0], net.shape[1]
        off = ~np.eye(N, dtype=bool)
        sup = (net != 0) & off
        dsup = sup != sup[0]
        n_sub, n_ent = int((dsup.sum((1, 2)) > 0).sum()), int(dsup.sum())
        distinct = len({tuple(m.ravel()) for m in sup})
        sgn = sup & sup[0] & (np.sign(net) != np.sign(net[0]))
        s_sub, s_ent = int((sgn.sum((1, 2)) > 0).sum()), int(sgn.sum())
        ident = sum(np.array_equal(net[s], net[0]) for s in range(S))
        E = sup[0]
        w = net[:, E]                                                    # (S, E)
        cv = w.std(0) / np.abs(w.mean(0))
        wt = w[w != 0]
        tag = " (skipped by the converter)" if k == 13 else ""
        print("%-6s %3d %4d %4d %-12d %-13d %-9d %-11d %-12d %-9d %-9.4f %-8.4f %.3f..%.3f%s" % (
            "sim%d" % k, S, N, int(E.sum()), n_sub, n_ent, distinct, s_sub, s_ent, ident, cv.mean(), cv.max(), wt.min(), wt.max(), tag))
        if k != 13 and (n_sub or s_sub):
            bad.append(k)
    if bad:
        print("VERDICT part 2: 'one shared graph = net[0] support' is NOT EXACT for the 27 used sims; differing sims: %s"
              % ", ".join("sim%d" % k for k in bad))
    else:
        print("VERDICT part 2: 'one shared graph = net[0] support' is EXACT for the 27 used sims (no subject's off-diagonal "
              "support or sign differs from net[0]); edge weights may still vary across subjects (CV columns)")


if __name__ == "__main__":
    part1()
    part2()
