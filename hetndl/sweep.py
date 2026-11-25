"""Dataset grids and a tolerant run queue over several GPUs.

A failing job is recorded in queue.csv and never stops the queue. Jobs whose
out_dir already holds metrics.json are skipped (resume support).
"""
import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MOTIV_MODELS = ("orig", "add", "add_z", "add_z_signed")
# Specs of the earliest runs, made when hetndl.train defaulted to lambda_z = 1e-4 (now 0).
LEGACY_LAMBDA_Z_SPECS = ("motiv", "a1", "a1v2", "replication", "diffusive")
# One node-latent dimension for every benchmark. Earlier specs used d_z = 1 or 2 depending on the
# KIND of heterogeneity the generator uses — knowledge of the generating process. Legacy specs keep their historical
# values only so that old runs stay reproducible.
D_Z = 2
# Specs that share another spec's runs/ folder (default run_name).
SHARED_RUN_DIR = {"a2n60": "a2", "grid_restart": "grid"}


def motiv_priority(sigma: float, neg: float, model: str) -> int:
    """0 = decides the track (run first), 1 = fills the curves, 2 = completeness."""
    if neg > 0:
        return 0 if model in ("orig", "add_z_signed") else (1 if model == "add_z" else 2)
    if model in ("orig", "add_z"):
        return 0 if sigma in (0.0, 0.3, 2.0) else 1
    return 1 if sigma in (0.0, 0.3) else 2


def _kura_gen(path: str, N: int, K: float, sigma: float, neg: float, clip: bool, samples: int,
              seed: int, graph_seed: int, graph: str = "ER", T: float = 10.0, dt: float = 0.01, m: int = 1) -> List[str]:
    cmd = [PY, "-m", "hetndl.gen.kuramoto", "--N", str(N), "--graph", graph, "--K", str(K),
           "--sigma", str(sigma), "--neg_frac", str(neg), "--samples", str(samples), "--T", str(T),
           "--dt", str(dt), "--seed", str(seed), "--graph_seed", str(graph_seed), "--out", path]
    cmd.append("--clip" if clip else "--no_clip")
    if graph == "BA":
        cmd += ["--m", str(m)]
    return cmd


def _train(data: str, model: str, out: str, epochs: int, seed: int, extra: List[str], **kw) -> List[str]:
    cmd = [PY, "-m", "hetndl.train", "--data", data, "--model", model, "--out", out,
           "--epochs", str(epochs), "--seed", str(seed), "--device", "{device}"]
    for k, v in kw.items():
        cmd += ["--" + k, str(v)]
    return cmd + list(extra)


def build_jobs(spec: str, root: str, seeds: List[int], epochs: int, extra: List[str],
               run_name: Optional[str] = None) -> List[Dict]:
    data_dir = os.path.join(root, "data")
    runs = os.path.join(root, "runs", run_name or SHARED_RUN_DIR.get(spec, spec))
    jobs: List[Dict] = []
    if spec == "motiv":
        settings = [(s, 0.0) for s in (0.0, 0.3, 0.6, 1.0, 2.0)] + [(0.3, 0.2), (0.3, 0.5)]
        for sigma, neg in settings:
            name = "kura_ER100_K2_s%g_n%g" % (sigma, neg)
            path = os.path.join(data_dir, name + ".npz")
            gen = _kura_gen(path, 100, 2.0, sigma, neg, False, 20, 1, 1)
            for model in MOTIV_MODELS:
                for seed in seeds:
                    out = os.path.join(runs, name, "%s_seed%d" % (model, seed))
                    jobs.append(dict(name="%s/%s_seed%d" % (name, model, seed), dataset=name, dataset_path=path, gen_cmd=gen,
                                     model=model, seed=seed, out_dir=out, priority=motiv_priority(sigma, neg, model),
                                     train_cmd=_train(path, model, out, epochs, seed, extra, d_z=1, hidden=16,
                                                      lambda_l1="1e-8", time_range="dt", mask_diag=1)))
        jobs.sort(key=lambda j: j["priority"])  # stable: keeps dataset order within a priority class
    elif spec in ("a1", "a1v2"):
        # a1v2: short windows and more initial conditions so that most samples lie in the
        # transient where the three mechanisms differ, instead of near the common fixed point.
        v2 = spec == "a1v2"
        for seed in seeds:
            name = ("a1v2_N30_seed%d" if v2 else "a1_N30_seed%d") % seed
            path = os.path.join(data_dir, name + ".npz")
            gen = [PY, "-m", "hetndl.gen.a1_mixed", "--N", "30", "--p", "0.1", "--directed",
                   "--samples", "40" if v2 else "18", "--T", "4" if v2 else "10", "--dt", "0.02" if v2 else "0.05",
                   "--seed", str(seed), "--kappa", "0.5", "--out", path]
            for model in ("orig", "add_z"):
                out = os.path.join(runs, name, "%s_seed%d" % (model, seed))
                jobs.append(dict(name="%s/%s_seed%d" % (name, model, seed), dataset=name, dataset_path=path, gen_cmd=gen,
                                 model=model, seed=seed, out_dir=out,
                                 train_cmd=_train(path, model, out, epochs, seed, extra, d_z=2, hidden=16,
                                                  lambda_l1="1e-8", time_range="dt", mask_diag=1, cluster_k=3)))
    elif spec == "replication":
        name = "kura_orig_ER100_K5_s0.3_clip_seed1"
        path = os.path.join(data_dir, name + ".npz")
        gen = _kura_gen(path, 100, 5.0, 0.3, 0.0, True, 20, 1, 1)
        for seed in seeds:
            # orig_unit = the published protocol as closely as possible: [0,1] integration,
            # no diagonal mask, random split of pairs. orig_dt = the HetNDL protocol.
            for tag, tr, md, sp in (("orig_unit", "unit", 0, "pairs"), ("orig_dt", "dt", 1, "traj")):
                out = os.path.join(runs, name, "%s_seed%d" % (tag, seed))
                jobs.append(dict(name="%s/%s_seed%d" % (name, tag, seed), dataset=name, dataset_path=path, gen_cmd=gen,
                                 model="orig", seed=seed, out_dir=out,
                                 train_cmd=_train(path, "orig", out, epochs, seed, extra, d_z=1, hidden=16,
                                                  lambda_l1="1e-8", time_range=tr, mask_diag=md, split=sp)))
    elif spec == "diffusive":
        # Identifiability check: G(x_i,x_j) - G(x_i,x_i) removes the constant channel through
        # which node offsets leak into column sums of W. Kuramoto runs use trig features
        # (passed via extra); A1 states are not phases, so trig/rk4-specific flags are dropped.
        kura = [(0.3, 0.0, "add_z"), (1.0, 0.0, "add_z"), (2.0, 0.0, "add_z"),
                (0.3, 0.2, "add_z_signed"), (0.3, 0.5, "add_z_signed")]
        for sigma, neg, model in kura:
            name = "kura_ER100_K2_s%g_n%g" % (sigma, neg)
            path = os.path.join(data_dir, name + ".npz")
            gen = _kura_gen(path, 100, 2.0, sigma, neg, False, 20, 1, 1)
            for seed in seeds:
                out = os.path.join(runs, name, "%s_diff_seed%d" % (model, seed))
                jobs.append(dict(name="%s/%s_diff_seed%d" % (name, model, seed), dataset=name, dataset_path=path,
                                 gen_cmd=gen, model=model, seed=seed, out_dir=out,
                                 train_cmd=_train(path, model, out, epochs, seed, extra, d_z=1, hidden=16,
                                                  lambda_l1="1e-8", time_range="dt", mask_diag=1,
                                                  coupling="diffusive")))
        a1_extra = [e for e in extra if e not in ("--feat", "trig", "--method", "rk4")]
        for seed in seeds:
            name = "a1v2_N30_seed%d" % seed
            path = os.path.join(data_dir, name + ".npz")
            gen = [PY, "-m", "hetndl.gen.a1_mixed", "--N", "30", "--p", "0.1", "--directed", "--samples", "40",
                   "--T", "4", "--dt", "0.02", "--seed", str(seed), "--kappa", "0.5", "--out", path]
            out = os.path.join(runs, name, "add_z_diff_seed%d" % seed)
            jobs.append(dict(name="%s/add_z_diff_seed%d" % (name, seed), dataset=name, dataset_path=path, gen_cmd=gen,
                             model="add_z", seed=seed, out_dir=out,
                             train_cmd=_train(path, "add_z", out, 1000 if epochs == 600 else epochs, seed, a1_extra,
                                              d_z=2, hidden=16, lambda_l1="1e-8", time_range="dt", mask_diag=1,
                                              cluster_k=3, coupling="diffusive")))
    elif spec == "multiseed":
        # Replicates the lambda_z = 0 results. Seed k draws a new graph and new samples (seed 1 reuses the
        # existing datasets); HetNDL runs both free and diffusive coupling, orig is the NOGI baseline.
        kura = [(0.3, 0.0, "add_z"), (2.0, 0.0, "add_z"), (0.3, 0.2, "add_z_signed"), (0.3, 0.5, "add_z_signed")]
        common = dict(d_z=1, hidden=16, lambda_l1="1e-8", time_range="dt", mask_diag=1)
        for seed in seeds:
            for sigma, neg, model in kura:
                name = "kura_ER100_K2_s%g_n%g" % (sigma, neg) + ("" if seed == 1 else "_g%d" % seed)
                path = os.path.join(data_dir, name + ".npz")
                gen = _kura_gen(path, 100, 2.0, sigma, neg, False, 20, seed, seed)
                for tag, mdl, kw in (("orig", "orig", {}),
                                     ("%s_free_lz0" % model, model, dict(lambda_z=0, coupling="free")),
                                     ("%s_diff_lz0" % model, model, dict(lambda_z=0, coupling="diffusive"))):
                    out = os.path.join(runs, name, "%s_seed%d" % (tag, seed))
                    jobs.append(dict(name="%s/%s_seed%d" % (name, tag, seed), dataset=name, dataset_path=path,
                                     gen_cmd=gen, model=mdl, seed=seed, out_dir=out,
                                     train_cmd=_train(path, mdl, out, epochs, seed, extra, **common, **kw)))
        a1_extra = [e for e in extra if e not in ("--feat", "trig", "--method", "rk4")]
        for seed in seeds:
            name = "a1v2_N30_seed%d" % seed
            path = os.path.join(data_dir, name + ".npz")
            gen = [PY, "-m", "hetndl.gen.a1_mixed", "--N", "30", "--p", "0.1", "--directed", "--samples", "40",
                   "--T", "4", "--dt", "0.02", "--seed", str(seed), "--kappa", "0.5", "--out", path]
            for tag, mdl, kw in (("orig", "orig", {}), ("add_z_free_lz0", "add_z", dict(lambda_z=0, coupling="free")),
                                 ("add_z_diff_lz0", "add_z", dict(lambda_z=0, coupling="diffusive"))):
                out = os.path.join(runs, name, "%s_seed%d" % (tag, seed))
                jobs.append(dict(name="%s/%s_seed%d" % (name, tag, seed), dataset=name, dataset_path=path, gen_cmd=gen,
                                 model=mdl, seed=seed, out_dir=out,
                                 train_cmd=_train(path, mdl, out, 1000 if epochs == 600 else epochs, seed, a1_extra,
                                                  d_z=2, hidden=16, lambda_l1="1e-8", time_range="dt", mask_diag=1,
                                                  cluster_k=3, **kw)))
    elif spec in ("a2", "a2n60"):
        # Second mechanism benchmark: FHN / Van der Pol mixture on a shared (u, v) state, coupled on u.
        # a2n60: N = 60 with p halved, so the mean in-degree matches N = 30.
        a2_N, a2_p = (30, "0.1") if spec == "a2" else (60, "0.05")
        a2_extra = [e for e in extra if e not in ("--feat", "trig", "--method", "rk4")]
        for seed in seeds:
            name = "a2_N%d_seed%d" % (a2_N, seed)
            path = os.path.join(data_dir, name + ".npz")
            gen = [PY, "-m", "hetndl.gen.a2_mixed", "--N", str(a2_N), "--p", a2_p, "--directed", "--samples", "40",
                   "--T", "10", "--dt", "0.05", "--seed", str(seed), "--kappa", "0.3", "--out", path]
            for tag, mdl, kw in (("orig", "orig", {}), ("add_z_free_lz0", "add_z", dict(lambda_z=0, coupling="free")),
                                 ("add_z_diff_lz0", "add_z", dict(lambda_z=0, coupling="diffusive"))):
                out = os.path.join(runs, name, "%s_seed%d" % (tag, seed))
                jobs.append(dict(name="%s/%s_seed%d" % (name, tag, seed), dataset=name, dataset_path=path, gen_cmd=gen,
                                 model=mdl, seed=seed, out_dir=out,
                                 train_cmd=_train(path, mdl, out, 1000 if epochs == 600 else epochs, seed, a2_extra,
                                                  d_z=2, hidden=16, lambda_l1="1e-8", time_range="dt", mask_diag=1,
                                                  cluster_k=2, **kw)))
    elif spec == "a3":
        # Edge heterogeneity: A3 = A2 nodes with a source-class-dependent coupling form. Models: NOGI, ours
        # (diffusive, shared G), ours with endpoint-type coupling G(x_i, x_j, z_i, z_j) (edge_z) in diffusive
        # and free form; plus edge_z on A2 (homogeneous coupling) as a does-no-harm check.
        a3_extra = [e for e in extra if e not in ("--feat", "trig", "--method", "rk4")]
        common = dict(d_z=2, hidden=16, lambda_l1="1e-8", time_range="dt", mask_diag=1, cluster_k=2)
        for seed in seeds:
            name = "a3_N30_seed%d" % seed
            path = os.path.join(data_dir, name + ".npz")
            gen = [PY, "-m", "hetndl.gen.a3_mixed", "--N", "30", "--p", "0.1", "--directed", "--samples", "40",
                   "--T", "10", "--dt", "0.05", "--seed", str(seed), "--kappa", "0.3", "--out", path]
            a2name = "a2_N30_seed%d" % seed
            a2path = os.path.join(data_dir, a2name + ".npz")
            a2gen = [PY, "-m", "hetndl.gen.a2_mixed", "--N", "30", "--p", "0.1", "--directed", "--samples", "40",
                     "--T", "10", "--dt", "0.05", "--seed", str(seed), "--kappa", "0.3", "--out", a2path]
            plan = [(name, path, gen, "orig", "orig", {}),
                    (name, path, gen, "add_z_diff_lz0", "add_z", dict(lambda_z=0, coupling="diffusive")),
                    (name, path, gen, "add_z_diff_ez_lz0", "add_z", dict(lambda_z=0, coupling="diffusive", edge_z=1)),
                    (name, path, gen, "add_z_free_ez_lz0", "add_z", dict(lambda_z=0, coupling="free", edge_z=1)),
                    (a2name, a2path, a2gen, "add_z_diff_ez_lz0", "add_z", dict(lambda_z=0, coupling="diffusive", edge_z=1))]
            for ds, dpath, dgen, tag, mdl, kw in plan:
                out = os.path.join(runs, ds, "%s_seed%d" % (tag, seed))
                jobs.append(dict(name="%s/%s_seed%d" % (ds, tag, seed), dataset=ds, dataset_path=dpath, gen_cmd=dgen,
                                 model=mdl, seed=seed, out_dir=out,
                                 train_cmd=_train(dpath, mdl, out, 1000 if epochs == 600 else epochs, seed, a3_extra,
                                                  **common, **kw)))
    elif spec in ("netsim", "netsim_dopri5"):
        # Real-data benchmark of NOGI (Smith et al. 2011): 27 sims converted by hetndl.gen.netsim (sim13 skipped).
        # No generation step. Ours in the non-negative and in the signed form; the truth has no negative edges.
        # Fixed-step rk4 over the one-TR interval: dopri5 took 30-40 s/epoch on these noisy series. Long sims
        # (T 1200 / 2400 / 5000) are capped at about 20k (x_t, x_{t+1}) pairs by using 16 / 8 / 4 of the 50 subjects.
        ns_extra = [e for e in extra if e not in ("--feat", "trig", "--method", "rk4")]
        # netsim_dopri5: adaptive-solver top-ups for the sims where rk4-200ep looked conservative.
        sims, method = ((4, 7, 9), "dopri5") if spec == "netsim_dopri5" else (range(1, 29), "rk4")
        T_of = {5: 1200, 6: 1200, 7: 5000, 9: 5000, 19: 2400, 20: 2400, 25: 100, 26: 50, 27: 50, 28: 100}
        for k in sims:
            if k == 13:
                continue
            name = "netsim_sim%d" % k
            path = os.path.join(data_dir, name + ".npz")
            T = T_of.get(k, 200)
            cap = {} if 50 * (T - 1) <= 20000 else dict(n_traj=max(3, 20000 // (T - 1)))
            for seed in seeds:
                for tag, mdl, kw in (("orig", "orig", {}),
                                     ("add_z_diff_lz0", "add_z", dict(lambda_z=0, coupling="diffusive")),
                                     ("add_z_signed_diff_lz0", "add_z_signed", dict(lambda_z=0, coupling="diffusive"))):
                    out = os.path.join(runs, name, "%s_seed%d" % (tag, seed))
                    jobs.append(dict(name="%s/%s_seed%d" % (name, tag, seed), dataset=name, dataset_path=path, gen_cmd=None,
                                     model=mdl, seed=seed, out_dir=out,
                                     train_cmd=_train(path, mdl, out, epochs, seed, ns_extra, d_z=1, hidden=16,
                                                      lambda_l1="1e-8", time_range="dt", mask_diag=1, method=method,
                                                      **cap, **kw)))
    elif spec == "robust":
        # Data-size and noise curves (ours = diffusive lambda_z 0; signed on the signed graph), NOGI alongside.
        # Kuramoto p- 0.5 (N 100, 20 trajectories): budgets 3/5/10 trajectories, phase noise std 0.05/0.1/0.2 rad.
        # A2 N 60 (40 trajectories): budgets 5/10/20, SNR 30/20/10 dB. The full-data, noise-free cells already exist
        # (runs/multiseed, runs/lz0, runs/a2). Kuramoto keeps the trig / rk4 extras, A2 strips them.
        a2_extra = [e for e in extra if e not in ("--feat", "trig", "--method", "rk4")]
        for seed in seeds:
            kname = "kura_ER100_K2_s0.3_n0.5" + ("" if seed == 1 else "_g%d" % seed)
            kpath = os.path.join(data_dir, kname + ".npz")
            kgen = _kura_gen(kpath, 100, 2.0, 0.3, 0.5, False, 20, seed, seed)
            aname = "a2_N60_seed%d" % seed
            apath = os.path.join(data_dir, aname + ".npz")
            agen = [PY, "-m", "hetndl.gen.a2_mixed", "--N", "60", "--p", "0.05", "--directed", "--samples", "40",
                    "--T", "10", "--dt", "0.05", "--seed", str(seed), "--kappa", "0.3", "--out", apath]
            systems = [(kname, kpath, kgen, "add_z_signed", extra, 600 if epochs == 600 else epochs,
                        [("n%d" % n, dict(n_traj=n)) for n in (3, 5, 10)] + [("noise%g" % v, dict(noise_std=v)) for v in (0.05, 0.1, 0.2)],
                        dict(d_z=1, cluster_k=0)),
                       (aname, apath, agen, "add_z", a2_extra, 1000 if epochs == 600 else epochs,
                        [("n%d" % n, dict(n_traj=n)) for n in (5, 10, 20)] + [("snr%d" % v, dict(noise_snr=v)) for v in (30, 20, 10)],
                        dict(d_z=2, cluster_k=2))]
            for name, path, gen, ours, ex, ep, conds, kw0 in systems:
                for ctag, ckw in conds:
                    for tag, mdl, kw in (("orig_" + ctag, "orig", {}),
                                         ("%s_diff_lz0_%s" % (ours, ctag), ours, dict(lambda_z=0, coupling="diffusive"))):
                        out = os.path.join(runs, name, "%s_seed%d" % (tag, seed))
                        jobs.append(dict(name="%s/%s_seed%d" % (name, tag, seed), dataset=name, dataset_path=path, gen_cmd=gen,
                                         model=mdl, seed=seed, out_dir=out,
                                         train_cmd=_train(path, mdl, out, ep, seed, ex, hidden=16, lambda_l1="1e-8",
                                                          time_range="dt", mask_diag=1, **kw0, **kw, **ckw)))
    elif spec == "noise_sg":
        # Kuramoto p- 0.5 phase-noise grid with Savitzky-Golay smoothing of the observed series for EVERY method
        # (window = 5 % of the 1000-step trajectories = 51, fixed a priori). Training-level fix for the noise
        # overfitting: without it val MSE rises from epoch 1 and the val-MSE snapshot
        # lands on an untrained model. Supersedes the unsmoothed noise cells of spec robust (kept as appendix reference).
        for seed in seeds:
            kname = "kura_ER100_K2_s0.3_n0.5" + ("" if seed == 1 else "_g%d" % seed)
            kpath = os.path.join(data_dir, kname + ".npz")
            kgen = _kura_gen(kpath, 100, 2.0, 0.3, 0.5, False, 20, seed, seed)
            for v in (0.05, 0.1, 0.2):
                for tag, mdl, kw in (("orig_sg51_noise%g" % v, "orig", {}),
                                     ("add_z_signed_diff_lz0_sg51_noise%g" % v, "add_z_signed", dict(lambda_z=0, coupling="diffusive"))):
                    out = os.path.join(runs, kname, "%s_seed%d" % (tag, seed))
                    jobs.append(dict(name="%s/%s_seed%d" % (kname, tag, seed), dataset=kname, dataset_path=kpath, gen_cmd=kgen,
                                     model=mdl, seed=seed, out_dir=out,
                                     train_cmd=_train(kpath, mdl, out, epochs, seed, extra, hidden=16, lambda_l1="1e-8",
                                                      time_range="dt", mask_diag=1, d_z=1, cluster_k=0,
                                                      noise_std=v, smooth_window=51, **kw)))
    elif spec == "glv":
        # Second signed system: generalised Lotka-Volterra, N 30 directed ER p 0.1, half the edges negative,
        # c 0.5, 40 trajectories of T 3 (dt 0.03) in the transient. Coupling x_i x_j is NOT diffusive. NOGI, the signed
        # default (diffusive) and signed free coupling (ablation); d_z 1 (r_i), raw features, rk4, 1000 epochs.
        g_extra = [e for e in extra if e not in ("--feat", "trig")]
        for seed in seeds:
            name = "glv_N30_n0.5_seed%d" % seed
            path = os.path.join(data_dir, name + ".npz")
            gen = [PY, "-m", "hetndl.gen.glv", "--N", "30", "--p", "0.1", "--c", "0.5", "--neg_frac", "0.5",
                   "--samples", "40", "--T", "3", "--dt", "0.03", "--seed", str(seed), "--out", path]
            for tag, mdl, kw in (("orig", "orig", {}),
                                 ("add_z_signed_diff_lz0", "add_z_signed", dict(lambda_z=0, coupling="diffusive")),
                                 ("add_z_signed_free_lz0", "add_z_signed", dict(lambda_z=0, coupling="free"))):
                out = os.path.join(runs, name, "%s_seed%d" % (tag, seed))
                jobs.append(dict(name="%s/%s_seed%d" % (name, tag, seed), dataset=name, dataset_path=path, gen_cmd=gen,
                                 model=mdl, seed=seed, out_dir=out,
                                 train_cmd=_train(path, mdl, out, 1000 if epochs == 600 else epochs, seed, g_extra, d_z=1,
                                                  hidden=16, lambda_l1="1e-8", time_range="dt", mask_diag=1, **kw)))
    elif spec == "ba":
        # Graph-type robustness: scale-free BA(m = 2; mean degree ~4, close to the ER graphs) versions of
        # signed Kuramoto p- 0.5 (N 100) and A1 (N 30). NOGI and the signed default (add_z_signed + diffusive + lambda_z 0)
        # with the same training settings as their ER counterparts (Kuramoto: 600 epochs, rk4, trig via extra; A1: 1000
        # epochs, dopri5, raw features, d_z 2).
        a1_extra = [e for e in extra if e not in ("--feat", "trig", "--method", "rk4")]
        ours = dict(lambda_z=0, coupling="diffusive")
        for seed in seeds:
            kname = "kura_BA100_m2_K2_s0.3_n0.5_g%d" % seed
            kpath = os.path.join(data_dir, kname + ".npz")
            kgen = _kura_gen(kpath, 100, 2.0, 0.3, 0.5, False, 20, seed, seed, graph="BA", m=2)
            aname = "a1v2_BA30_m2_seed%d" % seed
            apath = os.path.join(data_dir, aname + ".npz")
            agen = [PY, "-m", "hetndl.gen.a1_mixed", "--N", "30", "--graph", "BA", "--m", "2", "--samples", "40",
                    "--T", "4", "--dt", "0.02", "--seed", str(seed), "--kappa", "0.5", "--out", apath]
            for name, path, gen, ep, ex, kw0 in ((kname, kpath, kgen, 600 if epochs == 600 else epochs, extra, dict(d_z=1)),
                                                 (aname, apath, agen, 1000 if epochs == 600 else epochs, a1_extra + ["--method", "dopri5"],
                                                  dict(d_z=2, cluster_k=3))):
                for tag, mdl, kw in (("orig", "orig", {}), ("add_z_signed_diff_lz0", "add_z_signed", ours)):
                    out = os.path.join(runs, name, "%s_seed%d" % (tag, seed))
                    jobs.append(dict(name="%s/%s_seed%d" % (name, tag, seed), dataset=name, dataset_path=path, gen_cmd=gen,
                                     model=mdl, seed=seed, out_dir=out,
                                     train_cmd=_train(path, mdl, out, ep, seed, ex, hidden=16, lambda_l1="1e-8",
                                                      time_range="dt", mask_diag=1, **kw0, **kw)))
    elif spec in ("grid", "grid_restart"):
        # Real topology: IEEE 118-bus swing dynamics (hetndl.gen.powergrid), 2-D state (theta, omega) with bounded
        # angles (raw features). Generators M = 1, loads M = 0.2; seeds change the perturbations, not the grid.
        case = os.path.join(root, "real", "matpower_raw", "case118.m")
        g_extra = [e for e in extra if e not in ("--feat", "trig")]   # keep the launcher's solver (--method rk4)
        # grid_restart: two more training restarts per cell on the SAME dataset (train seed 10 * data seed + r), so every
        # method is reported as best-of-3 restarts by validation MSE (one seed each of ours / NOGI diverged).
        restarts = [None] if spec == "grid" else [2, 3]
        for seed in seeds:
            name = "grid118_seed%d" % seed
            path = os.path.join(data_dir, name + ".npz")
            gen = [PY, "-m", "hetndl.gen.powergrid", "--case", case, "--kappa", "0.3", "--m_gen", "1.0", "--m_load", "0.2",
                   "--damping", "0.5", "--samples", "40", "--T", "10", "--dt", "0.05", "--seed", str(seed), "--out", path]
            for r in restarts:
                tseed, suffix = (seed, "") if r is None else (10 * seed + r, "_r%d" % r)
                for tag, mdl, kw in (("orig", "orig", {}), ("add_z_diff_lz0", "add_z", dict(lambda_z=0, coupling="diffusive")),
                                     ("add_z_free_lz0", "add_z", dict(lambda_z=0, coupling="free"))):
                    out = os.path.join(runs, name, "%s_seed%d%s" % (tag, seed, suffix))
                    jobs.append(dict(name="%s/%s_seed%d%s" % (name, tag, seed, suffix), dataset=name, dataset_path=path,
                                     gen_cmd=gen, model=mdl, seed=tseed, out_dir=out,
                                     train_cmd=_train(path, mdl, out, 1000 if epochs == 600 else epochs, tseed, g_extra,
                                                      d_z=2, hidden=16, lambda_l1="1e-8", time_range="dt", mask_diag=1,
                                                      cluster_k=2, **kw)))
    else:
        raise ValueError("unknown spec: %s" % spec)
    if spec not in LEGACY_LAMBDA_Z_SPECS:
        for j in jobs:                       # unified d_z for every z model (see D_Z)
            cmd = j["train_cmd"]
            if cmd[cmd.index("--model") + 1] in ("add_z", "add_z_signed") and "--d_z" in cmd:
                cmd[cmd.index("--d_z") + 1] = str(D_Z)
    if spec in LEGACY_LAMBDA_Z_SPECS:
        for j in jobs:
            cmd = j["train_cmd"]
            if "--lambda_z" not in cmd:  # insert before the user's extra args, which stay last
                i = cmd.index("{device}") + 1
                cmd[i:i] = ["--lambda_z", "1e-4"]
    return jobs


def mirror_job(run_dir: str, out_root: str, **overrides) -> Dict:
    """A job that reruns a finished run with its stored arguments (metrics.json "args") except `overrides`.
    Output goes to <out_root>/<dataset dir>/<run name with the model tag replaced>; None-valued args keep the default."""
    import json
    args = dict(json.load(open(os.path.join(run_dir, "metrics.json")))["args"])
    old_model = args["model"]; args.update(overrides)
    name = os.path.basename(os.path.normpath(run_dir))
    if "model" in overrides and name.startswith(old_model + "_"):
        name = overrides["model"] + name[len(old_model):]
    ds = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    out = os.path.join(out_root, ds, name)
    cmd = [PY, "-m", "hetndl.train"]
    for k, v in args.items():
        if k in ("out", "device") or v is None:
            continue
        cmd += ["--" + k, str(v)]
    cmd += ["--out", out, "--device", "{device}"]
    return dict(name="%s/%s" % (ds, name), dataset=ds, dataset_path=args["data"], gen_cmd=None, model=args["model"],
                seed=int(args["seed"]), out_dir=out, train_cmd=cmd)


def ensure_datasets(jobs: List[Dict], workers: int = 4) -> None:
    todo = {}
    for j in jobs:
        p = j.get("dataset_path")
        if p and j.get("gen_cmd") and not os.path.exists(p) and p not in todo:
            todo[p] = j["gen_cmd"]
    pending = list(todo.items())
    running = []
    while pending or running:
        while pending and len(running) < workers:
            p, cmd = pending.pop(0)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            print("[gen]", os.path.basename(p), flush=True)
            running.append((p, subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)))
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                if item[1].returncode != 0:
                    print("[gen] FAILED", item[0], flush=True)
        time.sleep(1.0)


def run_queue(jobs: List[Dict], gpus: List[str], per_gpu: int, queue_csv: str, poll: float = 1.0,
              skip_done: bool = True) -> None:
    slots = [g for g in gpus for _ in range(per_gpu)]
    os.makedirs(os.path.dirname(os.path.abspath(queue_csv)), exist_ok=True)
    with open(queue_csv, "w", newline="") as f:
        csv.writer(f).writerow(["name", "device", "rc", "sec", "out_dir"])

    def record(name, device, rc, sec, out_dir):
        with open(queue_csv, "a", newline="") as f:
            csv.writer(f).writerow([name, device, rc, round(sec, 1), out_dir])

    pending = []
    for j in jobs:
        if skip_done and os.path.exists(os.path.join(j["out_dir"], "metrics.json")):
            record(j["name"], "cached", 0, 0.0, j["out_dir"])
        else:
            pending.append(j)
    running: Dict[int, tuple] = {}
    while pending or running:
        for i, dev in enumerate(slots):
            if i not in running and pending:
                job = pending.pop(0)
                os.makedirs(job["out_dir"], exist_ok=True)
                device = "cpu" if dev == "cpu" else "cuda:%s" % dev
                cmd = [c.replace("{device}", device) for c in job["train_cmd"]]
                lf = open(os.path.join(job["out_dir"], "stdout.log"), "w")
                lf.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
                lf.flush()
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
                running[i] = (proc, job, time.time(), lf, device)
                print("[start] %s on %s" % (job["name"], device), flush=True)
        for i in list(running):
            proc, job, t0, lf, device = running[i]
            rc = proc.poll()
            if rc is not None:
                lf.close()
                record(job["name"], device, rc, time.time() - t0, job["out_dir"])
                print("[done] %s rc=%s %.0fs" % (job["name"], rc, time.time() - t0), flush=True)
                del running[i]
        if pending or running:
            time.sleep(poll)


def main(argv=None):
    ap = argparse.ArgumentParser(description="HetNDL sweep launcher")
    ap.add_argument("--spec", required=True, choices=["motiv", "a1", "a1v2", "replication", "diffusive", "multiseed", "a2", "a2n60", "a3", "netsim", "netsim_dopri5", "robust", "noise_sg", "ba", "glv", "grid", "grid_restart"])
    ap.add_argument("--run_name", default=None, help="output folder under runs/ (default: the spec name)")
    ap.add_argument("--root", default=os.path.join(ROOT, "experiments"))
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--per_gpu", type=int, default=3)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--extra", default="", help="extra args passed to hetndl.train, one string")
    ap.add_argument("--gen_workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",") if s]
    jobs = build_jobs(a.spec, a.root, seeds, a.epochs, shlex.split(a.extra), run_name=a.run_name)
    if a.dry_run:
        for j in jobs:
            print(" ".join(j["train_cmd"]))
        print("%d jobs" % len(jobs))
        return
    ensure_datasets(jobs, a.gen_workers)
    run_queue(jobs, a.gpus.split(","), a.per_gpu, os.path.join(a.root, "runs", a.run_name or SHARED_RUN_DIR.get(a.spec, a.spec),
                                                            "queue_%s.csv" % a.spec if a.spec in SHARED_RUN_DIR else "queue.csv"))


if __name__ == "__main__":
    main()
