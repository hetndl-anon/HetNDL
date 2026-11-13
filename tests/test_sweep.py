import csv
import sys

from hetndl.sweep import build_jobs, run_queue


def test_build_jobs_motiv_grid(tmp_path):
    jobs = build_jobs("motiv", root=str(tmp_path), seeds=[1], epochs=10, extra=[])
    assert len(jobs) == 7 * 4
    assert len({j["dataset_path"] for j in jobs}) == 7
    assert {j["model"] for j in jobs} == {"orig", "add", "add_z", "add_z_signed"}
    for j in jobs:
        assert "--model" in j["train_cmd"] and "{device}" in j["train_cmd"]
        assert j["gen_cmd"] is not None and "--no_clip" in j["gen_cmd"]
    assert len(build_jobs("a1", root=str(tmp_path), seeds=[1, 2], epochs=10, extra=[])) == 4
    assert len(build_jobs("replication", root=str(tmp_path), seeds=[1], epochs=10, extra=[])) == 2


def test_run_queue_continues_after_failure(tmp_path):
    jobs = [
        {"name": "bad", "out_dir": str(tmp_path / "bad"),
         "train_cmd": [sys.executable, "-c", "import sys; sys.exit(3)"]},
        {"name": "good", "out_dir": str(tmp_path / "good"),
         "train_cmd": [sys.executable, "-c", "print('ok {device}')"]},
    ]
    q = str(tmp_path / "queue.csv")
    run_queue(jobs, gpus=["cpu"], per_gpu=1, queue_csv=q)
    rows = list(csv.DictReader(open(q)))
    rc = {r["name"]: int(r["rc"]) for r in rows}
    assert rc == {"bad": 3, "good": 0}
    assert "ok cpu" in open(str(tmp_path / "good" / "stdout.log")).read()


def test_build_jobs_motiv_priority_order(tmp_path):
    jobs = build_jobs("motiv", root=str(tmp_path), seeds=[1], epochs=10, extra=[])
    first = {(j["dataset"], j["model"]) for j in jobs[:10]}
    expected = {("kura_ER100_K2_s0_n0", "orig"), ("kura_ER100_K2_s0_n0", "add_z"),
                ("kura_ER100_K2_s0.3_n0", "orig"), ("kura_ER100_K2_s0.3_n0", "add_z"),
                ("kura_ER100_K2_s2_n0", "orig"), ("kura_ER100_K2_s2_n0", "add_z"),
                ("kura_ER100_K2_s0.3_n0.2", "orig"), ("kura_ER100_K2_s0.3_n0.2", "add_z_signed"),
                ("kura_ER100_K2_s0.3_n0.5", "orig"), ("kura_ER100_K2_s0.3_n0.5", "add_z_signed")}
    assert first == expected
    assert all(j["priority"] <= jobs[i + 1]["priority"] for i, j in enumerate(jobs[:-1]))


def test_replication_unit_uses_pair_split(tmp_path):
    jobs = {j["name"].split("/")[-1]: j["train_cmd"] for j in build_jobs("replication", str(tmp_path), [1], 10, [])}
    unit, dt = jobs["orig_unit_seed1"], jobs["orig_dt_seed1"]
    assert unit[unit.index("--split") + 1] == "pairs"
    assert dt[dt.index("--split") + 1] == "traj"


def test_run_name_and_a1v2_spec(tmp_path):
    jobs = build_jobs("motiv", root=str(tmp_path), seeds=[1], epochs=10, extra=["--feat", "trig"],
                      run_name="motiv_trig")
    assert len(jobs) == 28 and all("/runs/motiv_trig/" in j["out_dir"] for j in jobs)
    assert all(j["train_cmd"][-2:] == ["--feat", "trig"] for j in jobs)
    v2 = build_jobs("a1v2", root=str(tmp_path), seeds=[1, 2], epochs=10, extra=[])
    assert len(v2) == 4 and {j["model"] for j in v2} == {"orig", "add_z"}
    g = v2[0]["gen_cmd"]
    assert g[g.index("--T") + 1] == "4" and g[g.index("--samples") + 1] == "40" and g[g.index("--dt") + 1] == "0.02"
    assert all("a1v2_N30_seed" in j["dataset"] for j in v2)


def test_diffusive_spec():
    from hetndl.sweep import build_jobs
    jobs = build_jobs("diffusive", "/tmp/x", [1], 600, ["--feat", "trig"])
    names = [j["name"] for j in jobs]
    assert len(jobs) == 6
    assert all("--coupling" in j["train_cmd"] and "diffusive" in j["train_cmd"] for j in jobs)
    assert all(j["model"] != "orig" for j in jobs)
    assert any(n.startswith("a1v2_N30_seed1/") for n in names)
    assert all("runs/diffusive/" in j["out_dir"] for j in jobs)
    a1 = [j for j in jobs if j["dataset"].startswith("a1v2")][0]
    assert "--feat" not in a1["train_cmd"]  # A1 states are not phases


def test_multiseed_spec():
    from hetndl.sweep import build_jobs
    jobs = build_jobs("multiseed", "/tmp/x", [1, 2], 600, ["--feat", "trig", "--method", "rk4"])
    assert len(jobs) == 2 * (4 * 3 + 3)
    assert all("runs/multiseed/" in j["out_dir"] for j in jobs)
    # every HetNDL model runs with lambda_z = 0; orig is the NOGI baseline
    for j in jobs:
        cmd = j["train_cmd"]
        if j["model"] != "orig":
            assert cmd[cmd.index("--lambda_z") + 1] == "0"
            assert cmd[cmd.index("--coupling") + 1] in ("free", "diffusive")
    # seed k regenerates the data: seed 1 reuses the existing names, seed >= 2 gets its own graph and samples
    ds = {j["dataset"] for j in jobs}
    assert "kura_ER100_K2_s0.3_n0.5" in ds and "kura_ER100_K2_s0.3_n0.5_g2" in ds
    g2 = [j for j in jobs if j["dataset"] == "kura_ER100_K2_s0.3_n0.5_g2"][0]
    gen = g2["gen_cmd"]
    assert gen[gen.index("--seed") + 1] == "2" and gen[gen.index("--graph_seed") + 1] == "2"
    assert g2["train_cmd"][g2["train_cmd"].index("--seed") + 1] == "2"
    # signed models on negative-edge data, unsigned add_z on non-negative data
    assert {j["model"] for j in jobs if j["dataset"].startswith("kura_ER100_K2_s0.3_n0.5")} == {"orig", "add_z_signed"}
    assert {j["model"] for j in jobs if j["dataset"].startswith("kura_ER100_K2_s2_n0")} == {"orig", "add_z"}
    a1 = [j for j in jobs if j["dataset"] == "a1v2_N30_seed2"]
    assert len(a1) == 3 and all("--feat" not in j["train_cmd"] and "--method" not in j["train_cmd"] for j in a1)
    assert all(j["train_cmd"][j["train_cmd"].index("--epochs") + 1] == "1000" for j in a1)
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names)


def test_legacy_specs_pin_lambda_z_1e4():
    # the earliest runs used the old default lambda_z = 1e-4; keep them reproducible
    from hetndl.sweep import build_jobs
    for spec in ("motiv", "a1", "a1v2", "diffusive"):
        for j in build_jobs(spec, "/tmp/x", [1], 600, []):
            cmd = j["train_cmd"]
            if j["model"] != "orig":
                assert "--lambda_z" in cmd and cmd[cmd.index("--lambda_z") + 1] == "1e-4", (spec, j["name"])


def test_a2_spec():
    from hetndl.sweep import build_jobs
    jobs = build_jobs("a2", "/tmp/x", [1, 2], 600, ["--z_norm", "none"])
    assert len(jobs) == 2 * 3 and all("runs/a2/" in j["out_dir"] for j in jobs)
    assert {j["model"] for j in jobs} == {"orig", "add_z"}
    gen = jobs[0]["gen_cmd"]
    assert "hetndl.gen.a2_mixed" in gen and gen[gen.index("--kappa") + 1] == "0.3"
    for j in jobs:
        cmd = j["train_cmd"]
        assert cmd[cmd.index("--epochs") + 1] == "1000" and cmd[cmd.index("--cluster_k") + 1] == "2"
        assert "--feat" not in cmd and cmd[-2:] == ["--z_norm", "none"]
        if j["model"] == "add_z":
            assert cmd[cmd.index("--lambda_z") + 1] == "0" and cmd[cmd.index("--coupling") + 1] in ("free", "diffusive")


def test_a2n60_spec():
    # A2 at N = 60 with p halved, so the mean in-degree (and the coupling regime) matches N = 30
    from hetndl.sweep import build_jobs
    jobs = build_jobs("a2n60", "/tmp/x", [1], 600, [])
    assert len(jobs) == 3 and all("runs/a2/a2_N60_seed1/" in j["out_dir"] for j in jobs)
    gen = jobs[0]["gen_cmd"]
    assert gen[gen.index("--N") + 1] == "60" and gen[gen.index("--p") + 1] == "0.05"
    assert gen[gen.index("--kappa") + 1] == "0.3"
    a2 = build_jobs("a2", "/tmp/x", [1], 600, [])
    assert [j["model"] for j in jobs] == [j["model"] for j in a2]


def test_a3_spec():
    # A3 (source-class-dependent coupling): NOGI, ours without / with endpoint-type coupling (edge_z), plus the
    # free-coupling edge_z control; and edge_z on A2 (homogeneous coupling) as a does-no-harm check
    from hetndl.sweep import build_jobs
    jobs = build_jobs("a3", "/tmp/x", [1], 600, ["--z_norm", "none"])
    names = {j["name"] for j in jobs}
    assert names == {"a3_N30_seed1/orig_seed1", "a3_N30_seed1/add_z_diff_lz0_seed1",
                     "a3_N30_seed1/add_z_diff_ez_lz0_seed1", "a3_N30_seed1/add_z_free_ez_lz0_seed1",
                     "a2_N30_seed1/add_z_diff_ez_lz0_seed1"}
    assert all("runs/a3/" in j["out_dir"] for j in jobs)
    a3 = [j for j in jobs if j["dataset"] == "a3_N30_seed1"][0]
    assert "hetndl.gen.a3_mixed" in a3["gen_cmd"] and a3["gen_cmd"][a3["gen_cmd"].index("--kappa") + 1] == "0.3"
    for j in jobs:
        cmd = j["train_cmd"]
        assert cmd[cmd.index("--epochs") + 1] == "1000" and cmd[-2:] == ["--z_norm", "none"]
        if "_ez_" in j["name"]:
            assert cmd[cmd.index("--edge_z") + 1] == "1"
        elif j["model"] != "orig":
            assert "--edge_z" not in cmd


def test_netsim_spec():
    # real-data benchmark of NOGI: 27 converted sims (no generation step), {NOGI, ours nonneg, ours signed}, d_z 1
    from hetndl.sweep import build_jobs
    jobs = build_jobs("netsim", "/tmp/x", [1], 600, ["--num_workers", "0"])
    assert len(jobs) == 27 * 3 and all(j["gen_cmd"] is None for j in jobs)
    assert {j["model"] for j in jobs} == {"orig", "add_z", "add_z_signed"}
    assert all(j["dataset_path"].endswith("/data/netsim_sim%d.npz" % int(j["dataset"][len("netsim_sim"):])) for j in jobs)
    assert not any(j["dataset"] == "netsim_sim13" for j in jobs)
    j = [j for j in jobs if j["name"] == "netsim_sim4/add_z_signed_diff_lz0_seed1"][0]
    cmd = j["train_cmd"]
    assert cmd[cmd.index("--d_z") + 1] == "2" and cmd[cmd.index("--coupling") + 1] == "diffusive"  # unified D_Z
    assert cmd[cmd.index("--lambda_z") + 1] == "0" and cmd[cmd.index("--epochs") + 1] == "600"
    assert "--feat" not in cmd and cmd[-2:] == ["--num_workers", "0"]
    # fixed-step rk4 (dopri5 took 30-40 s/epoch on the noisy fMRI series) and a cap of ~20k pairs per sim:
    # sims with T = 1200 / 2400 / 5000 use 16 / 8 / 4 of the 50 subjects; T <= 200 keeps all 50
    assert cmd[cmd.index("--method") + 1] == "rk4" and "--n_traj" not in cmd
    cap = {j["dataset"]: j["train_cmd"] for j in jobs if j["model"] == "orig"}
    for sim, n in (("netsim_sim5", "16"), ("netsim_sim19", "8"), ("netsim_sim7", "4")):
        c = cap[sim]; assert c[c.index("--n_traj") + 1] == n, sim
    assert "--n_traj" not in cap["netsim_sim1"] and "--n_traj" not in cap["netsim_sim26"]


def test_robust_spec():
    # data-size and noise curves on Kuramoto p-0.5 (signed diffusive) and A2 N60 (add_z diffusive), NOGI alongside
    from hetndl.sweep import build_jobs
    jobs = build_jobs("robust", "/tmp/x", [1, 2], 600, ["--feat", "trig", "--method", "rk4", "--num_workers", "0"])
    assert len(jobs) == 2 * 2 * (3 + 3) * 2            # seeds x systems x (3 budgets + 3 noise levels) x {orig, ours}
    assert all("runs/robust/" in j["out_dir"] for j in jobs)
    ds = {j["dataset"] for j in jobs}
    assert ds == {"kura_ER100_K2_s0.3_n0.5", "kura_ER100_K2_s0.3_n0.5_g2", "a2_N60_seed1", "a2_N60_seed2"}
    k = {j["name"]: j["train_cmd"] for j in jobs if j["dataset"] == "kura_ER100_K2_s0.3_n0.5"}
    assert "kura_ER100_K2_s0.3_n0.5/add_z_signed_diff_lz0_n5_seed1" in k and "kura_ER100_K2_s0.3_n0.5/orig_noise0.1_seed1" in k
    c = k["kura_ER100_K2_s0.3_n0.5/add_z_signed_diff_lz0_n5_seed1"]
    assert c[c.index("--n_traj") + 1] == "5" and "--noise_std" not in c and "--feat" in c and c[-2:] == ["--num_workers", "0"]
    c = k["kura_ER100_K2_s0.3_n0.5/orig_noise0.1_seed1"]
    assert c[c.index("--noise_std") + 1] == "0.1" and "--n_traj" not in c
    a = {j["name"]: j["train_cmd"] for j in jobs if j["dataset"] == "a2_N60_seed2"}
    c = a["a2_N60_seed2/add_z_diff_lz0_snr20_seed2"]
    assert c[c.index("--noise_snr") + 1] == "20" and "--feat" not in c and c[c.index("--epochs") + 1] == "1000"
    assert "a2_N60_seed2/orig_n10_seed2" in a
    g = [j for j in jobs if j["dataset"] == "kura_ER100_K2_s0.3_n0.5_g2"][0]["gen_cmd"]
    assert g[g.index("--seed") + 1] == "2" and g[g.index("--neg_frac") + 1] == "0.5"


def test_grid_spec():
    # IEEE 118 swing dynamics (real topology): NOGI, ours diffusive, ours free; 2-D state (theta, omega), d_z 2
    from hetndl.sweep import build_jobs
    jobs = build_jobs("grid", "/tmp/x", [1, 2], 600, ["--feat", "trig", "--method", "rk4", "--num_workers", "0"])
    assert len(jobs) == 2 * 3 and all("runs/grid/" in j["out_dir"] for j in jobs)
    assert {j["dataset"] for j in jobs} == {"grid118_seed1", "grid118_seed2"}
    g = jobs[0]["gen_cmd"]
    assert "hetndl.gen.powergrid" in g and g[g.index("--case") + 1].endswith("case118.m") and g[g.index("--kappa") + 1] == "0.3"
    names = {j["name"] for j in jobs}
    assert {"grid118_seed1/orig_seed1", "grid118_seed1/add_z_diff_lz0_seed1", "grid118_seed1/add_z_free_lz0_seed1"} <= names
    for j in jobs:
        cmd = j["train_cmd"]
        assert cmd[cmd.index("--epochs") + 1] == "1000" and cmd[cmd.index("--d_z") + 1] == "2"
        assert "--feat" not in cmd and cmd[-2:] == ["--num_workers", "0"]   # bounded angles: raw features
        # the solver passed by the launcher must survive: dopri5 on the stiff M = 0.2 loads blew up to 400-1700 s/epoch
        # because the spec silently dropped "--method rk4"
        assert cmd[cmd.index("--method") + 1] == "rk4"


def test_noise_sg_spec():
    # Kuramoto p-0.5 phase-noise grid with Savitzky-Golay smoothing of the observed series (window = 5 % of the
    # 1000-step trajectories = 51) applied to EVERY method: the training-level fix for the noise overfitting.
    from hetndl.sweep import build_jobs
    jobs = build_jobs("noise_sg", "/tmp/x", [1, 2, 3], 600, ["--feat", "trig", "--method", "rk4", "--num_workers", "0"])
    assert len(jobs) == 3 * 3 * 2                        # seeds x noise levels x {orig, ours}
    assert all("runs/noise_sg/" in j["out_dir"] for j in jobs)
    assert {j["dataset"] for j in jobs} == {"kura_ER100_K2_s0.3_n0.5", "kura_ER100_K2_s0.3_n0.5_g2", "kura_ER100_K2_s0.3_n0.5_g3"}
    k = {j["name"]: j["train_cmd"] for j in jobs}
    c = k["kura_ER100_K2_s0.3_n0.5_g2/add_z_signed_diff_lz0_sg51_noise0.1_seed2"]
    assert c[c.index("--noise_std") + 1] == "0.1" and c[c.index("--smooth_window") + 1] == "51"
    assert c[c.index("--coupling") + 1] == "diffusive" and c[c.index("--lambda_z") + 1] == "0" and "--feat" in c
    assert "--n_traj" not in c and c[c.index("--epochs") + 1] == "600"
    c = k["kura_ER100_K2_s0.3_n0.5/orig_sg51_noise0.05_seed1"]
    assert c[c.index("--smooth_window") + 1] == "51" and c[c.index("--model") + 1] == "orig"
    assert all(j["train_cmd"][j["train_cmd"].index("--smooth_window") + 1] == "51" for j in jobs)


def test_grid_restart_spec():
    # Two extra training restarts per grid cell (train seed 10*data_seed + r, r = 2, 3) on the SAME datasets, so that
    # every method can be reported as best-of-3 restarts by validation MSE (one seed each of ours and NOGI
    # blew up numerically at epoch 20-30 on the stiff loads). Outputs sit beside the first run: <tag>_seed<s>_r<r>.
    from hetndl.sweep import build_jobs
    jobs = build_jobs("grid_restart", "/tmp/x", [1, 2, 3], 600, ["--method", "rk4", "--num_workers", "0"])
    assert len(jobs) == 3 * 3 * 2 and all("runs/grid/" in j["out_dir"] for j in jobs)
    names = {j["name"] for j in jobs}
    assert {"grid118_seed2/add_z_diff_lz0_seed2_r2", "grid118_seed2/orig_seed2_r3", "grid118_seed1/add_z_free_lz0_seed1_r2"} <= names
    j = {j["name"]: j for j in jobs}["grid118_seed2/add_z_diff_lz0_seed2_r3"]
    cmd = j["train_cmd"]
    assert j["dataset"] == "grid118_seed2" and j["dataset_path"].endswith("grid118_seed2.npz") and j["seed"] == 23
    assert cmd[cmd.index("--seed") + 1] == "23" and cmd[cmd.index("--method") + 1] == "rk4"
    assert cmd[cmd.index("--epochs") + 1] == "1000" and cmd[cmd.index("--coupling") + 1] == "diffusive"
    assert j["out_dir"].endswith("runs/grid/grid118_seed2/add_z_diff_lz0_seed2_r3")
    g = j["gen_cmd"]
    assert g[g.index("--seed") + 1] == "2"                      # the dataset is the original one, not regenerated


def test_netsim_dopri5_spec():
    # dopri5 top-ups for the three sims where the rk4-200-epoch recipe looked conservative (sim4 N=50; sims 7 and 9
    # T=5000 with 4 subjects): same models / caps as spec netsim, adaptive solver, 600 epochs, own run dir.
    from hetndl.sweep import build_jobs
    jobs = build_jobs("netsim_dopri5", "/tmp/x", [1], 600, ["--num_workers", "0"])
    assert len(jobs) == 3 * 3 and {j["dataset"] for j in jobs} == {"netsim_sim4", "netsim_sim7", "netsim_sim9"}
    assert all("runs/netsim_dopri5/" in j["out_dir"] for j in jobs)
    for j in jobs:
        cmd = j["train_cmd"]
        assert cmd[cmd.index("--method") + 1] == "dopri5" and "--step_size" not in cmd
        assert cmd[cmd.index("--epochs") + 1] == "600"
    c = {j["dataset"]: j["train_cmd"] for j in jobs if j["model"] == "orig"}
    assert c["netsim_sim7"][c["netsim_sim7"].index("--n_traj") + 1] == "4" and "--n_traj" not in c["netsim_sim4"]


def test_mirror_job_copies_stored_args_with_overrides(tmp_path):
    # rerun a finished run with identical settings except the overrides (signed default on the benchmarks whose
    # headline runs used the non-negative variant)
    import json, os
    from hetndl.sweep import mirror_job
    run = tmp_path / "runs" / "a2" / "a2_N60_seed1" / "add_z_diff_lz0_snr10_seed1"; run.mkdir(parents=True)
    args = {"data": "/x/a2_N60_seed1.npz", "model": "add_z", "out": str(run), "d_z": 2, "epochs": 1000, "seed": 1,
            "device": "cuda:0", "noise_snr": 10.0, "noise_std": None, "n_traj": None, "coupling": "diffusive",
            "lambda_z": 0.0, "method": "dopri5", "step_size": None, "edge_z": 0}
    json.dump({"args": args}, open(run / "metrics.json", "w"))
    j = mirror_job(str(run), str(tmp_path / "runs" / "signed_default"), model="add_z_signed")
    cmd = j["train_cmd"]
    assert cmd[cmd.index("--model") + 1] == "add_z_signed" and cmd[cmd.index("--data") + 1] == "/x/a2_N60_seed1.npz"
    assert cmd[cmd.index("--device") + 1] == "{device}" and cmd[cmd.index("--noise_snr") + 1] == "10.0"
    assert "--noise_std" not in cmd and "--n_traj" not in cmd and "--step_size" not in cmd     # None = default
    assert cmd[cmd.index("--epochs") + 1] == "1000" and cmd[cmd.index("--method") + 1] == "dopri5"
    assert j["out_dir"] == str(tmp_path / "runs" / "signed_default" / "a2_N60_seed1" / "add_z_signed_diff_lz0_snr10_seed1")
    assert cmd[cmd.index("--out") + 1] == j["out_dir"] and j["seed"] == 1 and j["name"].endswith("add_z_signed_diff_lz0_snr10_seed1")


def test_ba_spec():
    # graph-type robustness: scale-free BA(m = 2) versions of signed Kuramoto p- 0.5 (N 100) and A1 (N 30);
    # NOGI and the signed default, same training settings as their ER counterparts
    from hetndl.sweep import build_jobs
    jobs = build_jobs("ba", "/tmp/x", [1, 2, 3], 600, ["--feat", "trig", "--method", "rk4", "--num_workers", "0"])
    assert len(jobs) == 3 * 2 * 2 and all("runs/ba/" in j["out_dir"] for j in jobs)
    assert {j["dataset"] for j in jobs} == {"kura_BA100_m2_K2_s0.3_n0.5_g%d" % s for s in (1, 2, 3)} | {"a1v2_BA30_m2_seed%d" % s for s in (1, 2, 3)}
    k = {j["name"]: j for j in jobs}
    j = k["kura_BA100_m2_K2_s0.3_n0.5_g2/add_z_signed_diff_lz0_seed2"]; c = j["train_cmd"]; g = j["gen_cmd"]
    assert g[g.index("--graph") + 1] == "BA" and g[g.index("--m") + 1] == "2" and g[g.index("--neg_frac") + 1] == "0.5"
    assert c[c.index("--coupling") + 1] == "diffusive" and c[c.index("--lambda_z") + 1] == "0" and "--feat" in c
    assert c[c.index("--epochs") + 1] == "600" and c[c.index("--method") + 1] == "rk4"
    j = k["a1v2_BA30_m2_seed3/add_z_signed_diff_lz0_seed3"]; c = j["train_cmd"]; g = j["gen_cmd"]
    assert g[g.index("--graph") + 1] == "BA" and "hetndl.gen.a1_mixed" in g and g[g.index("--T") + 1] == "4"
    assert "--feat" not in c and c[c.index("--method") + 1] == "dopri5" and c[c.index("--epochs") + 1] == "1000"
    assert c[c.index("--d_z") + 1] == "2" and k["a1v2_BA30_m2_seed1/orig_seed1"]["train_cmd"].count("--model") == 1


def test_glv_spec():
    # second signed system: GLV, multiplicative (non-diffusive) coupling; NOGI, signed default
    # (diffusive), signed free coupling (ablation: the true coupling does not vanish at x_j = x_i); 3 seeds
    from hetndl.sweep import build_jobs
    jobs = build_jobs("glv", "/tmp/x", [1, 2, 3], 600, ["--feat", "trig", "--method", "rk4", "--num_workers", "0"])
    assert len(jobs) == 3 * 3 and all("runs/glv/" in j["out_dir"] for j in jobs)
    assert {j["dataset"] for j in jobs} == {"glv_N30_n0.5_seed%d" % s for s in (1, 2, 3)}
    k = {j["name"]: j for j in jobs}
    g = k["glv_N30_n0.5_seed2/add_z_signed_diff_lz0_seed2"]["gen_cmd"]
    assert "hetndl.gen.glv" in g and g[g.index("--neg_frac") + 1] == "0.5" and g[g.index("--T") + 1] == "3"
    c = k["glv_N30_n0.5_seed2/add_z_signed_diff_lz0_seed2"]["train_cmd"]
    assert c[c.index("--coupling") + 1] == "diffusive" and c[c.index("--lambda_z") + 1] == "0" and "--feat" not in c
    assert c[c.index("--method") + 1] == "rk4" and c[c.index("--epochs") + 1] == "1000" and c[c.index("--d_z") + 1] == "2"  # unified D_Z
    c = k["glv_N30_n0.5_seed3/add_z_signed_free_lz0_seed3"]["train_cmd"]
    assert c[c.index("--coupling") + 1] == "free" and c[c.index("--model") + 1] == "add_z_signed"
    assert k["glv_N30_n0.5_seed1/orig_seed1"]["train_cmd"].count("--coupling") == 0


def test_current_specs_use_one_unified_d_z(tmp_path):
    # d_z had been set per benchmark from the kind of heterogeneity the generator uses (1 vs 2), i.e.
    # from knowledge of the generating process. Every current (non-legacy) spec must give every z model the same d_z.
    from hetndl.sweep import build_jobs, D_Z, LEGACY_LAMBDA_Z_SPECS
    specs = ["multiseed", "a2", "a2n60", "a3", "netsim", "robust", "noise_sg", "ba", "glv", "grid", "grid_restart",
             "netsim_dopri5"]
    seen = set()
    for spec in specs:
        assert spec not in LEGACY_LAMBDA_Z_SPECS
        jobs = build_jobs(spec, str(tmp_path / spec), [1], 10, [])
        for j in jobs:
            c = j["train_cmd"]
            if c[c.index("--model") + 1] in ("add_z", "add_z_signed"):
                seen.add(c[c.index("--d_z") + 1])
    assert seen == {str(D_Z)} and D_Z == 2
