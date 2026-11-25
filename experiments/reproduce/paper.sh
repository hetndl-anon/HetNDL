#!/usr/bin/env bash
# Every number and experiment figure of the paper from the finished runs (experiments/reproduce/run_jobs.py).
# Outputs go to results/*.txt (tables) and experiments/figures/out/ (figures); the numbers, tables and figure of the
# supplementary material (supplement.pdf) go to results/supplement/*.txt, results/supp_tables/*.tex and
# experiments/figures/out/fig_supp_topology.*.
#   PYTHON     interpreter of the main environment (default: python)
#   SR_PYTHON  interpreter of the PySR environment, for the distilled equations (optional; skipped if unset)
#   BENCH_GPU  CUDA device for the cost benchmarks, e.g. cuda:0 (optional; skipped if unset; needs an idle GPU)
#   PERNODE_JOBS  parallel processes for the per-node distillation of the self-fields (default 4; one CPU core each)
# Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
PY=${PYTHON:-python}
A=experiments/analysis
mkdir -p results experiments/runs/sel
sel() { $PY experiments/reproduce/selected_runs.py "$1"; }
step() { echo "== $1"; }

step "default procedure: lower validation MSE of the two candidates per cell -> runs/sel/selected.json"
cp experiments/reproduce/sel_manifest.json experiments/runs/sel/manifest.json
$PY $A/sel_tables.py > results/selection.txt
$PY $A/audit_settings.py > results/audit.txt

step "Table 2, edge signs, NetSim (Sec. 5.2)"
$PY $A/baseline_table.py > results/table2_topology.txt
$PY $A/mtgnn_forecast_check.py > results/mtgnn_forecast_check.txt
$PY $A/sel_vs_nogi.py > results/hetndl_vs_nogi.txt

step "Table 3, node-level read-outs (Sec. 5.3)"
$PY $A/omega_readout.py > results/table3_node_constants.txt
$PY $A/granger_omega.py > results/table3_granger_omega.txt
$PY $A/a1_fcurves.py $(sel 'a1v2_N30_seed[123]$') > results/table3_mixrelax_classes.txt
$PY $A/a2_fields.py $(sel 'a2_N(30|60)_seed[123]$') > results/table3_mixosc_classes.txt
$PY $A/grid_readout.py $(sel 'grid118_seed[123]$') > results/table3_grid_classes.txt
$PY $A/granger_mechanism.py $(for d in a1v2_N30 a2_N30 a2_N60 grid118; do for s in 1 2 3; do
  echo experiments/runs/granger/${d}_seed$s/seed$s; done; done) > results/table3_granger_classes.txt
$PY $A/arni_mechanism.py > results/table3_arni.txt
# the class rows of Table 3 as printed in the paper (ARNI: its fitted functions through the read-out of HetNDL)
$PY $A/node_table.py > results/table3_classes.txt

step "Table 4, distilled equations (Sec. 5.3): export the learned functions on grids, then PySR"
$PY $A/distill_export.py $(sel 'a1v2_N30_seed[123]$|a2_N(30|60)_seed[123]$|grid118_seed[123]$|kura_ER100_K2_s0.3_n0.5(_g[23])?$') \
    --out experiments/distill_sel
$PY $A/distill_export.py $(sel 'glv_N30_n0.5_seed[123]$') --out experiments/distill_glv_1106
if [ -n "${SR_PYTHON:-}" ]; then
  $SR_PYTHON $A/distill_pysr.py experiments/distill_sel/*/ > results/distill_pysr.txt
  $SR_PYTHON $A/distill_pysr.py experiments/distill_glv_1106/*/ > results/distill_pysr_glv.txt
  $PY $A/skeleton_match.py > results/table4_skeletons.txt
  # self-field equations (Table 4, top rows; Sect. S10): one form per read-out class, constants per node, the same
  # procedure on every benchmark; deterministic PySR searches, one process per run
  $PY $A/pernode_export.py
  ls experiments/distill_pernode | xargs -P "${PERNODE_JOBS:-4}" -I{} \
      sh -c "$SR_PYTHON $A/pernode_distill.py '{}' > experiments/distill_pernode/{}/log.txt 2>&1"
  $SR_PYTHON $A/pernode_distill.py --summary > results/pernode_distill.txt
else
  echo "SR_PYTHON not set: skipping the symbolic distillation"
fi

step "Fig. 3 and robustness (Sec. 5.4)"
mkdir -p experiments/figures/data
$PY $A/noise_table.py --json experiments/figures/data/noise_table.json > results/robust_noise.txt
$PY $A/noise_mech_readout.py > results/robust_noise_mechanisms.txt
$PY $A/ntraj_table.py > results/robust_trajectories.txt
$PY $A/rollout_baselines.py > results/rollout.txt
$PY $A/grid_rollout_diag.py > results/rollout_grid_diagnosis.txt

step "Table 5, ablations (Sec. 5.5)"
$PY $A/ablation_table.py > results/table5_ablations.txt
$PY $A/sign_member_table.py > results/table5_sign_candidates.txt

step "Sec. 5.6, source-dependent coupling (edge embeddings, four candidates)"
$PY $A/edge_b_compare.py > results/table6_edge_embeddings.txt

step "cost (Sec. 5.7) and scale (Sec. 5.4)"
if [ -n "${BENCH_GPU:-}" ]; then
  $PY $A/scale_bench.py "$BENCH_GPU"
  $PY experiments/launch/timing_bench.py "$BENCH_GPU"
fi
$PY $A/scale_table.py --json experiments/figures/data/scale_table.json > results/scale.txt
# the scale study with the penalty scaled by the number of node pairs (dashed line of Fig. 3b; supplement S11)
mkdir -p results/supplement
$PY $A/scale_penalty_table.py > results/supplement/scale_penalty.txt
[ -f experiments/runs/timing/timing.json ] && $PY $A/timing_table.py > results/timing.txt || true

step "figures"
mkdir -p experiments/figures/out
if [ -f experiments/runs/scale/bench.json ]; then
  $PY experiments/figures/fig_robust.py
else
  echo "no runs/scale/bench.json (set BENCH_GPU): skipping Fig. 3, whose panel (c) needs the cost benchmark"
fi
$PY experiments/figures/fig_laws.py

step "supplementary material (supplement.pdf): text outputs, then its LaTeX tables and Fig. S1"
S=results/supplement
mkdir -p $S
# S2 setup details: the NetSim protocol (needs the raw NetSim files) and the edge-conditioned coupling retrained with a
# normalisation that is constant within an epoch
if [ -d experiments/real/netsim_raw ]; then
  $PY $A/netsim_protocol.py > $S/netsim_protocol.txt
else
  echo "no experiments/real/netsim_raw: skipping netsim_protocol.py"
fi
$PY $A/edge_scale_table.py > $S/edge_scale.txt
# S2 baselines: NRI and AIDD in AIDD's setting (spring network) and on the five-oscillator Kuramoto of the NRI paper with
# one fixed graph, where HetNDL, ARNI and Granger run as well; the original NRI code in its own regime, on that system and
# on our benchmarks (its runs need the external code, see README.md); every neural baseline reports the best of three
# trainings on Grid-118
$PY $A/spring_check.py > $S/spring_check.txt
$PY $A/positive_control.py > $S/positive_control.txt
$PY $A/kuramoto5_recovery.py > $S/kuramoto5_recovery.txt
$PY $A/nri_regime.py > $S/nri_regime.txt
$PY $A/nri_official.py > $S/nri_official.txt
$PY $A/grid_budget.py > $S/grid_budget.txt
# S3 binary graphs (Table S2)
$PY $A/binary_graph_table.py > $S/binary_graph_table.txt
$PY $A/weight_spread.py > $S/weight_spread.txt
$PY $A/weight_coverage.py > $S/weight_coverage.txt
# signs of the true edges that the cut keeps, all twelve settings (Sect. 4.6)
$PY $A/kept_edge_signs.py > $S/kept_edge_signs.txt
# S4 per-seed results, worst case and mean rank, NetSim and model selection (Tables S3 to S6); baseline_table.txt is
# Table 2 again, where supp_tables.py reads it
$PY $A/baseline_table.py > $S/baseline_table.txt
$PY $A/baseline_table.py --per-seed > $S/baseline_table_per_seed.txt
$PY $A/seed_sd.py > $S/seed_sd.txt
$PY $A/worst_case_rank.py > $S/worst_case_rank.txt
$PY $A/netsim_undirected.py > $S/netsim_undirected.txt
$PY $A/selection_table.py > $S/selection_table.txt
# S5 SGODE baseline (Table S7) and the forecasting check of MTGNN
$PY $A/sgode_table.py > $S/sgode.txt
$PY $A/mtgnn_forecast_check.py > $S/mtgnn_forecast_check.txt
# S6 signed NOGI (Table S8)
$PY $A/snogi_table.py > $S/snogi.txt
# S7 combined heterogeneity (Table S9): MixRelax with inhibitory edges; MixOsc and MixOsc-E with bounded coupling laws
$PY $A/supp_readout.py x1 > $S/mixrelax_pm.txt
$PY $A/combined_table.py > $S/combined.txt
# S8 training restarts (Table S10)
$PY $A/restart_table.py > $S/restart.txt
# S9 node-level read-outs (Table S11)
$PY $A/node_constant_accuracy.py > $S/node_constant_accuracy.txt
$PY $A/readout_split_check.py > $S/readout_split_check.txt
$PY $A/arni_same_readout.py > $S/arni_same_readout.txt
$PY $A/readout_hparams.py > $S/readout_hparams.txt
$PY $A/class_contingency.py > $S/class_contingency.txt
$PY $A/glv_selffield.py > $S/glv_selffield.txt
# free coupling with the split fixed after training; MixRelax with coefficients that vary within each law
$PY $A/free_coupling_split.py > $S/free_coupling_split.txt
$PY $A/within_law_spread.py > $S/within_law_spread.txt
# S11 robustness details (scale_penalty.txt was written with the scale table above): node constants, sampling interval,
# roll-out on Grid-118, GLV-± on Barabasi-Albert graphs
$PY $A/glv_node_constant.py > $S/glv_node_constant.txt
$PY $A/subsample_table.py > $S/subsample.txt
$PY $A/grid_bus_error.py > $S/grid_bus_error.txt
$PY $A/glv_ba.py > $S/glv_ba.txt
# S12 sparsity penalty (Table S13) and the fixed scale of the shared coupling
$PY $A/supp_readout.py x2 > $S/l1_zero.txt
$PY $A/supp_readout.py x5 > $S/l1_sweep.txt
$PY $A/coupling_scale_table.py > $S/coupling_scale.txt
# S14 Allier et al. (2026) on the benchmarks (Tables S15, S16; their runs need the external code, see README.md)
if [ -d experiments/runs/allier/PDE_N5_supp8_fig2lr_ep20 ]; then
  $PY $A/allier_control_table.py > $S/allier_control.txt
  $PY $A/allier_table.py --model PDE_N5_supp8_fig2lr_ep20 > $S/allier_rerun.txt
  $PY $A/allier_checks.py --root PDE_N5_supp8_fig2lr_ep20 --root2 PDE_N5_supp8_fig2lr_ep10 > $S/allier_checks.txt
  $PY $A/allier_latents.py --root PDE_N5_supp8_fig2lr_ep20 > $S/allier_latents.txt
  [ -d experiments/runs/allier_control_hetndl ] && $PY $A/allier_control_hetndl.py > $S/allier_control_hetndl.txt
  $PY $A/allier_supp_tables.py
else
  echo "no runs of Allier et al. (2026): skipping S14"
fi
# S10 distilled equations (Table S12; the coupling rows of Table 4 use the diagonal-corrected laws of distill_diag.py)
# and the LaTeX tables; both need the PySR output of the Table 4 step
if ls experiments/distill_sel/*/pysr.json experiments/distill_glv_1106/*/pysr.json experiments/distill_pernode/*/equations.json > /dev/null 2>&1; then
  $PY $A/skeleton_match.py --full > $S/skeleton_match_full.txt
  $PY $A/distill_diag.py > $S/distill_diag.txt
  # the same 18 coupling laws with exp, abs and 1/x added to the operators (Sect. 5.3 and S10)
  if [ -n "${SR_PYTHON:-}" ]; then
    $SR_PYTHON $A/distill_operators.py fit 0 1
  fi
  [ -d experiments/distill_operators ] && $PY $A/distill_operators.py score > $S/distill_operators.txt || true
  $PY $A/supp_tables.py $S results/supp_tables
else
  echo "no distilled equations (set SR_PYTHON): skipping skeleton_match.py --full, distill_diag.py and supp_tables.py"
fi
# Fig. S1 and the counts quoted with it
$PY experiments/figures/fig_supp_topology.py
$PY experiments/figures/fig_supp_topology.py --counts > $S/fig_supp_topology_counts.txt
echo "done: results/ and experiments/figures/out/"
