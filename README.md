# HetNDL: signed interaction graphs and heterogeneous dynamics from networked time series

Code for the paper *Learning Signed Interaction Graphs and Heterogeneous Dynamics from Networked Time Series*
(anonymous submission).

* **Supplementary material:** [supplement.pdf](supplement.pdf) (the sections, tables and figures numbered S1, S2, …
  in the paper)
* **Code:** [github.com/hetndl-anon/HetNDL](https://github.com/hetndl-anon/HetNDL) (browse) or
  [main.zip](https://github.com/hetndl-anon/HetNDL/archive/refs/heads/main.zip) (download)

HetNDL is a graph neural ODE that learns, from trajectories alone,

* a **signed interaction graph** (edge weights may be negative, i.e. inhibitory),
* a **node latent** $z_i$ per node, so that nodes may follow different self-dynamics, from which node parameters and
  mechanism classes are read out without ground truth,
* a **coupling function** in diffusive form $G(x_i, x_j) - G(x_i, x_i)$, which fixes the split between a node's
  self-field and its incoming coupling,
* optionally **edge embeddings** for coupling laws that depend on the source node.

Held-out trajectories choose the model snapshot and between a non-negative and a signed candidate (and, with the
extension, between a shared and an edge-conditioned coupling).

## Repository layout

| Path | Content |
|---|---|
| `hetndl/model.py` | the model: node latents, additive self-field, diffusive coupling, signed / non-negative weights, edge embeddings, sign gauge; the NOGI-style shared-function model with non-negative or signed weights |
| `hetndl/train.py` | training (one-step targets, L1 on $W$, validation-MSE snapshot); one call trains one candidate |
| `hetndl/readout.py`, `hetndl/edge_readout.py` | truth-free read-outs: node constants, mechanism classes (number of classes by BIC), edge types |
| `hetndl/gen/` | benchmark generators: Kuramoto, MixRelax (`a1_mixed`), MixOsc (`a2_mixed`), MixOsc-E (`a3_mixed`, also with bounded coupling laws and inhibitory edges), GLV, Grid-118 (`powergrid`), NetSim converter; for the supplement the spring network of AIDD (`springs`) and coarser sampling intervals of an existing dataset (`subsample`) |
| `hetndl/baselines/` | our re-implementations of ARNI, lasso Granger causality, NRI, AIDD and SGODE (the public SGODE code covers only its traffic model, so its continuous-dynamics model is re-implemented from the paper; reading its matrix as an interaction graph is our addition) |
| `experiments/launch/` | runners for MTGNN, COSINE and the original NRI code (their code is obtained separately, see below) and the runtime benchmark |
| `experiments/external/nri/patches.diff` | our changes to the original NRI code (see below) |
| `experiments/analysis/` | the scripts that compute every number of the paper and of the supplement from finished runs |
| `experiments/figures/` | the experiment figures (Fig. 2, Fig. 3, and Fig. S1 of the supplement) |
| `experiments/reproduce/` | dataset and job lists, a job runner, and `paper.sh`, which produces all tables and figures |
| `tests/` | unit tests (`pytest`) |
| `supplement.pdf` | the supplementary material of the paper |

Internal names: `a1`/`a1v2` = MixRelax (`a1v2s` = MixRelax with inhibitory edges, supplement only), `a2` = MixOsc,
`a3` = MixOsc-E, `kura_ER100_K2_s<σ>_n<p⁻>` = Kuramoto (`n0` = Kuramoto-ω, `n0.2`/`n0.5` = Kuramoto-±), `glv` = GLV-±,
`grid118` = Grid-118. Supplement only: `a2t`/`a2ts` = MixOsc nodes with the saturating coupling on every edge, without and
with inhibitory edges; `a3u`/`a3s` = MixOsc-E nodes with two bounded coupling laws, without and with inhibitory edges;
`kura5_nri` = the five-oscillator Kuramoto system of the NRI paper with one fixed graph, `spring10` = the spring network
of AIDD, `glv_BA30_m2` = GLV-± on Barabási–Albert graphs, `<dataset>_sub2`/`_sub4` = every 2nd / 4th stored state of a
dataset, `a1v2_N30_cs10` = MixRelax with node coefficients that vary by up to 10% within each law
(`hetndl.gen.a1_mixed --coef_spread 0.1`).
Model variants of `hetndl.train --model`: `add_z` = HetNDL with $W\ge 0$ (candidate $\mathcal{C}_+$), `add_z_signed` = HetNDL with signed
$W$ ($\mathcal{C}_\pm$), `orig` = the NOGI-style model (one shared update function, non-negative weights),
`orig_signed` = the same model with signed weights ("signed NOGI", supplement); `--edge_emb 2`
adds the edge embeddings of the extension. Three switches serve the checks of the supplement and are off by default:
`--init_seed` (another initialisation on the same data split), `--edge_scale epoch` (edge embeddings with a
normalisation that is constant within an epoch) and `--gauge_scale ref` (a fixed scale of the shared coupling).
`--cluster_k` only adds a logged reference metric that uses the true
number of classes; it does not enter training, model selection or any reported read-out
(`experiments/analysis/audit_settings.py` checks this).

## Installation

Python 3.8 with the versions we used:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # 202 tests, about 40 s on a CPU (3 of them are skipped until the datasets exist)
```

The symbolic distillation (Table 4) uses [PySR](https://github.com/MilesCranmer/PySR) 2.7.2 in a separate environment
(Python 3.11, `pip install pysr==2.7.2 numpy sympy`); `paper.sh` calls it through `SR_PYTHON`.

## Quick start: one benchmark

MixRelax, data seed 1, with the paper's settings (each candidate takes about 30 min on one GPU):

```bash
export PYTHONPATH=.
python -m hetndl.gen.a1_mixed --N 30 --p 0.1 --directed --samples 40 --T 4 --dt 0.02 --seed 1 --kappa 0.5 \
    --out experiments/data/a1v2_N30_seed1.npz
# both candidates of the default procedure (non-negative and signed W)
python experiments/reproduce/run_jobs.py --name '^(a1_lz0|signed_default)/a1v2_N30_seed1/' --gpus 0 --per_gpu 2
# keep the candidate with the lower validation MSE, then read out the mechanism classes
RUN=$(python -c "import json; c=['experiments/runs/a1_lz0/a1v2_N30_seed1/add_z_diff_lz0_seed1', \
  'experiments/runs/signed_default/a1v2_N30_seed1/add_z_signed_diff_lz0_seed1']; \
  print(min(c, key=lambda r: json.load(open(r + '/metrics.json'))['val_mse_best']))")
python experiments/analysis/a1_fcurves.py $RUN
```

`metrics.json` of a run holds the graph scores (AUROC / AUPRC against the true graph, computed after training and used
only for scoring) and the validation error; `W_by_mse.npy` holds the learned weights of the reported snapshot.

## Reproducing the paper

### 1. Data

```bash
python experiments/reproduce/make_datasets.py --workers 8
```

generates the 87 synthetic datasets of `experiments/reproduce/datasets.json` (we regenerated all of them with this
script: every array of the files we used is reproduced exactly; the docstring of `make_datasets.py` names the stored
extras in which the regenerated files differ). Two benchmarks need a download first:

* **Grid-118**: the MATPOWER IEEE 118-bus case file
  ([case118.m](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case118.m), sha256 `bc2e6f22…14c4b5f7`)
  at `experiments/real/matpower_raw/case118.m`. It is parsed as text, never executed.
* **NetSim** (Smith et al., 2011): [sims.tar.gz](https://www.fmrib.ox.ac.uk/datasets/netsim/sims.tar.gz)
  (sha256 `fec50406…a7981a21`) unpacked into `experiments/real/netsim_raw/` (`sim1.mat` … `sim28.mat`; sim13 is
  skipped, as in NOGI).

### 2. Training runs

`experiments/reproduce/jobs.json` lists the 1902 runs whose results the paper and its supplement report: HetNDL (both
candidates of every cell, robustness, scale, ablations, extension, and the restarts and variants of the supplement), the
NOGI-style model with non-negative and with signed weights, ARNI, Granger, NRI, AIDD, MTGNN, COSINE, SGODE and the
original NRI code. Every
command was rebuilt from the arguments the original run recorded and checked by parsing it back with the entry
point's own argument parser. Each job lists the analysis scripts that read it (`used_by`), so parts of the paper can
be reproduced alone:

```bash
python experiments/reproduce/run_jobs.py --gpus 0,1 --per_gpu 3 --cpu 8                       # everything
python experiments/reproduce/run_jobs.py --gpus 0 --used_by baseline_table,sel_tables          # Table 2 only
python experiments/reproduce/run_jobs.py --dry_run --used_by scale_table                        # list commands
python experiments/reproduce/run_jobs.py --gpus 0 --used_by snogi_table,combined_table          # two tables of the supplement
python experiments/reproduce/run_jobs.py --gpus 0 --cpu 4 --used_by spring_check,positive_control,kuramoto5_recovery
```

Finished jobs are skipped, so the runner can be restarted. Compute: on one otherwise idle RTX 3090, a HetNDL candidate
takes 0.5–1 h at $N\le 100$ (5 h at $N=400$ on an A100); our logs add up to about 3,200 GPU-hours for the 1,064 HetNDL
and NOGI-style runs on shared machines (870 h of them for the 341 runs added for the supplement), 190 h for COSINE, 21 h
for MTGNN and 217 h for SGODE. The checks of the supplement on sampling
intervals, Barabási–Albert graphs and the Grid-118 training budget took about 30 GPU-hours for 30 HetNDL runs and
about 40 for the extra Grid-118 trainings of COSINE, SGODE and MTGNN; the six HetNDL runs on the five-oscillator
Kuramoto data ran on a CPU (about 5 h each). GPU training is not bit-wise
deterministic, so retrained numbers can differ slightly from ours.

### 3. Tables and figures

```bash
PYTHON=python SR_PYTHON=/path/to/pysr-env/bin/python BENCH_GPU=cuda:0 bash experiments/reproduce/paper.sh
```

writes the tables to `results/` and the figures to `experiments/figures/out/`. It first applies the default procedure
(the candidate with the lower validation MSE per cell, `runs/sel/selected.json`) and runs the settings and leakage
audit. `SR_PYTHON` (PySR) and `BENCH_GPU` (cost benchmarks on an idle GPU) are optional; the steps that need them are
skipped otherwise. The self-field searches are deterministic (serial PySR with a fixed seed; `PERNODE_JOBS` sets how
many runs go in parallel, about one hour on 18 cores); the coupling-law searches are multithreaded, so those expressions
can differ in form from ours.

| Paper | Script(s) in `experiments/analysis/` | Output in `results/` |
|---|---|---|
| Table 2, edge signs, NetSim | `baseline_table.py`, `mtgnn_forecast_check.py` | `table2_topology.txt`, `mtgnn_forecast_check.txt` |
| Table 3 (node read-outs) | `omega_readout.py`, `granger_omega.py`, `node_table.py` (the class rows as printed), `a1_fcurves.py`, `a2_fields.py`, `grid_readout.py`, `granger_mechanism.py`, `arni_mechanism.py` | `table3_*.txt` |
| Table 4 (equations) | `distill_export.py`, `pernode_export.py` and `pernode_distill.py` (self-fields: one form per read-out class, constants per node), `distill_pysr.py` (coupling laws), `skeleton_match.py`, `distill_diag.py` (coupling laws in diffusive form), `distill_operators.py` (extended operator library) | `pernode_distill.txt`, `distill_pysr*.txt`, `table4_skeletons.txt`, `supplement/skeleton_match_full.txt`, `supplement/distill_diag.txt`, `supplement/distill_operators.txt` |
| Fig. 3, robustness, roll-outs | `noise_table.py`, `noise_mech_readout.py`, `ntraj_table.py`, `scale_table.py`, `scale_penalty_table.py`, `rollout_baselines.py`, `grid_rollout_diag.py` | `robust_*.txt`, `scale.txt`, `supplement/scale_penalty.txt`, `rollout*.txt` |
| Table 5 (ablations, candidates) | `ablation_table.py`, `sign_member_table.py` | `table5_*.txt` |
| Source-dependent coupling | `edge_b_compare.py` | `table6_edge_embeddings.txt` |
| Runtime | `scale_bench.py`, `../launch/timing_bench.py`, `timing_table.py` | `scale.txt`, `timing.txt` |
| Fig. 2, Fig. 3 | `experiments/figures/fig_laws.py`, `fig_robust.py` | `experiments/figures/out/` |

The last block of `paper.sh` reproduces the supplementary material (`supplement.pdf`). Its text outputs go to
`results/supplement/`; `supp_tables.py` then writes the LaTeX tables of the supplement from these files to
`results/supp_tables/` (it only selects and reformats, so no number is typed by hand), and
`experiments/figures/fig_supp_topology.py` draws Fig. S1. Table S12 and the LaTeX tables need the distilled equations,
hence `SR_PYTHON`; `netsim_protocol.py` needs the downloaded NetSim files; `nri_regime.py`, `nri_official.py` and the
rows of the original NRI code in `spring_check.py` and `kuramoto5_recovery.py` need its runs (external code, see below).

| Supplement | Script(s) in `experiments/analysis/` | Output in `results/supplement/` |
|---|---|---|
| S2, setup details: NetSim protocol, edge-conditioned coupling with a normalisation fixed within an epoch | `netsim_protocol.py`, `edge_scale_table.py` | `netsim_protocol.txt`, `edge_scale.txt` |
| S2, baselines: NRI and AIDD on the spring network and on the five-oscillator Kuramoto system, the original NRI code, the best of three trainings on Grid-118 | `spring_check.py`, `positive_control.py`, `kuramoto5_recovery.py`, `nri_regime.py`, `nri_official.py`, `grid_budget.py` | `spring_check.txt`, `positive_control.txt`, `kuramoto5_recovery.txt`, `nri_regime.txt`, `nri_official.txt`, `grid_budget.txt` |
| S3, binary graphs (Table S2, Fig. S1) | `binary_graph_table.py`, `weight_spread.py`, `../figures/fig_supp_topology.py` | `binary_graph_table.txt`, `weight_spread.txt`, `fig_supp_topology_counts.txt` |
| S4, per-seed results, worst case and mean rank, NetSim, model selection (Tables S3 to S6) | `baseline_table.py [--per-seed]`, `worst_case_rank.py`, `netsim_undirected.py`, `selection_table.py` | `baseline_table_per_seed.txt`, `baseline_table.txt`, `worst_case_rank.txt`, `netsim_undirected.txt`, `selection_table.txt` |
| S5, SGODE baseline (Table S7) | `sgode_table.py`, `mtgnn_forecast_check.py` | `sgode.txt`, `mtgnn_forecast_check.txt` |
| S6, signed NOGI (Table S8) | `snogi_table.py` | `snogi.txt` |
| S7, combined heterogeneity (Table S9) | `supp_readout.py x1`, `combined_table.py` | `mixrelax_pm.txt`, `combined.txt` |
| S8, training restarts (Table S10), HetNDL on Grid-118 with its first training run only | `restart_table.py`, `grid_budget.py` | `restart.txt`, `grid_budget.txt` |
| S9, node-level read-outs (Table S11) | `node_constant_accuracy.py`, `readout_split_check.py`, `arni_same_readout.py`, `readout_hparams.py`, `glv_selffield.py` | `node_constant_accuracy.txt`, `readout_split_check.txt`, `arni_same_readout.txt`, `readout_hparams.txt`, `glv_selffield.txt` |
| S10, distilled equations (Table S12), extended operator library | `pernode_distill.py`, `skeleton_match.py --full`, `distill_diag.py`, `distill_operators.py` | `skeleton_match_full.txt`, `distill_diag.txt`, `distill_operators.txt` |
| S11, robustness details: node constants, sampling interval, roll-out on Grid-118, GLV-± on BA graphs, scale | `glv_node_constant.py`, `subsample_table.py`, `grid_bus_error.py`, `glv_ba.py`, `scale_penalty_table.py` | `glv_node_constant.txt`, `subsample.txt`, `grid_bus_error.txt`, `glv_ba.txt`, `scale_penalty.txt` |
| S12, sparsity penalty (Table S13), scale of the shared coupling | `supp_readout.py x2`, `supp_readout.py x5`, `coupling_scale_table.py` | `l1_zero.txt`, `l1_sweep.txt`, `coupling_scale.txt` |

## External baselines

Their code is not redistributed here.

* **MTGNN** (Wu et al., KDD 2020; MIT license): the official repository
  [nnzhan/MTGNN](https://github.com/nnzhan/MTGNN) at commit `f811746` (master zip sha256 `6d2fa2e1…f13423c`), unmodified,
  with its Python files in `experiments/external/mtgnn/src/`. `experiments/launch/mtgnn_run.py` documents the protocol.
* **COSINE**: the anonymous repository COSINE-6D43 (zip sha256 `40e98da0…c78ff690`), unmodified, in
  `experiments/external/cosine/src/`, run through `experiments/external/cosine/cosine_entry.py`, which only restricts
  the attributes that language-model-generated basis expressions may use. COSINE needs its own environment (set
  `COSINE_PYTHON` to its interpreter) and an OpenAI-compatible language-model server; we served `qwen3:32b` at
  `http://localhost:11434/v1/`. Its early stopping on the true graph is disabled (see `experiments/launch/cosine_run.py`).
* **NRI, original code** (Kipf et al., ICML 2018; MIT license), for the checks of the supplement (S2): the official
  repository [ethanfetaya/NRI](https://github.com/ethanfetaya/NRI) at commit `e63fcb0`
  (`e63fcb0144bca60eb1cffed9a94489de928d6c23`). `experiments/external/nri/patches.diff` changes only what current
  PyTorch and NumPy require, adds a loader for our input files, lets the test step score windows shorter than two
  training windows (edge posteriors only) and saves the test posteriors:

  ```bash
  git clone https://github.com/ethanfetaya/NRI experiments/external/nri/src
  git -C experiments/external/nri/src checkout e63fcb0144bca60eb1cffed9a94489de928d6c23
  mkdir -p experiments/external/nri/patched
  git -C experiments/external/nri/src archive HEAD | tar -x -C experiments/external/nri/patched
  patch -p1 -d experiments/external/nri/patched < experiments/external/nri/patches.diff
  ```

  It runs in the main environment. `experiments/launch/nri_original_run.py` writes the input files into
  `experiments/external/nri/patched/data/`, starts its `train.py` and documents the protocol: NRI's own Kuramoto regime
  (a new graph for every one of 70,000 simulations), the five-oscillator system with one fixed graph, our benchmarks with
  $N\le60$ and NetSim, and the spring network.

## License

MIT (see `LICENSE`). The downloaded data (MATPOWER, NetSim) and the external baselines keep their own licenses.
