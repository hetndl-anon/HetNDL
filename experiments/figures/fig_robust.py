"""Fig. 3 (paper): robustness, scaling and dynamics fidelity. Every number comes from a protocol script:
  (a) Kuramoto-± p- 0.5, phase noise: experiments/analysis/noise_table.py --json (HetNDL and NOGI with the same SG
      smoothing; ARNI with its window tuned per seed against the true graph = the protocol exception in its favour);
      the noise-free point = main-table runs (scale_table.py N = 100 for HetNDL / ARNI, sel_vs_nogi.py paths for NOGI).
  (b) Kuramoto-± p- 0.5 at N = 100, 200, 400: scale_table.py --json (HetNDL default procedure, ARNI best of its two
      variants by AUROC as in the main table, Granger). Dashed: HetNDL with lambda_1 scaled by the number of ordered
      node pairs, 1e-8 * 9900 / (N (N - 1)) (results/supplement/scale_penalty.txt from
      experiments/analysis/scale_penalty_table.py); at N = 100 this is the default, so the first point is shared.
  (c) seconds per training epoch vs N (one idle A100, training iterations only): scale_table.py --json (bench.json).
  (d) 100-step roll-out NRMSE on the test trajectories, every main-table setting: rollout_baselines.py.
Run from the project root with PYTHONPATH=. (hetndl env)."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from paperstyle import plt, W_IN, METHOD, MARKER, INK, INK2, INK3, GRID, panel_title, ygrid
from hetndl.snapshot import metrics_of
from experiments.analysis.paper_seeds import PAPER_SEEDS

DATA = "experiments/figures/data"
OUT = "experiments/figures/out/fig_robust"
R = "experiments/runs"
TOP = 1000.0                                                            # row for diverged runs
SETTINGS = [("Kuramoto σ/K 0.15", r"K-$\omega$ 0.15"), ("Kuramoto σ/K 1.0", r"K-$\omega$ 1.0"), ("Kuramoto p⁻ 0.2", r"K-$\pm$ 0.2"),
            ("Kuramoto p⁻ 0.5", r"K-$\pm$ 0.5"), ("BA Kuramoto p⁻ 0.5", r"K-$\pm$ BA"), ("A1 (N30)", "MixRelax"),
            ("BA A1", "MixRelax BA"), ("A2 N30", "MixOsc 30"), ("A2 N60", "MixOsc 60"), ("A3 (edge types)", "MixOsc-E"),
            ("GLV (signed)", r"GLV-$\pm$"), ("IEEE-118 grid", "Grid-118")]


def line(ax, x, per_seed, name, label=None, ms=3.2):
    """per_seed: list over x of lists of values. Mean line + faint per-seed points."""
    mean = [np.mean(v) if len(v) else np.nan for v in per_seed]
    for xi, v in zip(x, per_seed):
        ax.plot([xi] * len(v), v, MARKER[name], color=METHOD[name], ms=1.8, alpha=0.35, mew=0, zorder=2)
    ax.plot(x, mean, "-" + MARKER[name], color=METHOD[name], ms=ms, lw=1.0, label=label or name, zorder=3,
            mec="white", mew=0.4)


def scaled_l1():
    """{N: [AUROC per data seed]} of the runs with lambda_1 scaled by N (N - 1), read from the output of
    scale_penalty_table.py."""
    out, N = {}, None
    for l in open("results/supplement/scale_penalty.txt", encoding="utf-8"):
        m = re.match(r"== N = (\d+)", l)
        if m:
            N = int(m.group(1)); out[N] = []
        m = re.match(r"\s+seed \d\s+published .*\| scaled \w AUROC ([\d.]+)", l)
        if m:
            out[N].append(float(m.group(1)))
    assert all(len(out[n]) == len(PAPER_SEEDS) for n in (200, 400)), out
    return out


def nogi_clean():
    paths = ["motiv_trig/kura_ER100_K2_s0.3_n0.5/orig_seed1", "multiseed/kura_ER100_K2_s0.3_n0.5_g2/orig_seed2",
             "multiseed/kura_ER100_K2_s0.3_n0.5_g3/orig_seed3"]                       # = sel_vs_nogi.py pairs
    return [metrics_of(json.load(open(os.path.join(R, p, "metrics.json"))))["auroc"] for p in paths]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    noise = json.load(open(os.path.join(DATA, "noise_table.json")))["kuramoto"]
    scale = json.load(open(os.path.join(DATA, "scale_table.json")))
    fig = plt.figure(figsize=(W_IN, 2.28))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.02], hspace=0.78, wspace=0.42, left=0.085, right=0.995, top=0.86, bottom=0.165)
    ax_n, ax_s, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_r = fig.add_subplot(gs[1, :])

    # (a) noise
    lv = ["0.05", "0.1", "0.2"]; x = [0.0] + [float(s) for s in lv]
    clean = scale["acc"]["100"]
    vals = lambda key, j=0: [[r[j] for r in noise[s]["rows"][key] if r is not None] for s in lv]
    line(ax_n, x, [[v[0] for v in clean["ours"]]] + vals("ours"), "HetNDL")
    line(ax_n, x, [nogi_clean()] + vals("nogi"), "NOGI")
    line(ax_n, x, [[v[0] for v in clean["arni"]]] + vals("arni_best"), "ARNI")
    ax_n.set_xlabel("phase-noise std (rad)"); ax_n.set_ylabel("AUROC"); ax_n.set_ylim(0.4, 1.03); ax_n.set_xticks(x)
    ax_n.set_xticklabels(["0", ".05", ".1", ".2"]); panel_title(ax_n, "a", r"noise, K-$\pm$"); ygrid(ax_n)

    # (b) size: accuracy
    Ns = [100, 200, 400]
    for key, name in (("ours", "HetNDL"), ("arni", "ARNI"), ("granger", "Granger")):
        line(ax_s, Ns, [[v[0] for v in scale["acc"][str(N)][key] if v] for N in Ns], name)
    sc = scaled_l1()
    per = [[v[0] for v in scale["acc"]["100"]["ours"] if v]] + [sc[N] for N in (200, 400)]
    ax_s.plot(Ns, [np.mean(v) for v in per], "--o", color=METHOD["HetNDL"], ms=3.2, lw=1.0, mfc="white", mew=0.7, zorder=4)
    ax_s.text(0.04, 0.44, r"dashed: scaled $\lambda_1$", transform=ax_s.transAxes, color=INK2, ha="left", va="top", fontsize=5.8)
    ax_s.set_xscale("log", base=2); ax_s.set_xticks(Ns); ax_s.set_xticklabels([str(n) for n in Ns]); ax_s.minorticks_off()
    ax_s.set_xlabel("nodes $N$"); ax_s.set_ylim(0.4, 1.03); panel_title(ax_s, "b", r"size, K-$\pm$"); ygrid(ax_s)

    # (c) size: seconds per epoch (HetNDL vs NOGI, same GPU)
    b = scale["bench"]["results"]; Nt = [100, 200, 400, 800]
    for name in ("HetNDL", "NOGI"):
        y = [b["%s/N%d" % (name, n)]["sec_per_epoch"] for n in Nt]
        ax_t.plot(Nt, y, "-" + MARKER[name], color=METHOD[name], lw=1.0, ms=3.2 if name == "HetNDL" else 4.2,
                  mfc=METHOD[name] if name == "HetNDL" else "none", mec=METHOD[name] if name == "NOGI" else "white", mew=0.6, label=name, zorder=3 if name == "HetNDL" else 2)
    y0 = b["HetNDL/N100"]["sec_per_epoch"]
    ax_t.plot(Nt, [y0 * (n / 100) ** 2 for n in Nt], color=INK3, lw=0.6, zorder=1)
    ax_t.text(0.04, 0.97, r"grey line: $\propto N^2$", transform=ax_t.transAxes, color=INK2, ha="left", va="top", fontsize=5.8)
    ax_t.text(800, b["HetNDL/N800"]["sec_per_epoch"] * 0.42, "%.0f GiB" % b["HetNDL/N800"]["peak_gib"], color=INK2, ha="right", va="top", fontsize=5.8)
    ax_t.set_xscale("log", base=2); ax_t.set_yscale("log"); ax_t.set_xticks(Nt); ax_t.set_xticklabels([str(n) for n in Nt]); ax_t.minorticks_off()
    ax_t.set_xlabel("nodes $N$"); ax_t.set_ylabel("s / epoch", labelpad=1); panel_title(ax_t, "c", "training cost"); ygrid(ax_t)

    # (d) roll-out
    rp = os.path.join(R, "rollout", "rollout_summary.json")
    roll = json.load(open(rp)) if os.path.exists(rp) else {}
    off = {"HetNDL": -0.24, "NOGI": -0.08, "ARNI": 0.08, "Granger": 0.24}
    for k, (key, lab) in enumerate(SETTINGS):
        rows = [r for r in roll.values() if r["bench"] == key and r["seed"] in PAPER_SEEDS]       # paper seeds 1-3
        for name, field in (("HetNDL", "hetndl"), ("NOGI", "nogi"), ("ARNI", "arni"), ("Granger", "granger")):
            v = np.array([r[field] for r in rows if r.get(field) is not None], float)
            if not len(v):
                continue
            if np.isfinite(v).all():                                   # mean over seeds, true value
                ax_r.plot(k + off[name], v.mean(), MARKER[name], color=METHOD[name], ms=3.6 if name == "HetNDL" else 3.2,
                          mec="white", mew=0.4, zorder=3, clip_on=False)
            else:                                                       # some seed diverged: hollow marker at the top
                ax_r.plot(k + off[name], TOP, MARKER[name], mfc="white", mec=METHOD[name], mew=0.8, ms=3.4, zorder=3, clip_on=False)
    ax_r.set_yscale("log"); ax_r.set_ylim(1e-3, TOP * 1.4); ax_r.set_yticks([1e-2, 1e-1, 1, 10, 100]); ax_r.set_yticklabels(["0.01", "0.1", "1", "10", "100"]); ax_r.minorticks_off(); ax_r.set_xlim(-0.6, len(SETTINGS) - 0.4)
    ax_r.set_xticks(range(len(SETTINGS))); ax_r.set_xticklabels([s[1] for s in SETTINGS], rotation=25, ha="right", rotation_mode="anchor")
    for k in range(len(SETTINGS) - 1):
        ax_r.axvline(k + 0.5, color=GRID, lw=0.5, zorder=0)
    ax_r.set_ylabel("roll-out NRMSE", labelpad=1); panel_title(ax_r, "d", "100-step roll-out error on test trajectories (lower is better)"); ygrid(ax_r)

    h = [plt.Line2D([], [], color=METHOD[n], marker=MARKER[n], lw=1.0, ms=3.4, mec="white", mew=0.4) for n in METHOD]
    fig.legend(h, list(METHOD), loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=4, handlelength=1.4, columnspacing=1.6)
    for ax in (ax_n, ax_s, ax_t, ax_r):
        ax.tick_params(pad=1.5)
    fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=300)
    print("saved", OUT, "| roll-out rows:", len(roll))


if __name__ == "__main__":
    main()
