"""Fig. 2 (paper): what HetNDL recovers, against the truth and the baselines, from the reported runs only.
Top row, (a)-(e): signed graph recovery on Kuramoto-± p- 0.5, data seed 1 (N = 100, 430 edges, 216 inhibitory). Each
    panel shows the upper-left 40 x 40 block of a weight matrix (row j, column i = edge j -> i): (a) the true signed
    weights; (b) HetNDL (reported run, runs/sel/selected.json, gauge sign applied); (c) ARNI (its better variant by AUROC,
    as in Table 2); (d) NOGI (validation-MSE snapshot, as in Table 2); (e) Granger. Every method's matrix is divided by
    its own 98th percentile of |W| over all off-diagonal pairs (truth-free display scale) and clipped to [-1, 1]; NOGI
    and Granger scores are non-negative. AUROC under each panel = full 100-node graph, same number as the protocol
    scripts for this seed.
Bottom row, (f)-(h): (f) MixRelax: every node's self-field F(x, z_i) on the read-out grid, coloured by the truth-free
    read-out class (select_k), true laws dashed; (g) MixOsc N30: the u-component of every self-field along u at v ~ 0;
    (h) MixOsc-E N30, edge-conditioned candidate chosen by validation error: the coupling curve of every learned edge
    (hetndl.edge_readout, unit-normalised), coloured by its read-out edge type; true laws dashed.
Display only: read-out groups are matched to the closest true law (majority vote) to pick their colour.
Run from the project root with PYTHONPATH=. (hetndl env)."""
import json, os, sys
import numpy as np
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(__file__))
from paperstyle import plt, W_IN, CLASS, INK, INK2, INK3, TRUE_LS, panel_title, ygrid
from hetndl.edge_readout import edge_type_readout
from hetndl.gen.kuramoto import load_dataset

D = "experiments/distill_sel"
KURA = os.path.join(D, "sel_dz2__nowu__kura_ER100_K2_s0.3_n0.5__add_z_signed_diff_lz0_seed1")
A1 = os.path.join(D, "a1_lz0__a1v2_N30_seed1__add_z_diff_lz0_seed1")
A2 = os.path.join(D, "a2__a2_N30_seed1__add_z_diff_lz0_seed1")
OUT = "experiments/figures/out/fig_laws"
R = "experiments/runs"
CROP = 40
EXC, INH = "#2f63ad", "#c26a44"


def majority_map(cluster, truth):
    return {c: int(np.bincount(truth[cluster == c]).argmax()) for c in np.unique(cluster)}


def matrices():
    """True signed weights and every method's score matrix for Kuramoto-± p- 0.5, data seed 1 (see docstring)."""
    z = np.load(os.path.join(KURA, "grids.npz")); info = json.load(open(os.path.join(KURA, "info.json")))
    A = z["A"] > 0; off = ~np.eye(len(A), dtype=bool)
    ds = "kura_ER100_K2_s0.3_n0.5"
    arni = max(("full", "greedy"), key=lambda v: json.load(open("%s/arni/%s/%s_seed1/metrics.json" % (R, ds, v)))["auroc"])
    M = [("truth", z["W_true"]), ("HetNDL", z["W"] * info["gauge_sign"]),
         ("ARNI", np.load("%s/arni/%s/%s_seed1/W.npy" % (R, ds, arni))),
         ("NOGI", np.load("%s/motiv_trig/%s/orig_seed1/W_by_mse.npy" % (R, ds))),
         ("Granger", np.load("%s/granger/%s/seed1/W.npy" % (R, ds)))]
    out = []
    for name, W in M:
        W = np.where(off, W, 0.0)
        sc = np.percentile(np.abs(W[off]), 98)                     # same display rule for the truth and every method
        au = None if name == "truth" else roc_auc_score(A[off], np.abs(W[off]))
        out.append((name, np.clip(W / sc, -1, 1), au))
    return out


def panel_matrices(axs, cax):
    cmap = LinearSegmentedColormap.from_list("signed", [INH, "#ffffff", EXC])
    for k, (ax, (name, W, au)) in enumerate(zip(axs, matrices())):
        im = ax.imshow(W[:CROP, :CROP], cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(INK3); sp.set_linewidth(0.5)
        panel_title(ax, "abcde"[k], name)
        ax.set_xlabel("216 of 430 inhibitory" if au is None else "AUROC %.3f" % au, labelpad=2,
                      color=INK2, fontsize=5.8)
    cb = plt.colorbar(im, cax=cax, ticks=[-1, 0, 1]); cb.outline.set_linewidth(0.4)
    cb.ax.set_yticklabels(["inhib.", "0", "excit."]); cb.ax.tick_params(length=1.5, pad=1)


def panel_selfield(ax, g, letter, title, labels):
    x = g["grid"][:, 0]; F = g["F_node"][:, :, 0]; cl = g["cluster"]; tr = g["classes_true"]; m = majority_map(cl, tr)
    for i in range(F.shape[0]):
        ax.plot(x, F[i], color=CLASS[m[cl[i]]], lw=0.6, alpha=0.75, zorder=2)
    for k in range(g["F_true"].shape[0]):
        ax.plot(x, g["F_true"][k, :, 0], color=INK, lw=0.7, ls=TRUE_LS, zorder=3)
    panel_title(ax, letter, title); ax.set_xlabel("own state $x$"); ax.set_xlim(x[0], x[-1]); ygrid(ax)
    ax.set_ylabel(r"self-field $F(x, z_i)$", labelpad=1)
    ax.text(0.97, 0.96, "class ARI %.2f" % adjusted_rand_score(tr, cl), transform=ax.transAxes, ha="right", va="top", color=INK2)
    h = [plt.Line2D([], [], color=CLASS[k], lw=1.4) for k in sorted(labels)] + [plt.Line2D([], [], color=INK, lw=0.7, ls=TRUE_LS)]
    ax.legend(h, [labels[k] for k in sorted(labels)] + ["true law"], loc="lower left", handlelength=1.0,
              labelspacing=0.15, borderaxespad=0.1, fontsize=5.6)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig = plt.figure(figsize=(W_IN, 2.38))
    top = fig.add_gridspec(1, 5, left=0.005, right=0.915, top=0.935, bottom=0.6, wspace=0.08)
    mats = [fig.add_subplot(top[0, k]) for k in range(5)]
    cax = fig.add_axes([0.932, 0.655, 0.011, 0.225])
    panel_matrices(mats, cax)
    bot = fig.add_gridspec(1, 3, left=0.075, right=0.995, top=0.455, bottom=0.115, wspace=0.36)
    axs = [fig.add_subplot(bot[0, k]) for k in range(3)]

    # (f) MixRelax self-fields
    g = np.load(os.path.join(A1, "grids.npz"))
    panel_selfield(axs[0], g, "f", "MixRelax", {0: r"$-x$", 1: r"$-x-0.5x^3$", 2: r"$-1.2x+0.6\tanh x$"})

    # (g) MixOsc u-component along u at v ~ 0
    g2 = np.load(os.path.join(A2, "grids.npz")); grid = g2["grid"]
    vv = np.unique(grid[:, 1]); v0 = vv[np.argmin(np.abs(vv))]
    rows = np.isclose(grid[:, 1], v0); u = grid[rows, 0]; o = np.argsort(u); u = u[o]
    cl, tr = g2["cluster"], g2["classes_true"]; m = majority_map(cl, tr)
    for i in range(g2["F_node"].shape[0]):
        axs[1].plot(u, g2["F_node"][i, rows, 0][o], color=CLASS[m[cl[i]]], lw=0.6, alpha=0.75)
    for k in range(g2["F_true"].shape[0]):
        axs[1].plot(u, g2["F_true"][k, rows, 0][o], color=INK, lw=0.7, ls=TRUE_LS)
    h = [plt.Line2D([], [], color=CLASS[k], lw=1.4) for k in (0, 1)]
    axs[1].legend(h, ["FHN", "VdP"], loc="upper left", handlelength=0.9, labelspacing=0.2, borderaxespad=0.1)
    panel_title(axs[1], "g", r"MixOsc, $\dot u$ at $v\approx0$"); axs[1].set_xlabel("own state $u$"); ygrid(axs[1])
    axs[1].set_xlim(u[0], u[-1])
    axs[1].text(0.97, 0.04, "class ARI %.2f" % adjusted_rand_score(tr, cl), transform=axs[1].transAxes, ha="right", va="bottom", color=INK2)

    # (h) MixOsc-E edge types per source (edge-conditioned candidate with the lower validation error)
    cands = ["experiments/runs/edge_b/%s/a3_N30_seed1/%s_ee2_seed1" % (v, t) for v, t in (("nonneg", "add_z_diff_lz0"), ("nowu", "add_z_signed_diff_lz0"))]
    cands = [c for c in cands if os.path.exists(os.path.join(c, "metrics.json"))]
    run = min(cands, key=lambda c: json.load(open(os.path.join(c, "metrics.json")))["val_mse_best"])
    r = edge_type_readout(run, keep_curves=True)
    d = load_dataset(json.load(open(os.path.join(run, "metrics.json")))["args"]["data"])
    et = d["etypes"][r["edge_src"], r["edge_dst"]]; m = majority_map(r["edge_groups"], et)          # display colour only
    du = r["du"]; ecol = {0: "#2f63ad", 1: "#1f9a72"}
    for c, gr in zip(r["edge_curves"], r["edge_groups"]):
        axs[2].plot(du, c, color=ecol[m[gr]], lw=0.45, alpha=0.5)
    for f in (du, np.tanh(2 * du)):                                  # source class 0 (FHN): linear; 1 (VdP): tanh(2 du)
        axs[2].plot(du, f / np.linalg.norm(f), color=INK, lw=0.7, ls=TRUE_LS)
    h = [plt.Line2D([], [], color=ecol[k], lw=1.4) for k in (0, 1)]
    axs[2].legend(h, [r"$\propto \Delta u$", r"$\propto \tanh 2\Delta u$"], loc="upper left", handlelength=0.9, labelspacing=0.2, borderaxespad=0.1)
    axs[2].text(0.97, 0.04, "%d edges\ntype ARI %.2f" % (len(et), r["ari_source_edges"]), transform=axs[2].transAxes, ha="right", va="bottom", color=INK2, fontsize=5.8)
    panel_title(axs[2], "h", "MixOsc-E, coupling laws"); axs[2].set_xlabel(r"$\Delta u = u_j - u_i$"); ygrid(axs[2]); axs[2].set_xlim(du[0], du[-1])
    for ax in axs:
        ax.tick_params(pad=1.5)
    fig.savefig(OUT + ".pdf", dpi=600); fig.savefig(OUT + ".png", dpi=300)
    print("saved", OUT, "| run (h):", run)


if __name__ == "__main__":
    main()
