"""Supplementary figure (Fig. S1 of the supplement): recovered signed graphs drawn as networks, data seed 1.
Columns = the methods of Fig. 2a-e (truth, HetNDL, ARNI, NOGI, Granger), rows = settings. Each panel draws, on the same
circular layout of the shown nodes, the k pairs with the largest |score| among the shown ordered pairs, k = number of
true edges among them (the edge budget of the graph-recovery figures of NOGI and NetNDDE; it uses the truth for display
only - the binary graph of the paper is the truth-free 2-means cut). Encoding:
  kept true edge, sign correct  solid, blue (excitatory) / terracotta (inhibitory), the colours of Fig. 2a
  kept true edge, sign wrong    dashed, colour of the predicted sign
  kept true edge, no sign       solid ink (NOGI and Granger output unsigned scores)
  spurious pair                 dashed teal
  true edge not kept            faint grey
The two directions of a pair are drawn as opposite arcs. AUROC under each panel = the full graph of that run (same
number as the per-seed tables). Large networks (Kuramoto N = 100, Grid-118) show a 24-node breadth-first ball of node 0
of the true graph (a display choice); nodes sit on the circle in the order of the Fiedler vector of the true graph.
Runs: exactly those of the tables (baseline_table.run_dirs, selected.json; ARNI = the better of full / greedy by AUROC,
the main-table rule; MixRelax-+/- = the runs of supp_readout.py x1).
Usage: PYTHONPATH=. python experiments/figures/fig_supp_topology.py [row keys ...]  (default: the rows in ROWS)
       ... --counts > results/supplement/fig_supp_topology_counts.txt  (numbers quoted in the text)"""
import glob, json, os, sys
import numpy as np
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from paperstyle import plt, W_IN, INK, INK2, INK3
from experiments.analysis.baseline_table import R, met, run_dirs, score_matrix, sel, sel_key
from experiments.analysis.binary_graph_table import signed_scores
from hetndl.gen.kuramoto import load_dataset

EXC, INH = "#2f63ad", "#c26a44"
FALSE, MISS = "#2a9d8f", "#dfe2e6"     # spurious: teal, checked for contrast against EXC / INH
OUT = "experiments/figures/out/fig_supp_topology"
COLS = ["Truth", "HetNDL", "ARNI", "NOGI", "Granger"]
# key: (label, table label of baseline_table.B or None, dataset, signed setting, ball size or None)
ALL = {"kura": (r"Kuramoto-$\pm$, $p^-=0.5$", "Kuramoto p⁻ 0.5", "kura_ER100_K2_s0.3_n0.5", True, 24),
       "mixrelax_pm": (r"MixRelax-$\pm$", None, "a1v2s_N30_n0.5_seed1", True, None),
       "mixosc_e": ("MixOsc-E, $N=30$", "A3 (edge types)", "a3_N30_seed1", False, None),
       "glv": (r"GLV-$\pm$", "GLV (signed)", "glv_N30_n0.5_seed1", True, None),
       "grid": ("Grid-118", "IEEE-118 grid", "grid118_seed1", False, 24)}
ROWS = ["kura", "mixrelax_pm", "glv"]


def runs_of(lab, ds):
    """{method: run directory relative to R} of data seed 1."""
    if lab is None:                                                       # MixRelax-+/- (supp_readout.py x1)
        b = os.path.join("a1s", ds)
        c = [os.path.join(b, "add_z%s_diff_lz0_seed1" % v) for v in ("", "_signed")]
        ours = min((p for p in c if met(p)), key=lambda p: met(p)["val_mse_best"])
        arni = [os.path.relpath(p, R) for p in sorted(glob.glob(os.path.join(R, b, "arni_*"))) if met(os.path.relpath(p, R))]
        return {"HetNDL": ours, "NOGI": os.path.join(b, "orig_seed1"), "Granger": os.path.join(b, "granger_seed1"),
                "ARNI": max(arni, key=lambda p: met(p)["auroc"])}
    rd = run_dirs(lab, ds, 1)
    arni = [p for p in (rd["ARNI full"], rd["ARNI greedy"]) if p]
    return {"HetNDL": os.path.relpath(sel[sel_key(lab, ds, 1)]["run"], R), "NOGI": rd["NOGI"], "Granger": rd["Granger"],
            "ARNI": max(arni, key=lambda p: met(p)["auroc"])}


def matrices(lab, ds):
    """{column: (signed or unsigned score matrix, has_sign, full-graph AUROC)}; truth first."""
    d = load_dataset("experiments/data/%s.npz" % ds); S = np.asarray(d["S"], float); A = S != 0
    off = ~np.eye(len(A), dtype=bool); out = {"Truth": (S * np.abs(d.get("W_true", np.abs(S))), True, None)}
    for m, p in runs_of(lab, ds).items():
        sg = signed_scores("ours" if m == "HetNDL" else ("ARNI full" if m == "ARNI" else m), p)
        W = sg[0] if sg is not None else score_matrix(p)
        out[m] = (W, sg is not None, roc_auc_score(A[off], np.abs(score_matrix(p))[off]))
    return out, A, S


def ball(A, n):
    U = A | A.T; order, seen = [0], {0}
    while len(order) < n:
        for v in list(order):
            for w in np.flatnonzero(U[v]):
                if w not in seen and len(order) < n:
                    seen.add(w); order.append(int(w))
    return np.array(order)


def spectral_order(A):
    """Circle order of the shown nodes: the Fiedler vector of the true undirected graph (fewer long chords; display only)."""
    U = (A | A.T).astype(float); L = np.diag(U.sum(1)) - U
    return np.argsort(np.linalg.eigh(L)[1][:, 1])


def draw(ax, W, signed, A, S, nodes, pos, truth=False):
    sub = np.ix_(nodes, nodes); Wn, An, Sn = W[sub], A[sub], S[sub]
    off = ~np.eye(len(nodes), dtype=bool); k = int(An[off].sum())
    if truth:
        keep = An & off
    else:
        sc = np.where(off, np.abs(Wn), -1).ravel(); keep = np.zeros(Wn.size, bool); keep[np.argsort(-sc)[:k]] = True
        keep = keep.reshape(Wn.shape)

    def arc(j, i, **kw):                                                   # edge j -> i (row j, column i)
        ax.add_patch(FancyArrowPatch(pos[j], pos[i], arrowstyle="-", connectionstyle="arc3,rad=0.12", shrinkA=1.5, shrinkB=1.5, **kw))
    for j, i in zip(*np.nonzero(An & off & ~keep)):
        arc(j, i, color=MISS, lw=0.5, zorder=1)
    for j, i in zip(*np.nonzero(keep)):
        if not An[j, i]:
            arc(j, i, color=FALSE, lw=0.55, ls=(0, (2.2, 1.4)), zorder=2)
        elif not signed:
            arc(j, i, color=INK2, lw=0.6, zorder=3)
        else:
            col = EXC if Wn[j, i] > 0 else INH
            ok = truth or np.sign(Wn[j, i]) == np.sign(Sn[j, i])
            arc(j, i, color=col, lw=0.7, ls="-" if ok else (0, (2.2, 1.4)), zorder=3)
    ax.scatter(pos[:, 0], pos[:, 1], s=4, color="white", edgecolors=INK3, linewidths=0.45, zorder=4)
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18); ax.set_aspect("equal"); ax.axis("off")
    return k


def main(rows):
    fig, axs = plt.subplots(len(rows), len(COLS), figsize=(W_IN, 1.02 * len(rows) + 0.35), squeeze=False)
    for r, key in enumerate(rows):
        name, lab, ds, signed_setting, nball = ALL[key]
        M, A, S = matrices(lab, ds)
        nodes = ball(A, nball) if nball else np.arange(len(A))
        nodes = nodes[spectral_order(A[np.ix_(nodes, nodes)])]
        t = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False) + np.pi / 2; pos = np.c_[np.cos(t), np.sin(t)]
        for c, col in enumerate(COLS):
            W, has_sign, au = M[col]; ax = axs[r, c]
            k = draw(ax, W, has_sign, A, S, nodes, pos, truth=(col == "Truth"))
            if r == 0:
                ax.set_title(col, fontsize=6.8, pad=2)
            if c == 0:
                ax.text(-0.18, 0.5, name, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.3)
                sub = "%d of %d nodes, %d edges" % (len(nodes), len(A), k) if nball else "%d nodes, %d edges" % (len(A), k)
                ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=5.4, color=INK2)
            else:
                ax.text(0.5, -0.06, "AUROC %.3f" % au, transform=ax.transAxes, ha="center", va="top", fontsize=5.6, color=INK2)
    h = [plt.Line2D([], [], color=EXC, lw=0.9, label="excitatory, sign correct"),
         plt.Line2D([], [], color=INH, lw=0.9, label="inhibitory, sign correct"),
         plt.Line2D([], [], color=EXC, lw=0.9, ls=(0, (2.2, 1.4)), label="true edge, wrong sign"),
         plt.Line2D([], [], color=INK2, lw=0.9, label="true edge (unsigned score)"),
         plt.Line2D([], [], color=FALSE, lw=0.9, ls=(0, (2.2, 1.4)), label="spurious"),
         plt.Line2D([], [], color=MISS, lw=1.2, label="true edge not in top $k$")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=5.6, bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.95, bottom=0.30 / (1.02 * len(rows) + 0.35) + 0.05, wspace=0.04, hspace=0.18)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tag = "" if rows == ROWS else "_" + "_".join(rows)
    fig.savefig(OUT + tag + ".pdf"); fig.savefig(OUT + tag + ".png", dpi=250)
    print("saved", OUT + tag)


def counts(rows):
    """Per panel: true edges kept among the k drawn pairs, spurious pairs, kept edges with the wrong sign (the numbers the
    supplement's text quotes)."""
    for key in rows:
        name, lab, ds, _, nb = ALL[key]; M, A, S = matrices(lab, ds)
        nodes = ball(A, nb) if nb else np.arange(len(A)); sub = np.ix_(nodes, nodes); An, Sn = A[sub], S[sub]
        off = ~np.eye(len(nodes), dtype=bool); k = int(An[off].sum())
        for c in COLS[1:]:
            W, has, au = M[c]; Wn = W[sub]; sc = np.where(off, np.abs(Wn), -1).ravel()
            keep = np.zeros(Wn.size, bool); keep[np.argsort(-sc)[:k]] = True; keep = keep.reshape(Wn.shape); tp = int((keep & An).sum())
            wrong = "%d" % int((keep & An & (np.sign(Wn) != np.sign(Sn))).sum()) if has else "n/a (unsigned)"
            print("%-12s %-8s nodes %3d  k %3d  true kept %3d  spurious %3d  wrong sign %s  full-graph AUROC %.3f" % (key, c, len(nodes), k, tp, k - tp, wrong, au))


if __name__ == "__main__":
    if "--counts" in sys.argv:
        counts([a for a in sys.argv[1:] if a != "--counts"] or ROWS)
    else:
        main(sys.argv[1:] or ROWS)
