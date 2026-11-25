"""Shared style for the paper's result figures (Fig. 2, Fig. 3). Width = LNCS text width 12.2 cm = 4.8 in,
fonts as in the framework figure (Arial, mathtext Arial), printed sizes >= 6 pt.

Colours (checked for contrast and colour-vision deficiency on a light surface, all pairs):
  METHOD  HetNDL / NOGI / ARNI / Granger  -> ALL CHECKS PASS (worst CVD dE 9.0; violet needs labels: legend + markers)
  CLASS   mechanism classes A / B / C      -> ALL CHECKS PASS (A / B are the framework figure's terracotta / violet,
          C aqua; worst CVD dE 7.2 -> every class curve is also labelled directly with its equation)
Truth is drawn in ink, dashed only for the true law (reference), never in a series colour."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W_IN = 4.8
METHOD = {"HetNDL": "#2f63ad", "NOGI": "#d0683a", "ARNI": "#1f9a72", "Granger": "#b08ae0"}
MARKER = {"HetNDL": "o", "NOGI": "s", "ARNI": "^", "Granger": "D"}
CLASS = ["#c26a44", "#6f62b8", "#1f9a72"]
INK, INK2, INK3, GRID = "#1f2328", "#50565e", "#868c94", "#e3e6ea"
TRUE_LS = (0, (3.0, 1.6))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "font.size": 6.5, "axes.titlesize": 6.8, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK3, "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600,
})


def panel_title(ax, letter, text):
    ax.set_title(r"$\bf{(%s)}$ %s" % (letter, text), loc="left", pad=3)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
