# -*- coding: utf-8 -*-
"""
Fig. 2: combinations could not be searched directly in small samples.

(a) Selection probability pi (largest over the four strains) of the 30 most stable single metabolites, pairs and
    triads over 50 complementary half-samples of the wheat panel; solid: BRS, dashed: TreeSHAP interaction values of a
    random forest, which score every pair in every half-sample; dotted: pi expected if top-10 membership were random.
(b) Sample-size simulation: recurrence over 50 half-samples of the planted pair's interaction gain, of the planted
    pair in the TreeSHAP top 10, of the planted triad's interaction gain and of the interaction gain of a true pair
    without interaction; median over ten datasets per sample size, band = interquartile range.

Input : results/source_data/fig2_source_data.xlsx (benchmark outputs; Supplementary Tables S10 part B, S17, S18)
Output: figures/output/Fig2_combinations.pdf, .png
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import SOURCE, save

SRC = SOURCE / "fig2_source_data.xlsx"
ORDERS = ["single", "pair", "triad"]
N_SHOW, PI_SIG, PI_BOUND, PI_REPRO, TOP_K, N_WHEAT = 30, 0.4, 0.7, 0.7, 10, 44
ORDER_COLOR = {"single": "#E69F00", "pair": "#CC79A7", "triad": "#0072B2"}
ORDER_MARKER = {"single": "o", "pair": "s", "triad": "D"}
ORDER_LABEL = {"single": "Single metabolites", "pair": "Pairs", "triad": "Triads"}
SIM_SERIES = [   # column -> (label, colour, line style, marker)
    ("planted pair: pi_int", "Pair: interaction gain", ORDER_COLOR["pair"], "-", "s"),
    ("treeshap_pi_planted_pair_top10", "Pair: TreeSHAP top 10", ORDER_COLOR["pair"], (0, (3, 1.6)), None),
    ("planted triad: pi_int", "Triad: interaction gain", ORDER_COLOR["triad"], "-", "D"),
    ("true pair, no interaction: pi_int", "Non-interacting pair", "#8C8C8C", (0, (1, 1.2)), None),
]
FIG2_H, BOT, H = 2.85, 0.42, 1.85
INK, INK_2, GRID = "#1A1A1A", "#555555", "#E3E3E3"
FS, FS_PANEL = 7.0, 9.0
mpl.rcParams.update({
    "font.family": "Arial", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.edgecolor": INK_2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.color": INK, "ytick.color": INK,
    "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def axes_in(fig, l, b, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([l / fw, b / fh, w / fw, h / fh])


def letter(fig, x_in, y_in, txt):
    fw, fh = fig.get_size_inches()
    fig.text(x_in / fw, y_in / fh, txt, fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")


brs = pd.read_excel(SRC, sheet_name="wheat_brs")
pi = {o: brs[brs.order == o] for o in ORDERS}
p_cand = pd.read_excel(SRC, sheet_name="p_candidates").groupby("order")["p_candidates"].mean().to_dict()
ts = pd.read_excel(SRC, sheet_name="wheat_treeshap")
pi_ts = {o: ts[ts.order == o].groupby("set")["pi"].max() for o in ORDERS}   # max over strains
sim = pd.read_excel(SRC, sheet_name="simulation")

fig = plt.figure(figsize=(7.0, FIG2_H), facecolor="white")
LEG_Y = (BOT + H + 0.04) / FIG2_H

# (a) wheat panel
axA = axes_in(fig, 0.45, BOT, 2.85, H)
style(axA)
axA.axhspan(PI_BOUND, 1.0, color="#F2F2F2", zorder=0, lw=0)
axA.text(N_SHOW, 0.985, f"error-bound regime (π ≥ {PI_BOUND})", ha="right", va="top", color=INK_2)
axA.axhline(PI_SIG, color=INK_2, lw=0.8, ls=(0, (4, 2)), zorder=1)
axA.text(N_SHOW, PI_SIG + 0.015, f"selection threshold (π ≥ {PI_SIG})", ha="right", va="bottom", color=INK_2)
x = np.arange(1, N_SHOW + 1)
for o in ORDERS:
    v = np.sort(pi[o]["pi_max"].values)[::-1][:N_SHOW]
    axA.plot(x, v, color=ORDER_COLOR[o], lw=1.2, marker=ORDER_MARKER[o], ms=2.6, mec="white", mew=0.3, zorder=3)
    vt = np.sort(pi_ts[o].values)[::-1][:N_SHOW]
    axA.plot(np.arange(1, len(vt) + 1), vt, color=ORDER_COLOR[o], lw=1.0, ls=(0, (3, 1.6)), zorder=3)
q_over_p = TOP_K / p_cand["single"]
axA.axhline(q_over_p, color=ORDER_COLOR["single"], lw=0.8, ls=(0, (1, 1.2)), zorder=2)
axA.text(N_SHOW, q_over_p + 0.012, f"random selection, singles (q/p = {q_over_p:.2f})", ha="right", va="bottom",
         color=INK_2)
axA.set_xlim(0, N_SHOW + 0.5)
axA.set_ylim(0, 1.0)
axA.set_xticks([1, 10, 20, 30])
axA.set_xlabel("Rank of set (by selection probability)", labelpad=2)
axA.set_ylabel("Selection probability π\n(wheat, max over strains)", labelpad=2, linespacing=1.0)
axA.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
hA = [Line2D([0], [0], color=ORDER_COLOR[o], lw=1.2, marker=ORDER_MARKER[o], ms=2.6, mec=ORDER_COLOR[o], mew=0.3)
      for o in ORDERS] + [Line2D([0], [0], color=INK_2, lw=1.2), Line2D([0], [0], color=INK_2, lw=1.0, ls=(0, (3, 1.6)))]
fig.legend([hA[0], hA[3], hA[1], hA[4], hA[2]], [ORDER_LABEL["single"], "BRS", ORDER_LABEL["pair"],
           "TreeSHAP (RF)", ORDER_LABEL["triad"]], loc="lower left", bbox_to_anchor=(0.45 / 7.0, LEG_Y), ncol=3,
           frameon=False, handlelength=1.8, borderaxespad=0, columnspacing=1.0, labelspacing=0.25)

# (b) sample-size simulation
axB = axes_in(fig, 4.05, BOT, 2.85, H)
style(axB)
axB.axhspan(PI_REPRO, 1.0, color="#F2F2F2", zorder=0, lw=0)
axB.text(23, 0.985, f"reproducible (π ≥ {PI_REPRO})", ha="left", va="top", color=INK_2, zorder=4,
         bbox=dict(boxstyle="square,pad=0.1", fc="#F2F2F2", ec="none"))
axB.plot([N_WHEAT, N_WHEAT], [0, 0.9], color=INK_2, lw=0.8, ls=(0, (4, 2)), zorder=1)
axB.text(N_WHEAT + 3, 0.86, f"wheat panel\n(n = {N_WHEAT})", ha="left", va="top", color=INK_2, linespacing=1.0)
ns = sorted(sim["n"].unique())
hB, lB = [], []
for col, lab, c, ls, mk in SIM_SERIES:
    g = sim.groupby("n")[col]
    med, q1, q3 = g.median().reindex(ns), g.quantile(0.25).reindex(ns), g.quantile(0.75).reindex(ns)
    if ls == "-":
        axB.fill_between(ns, q1, q3, color=c, alpha=0.14, lw=0, zorder=2)
    axB.plot(ns, med, color=c, lw=1.2 if ls == "-" else 1.0, ls=ls, marker=mk, ms=2.8, mec="white", mew=0.3,
             zorder=3)
    hB.append(Line2D([0], [0], color=c, lw=1.2 if ls == "-" else 1.0, ls=ls, marker=mk, ms=2.8, mec=c, mew=0.3))
    lB.append(lab)
axB.set_xlim(20, 210)
axB.set_ylim(0, 1.0)
axB.set_xticks(ns)
axB.set_xlabel("Number of samples (synthetic)", labelpad=2)
axB.set_ylabel("Recurrence over half-samples\n(median, IQR band)", labelpad=2, linespacing=1.0)
axB.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
fig.legend(hB, lB, loc="lower left", bbox_to_anchor=(3.80 / 7.0, LEG_Y), ncol=2, frameon=False, handlelength=2.0,
           borderaxespad=0, columnspacing=1.0, labelspacing=0.25)

letter(fig, 0.02, FIG2_H - 0.04, "(a)")
letter(fig, 3.55, FIG2_H - 0.04, "(b)")
save(fig, "Fig2_combinations")
for o in ORDERS:
    print(f"{o}: largest pi {pi[o]['pi_max'].max():.2f}")
