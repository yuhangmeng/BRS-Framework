# -*- coding: utf-8 -*-
"""
Fig. S8: nested cross-validation of the consensus list (the ROC and AUC panels of the former Fig. 3; the selection
probabilities themselves are shown in the stability map, Fig. 3, and Table 2).

Panels
    (a) out-of-fold ROC curves under nested cross-validation (5-fold x 2 repeats; vertical average of the
        two repeats) for the consensus list and the comparators
    (b) pooled out-of-fold AUC with stratified bootstrap 95% CI; random sets: mean and 95% range of
        500 draws of 14 metabolites from the prescreened pool

Input: results_nestedcv/nested_cv_summary.xlsx
Output: 论文/overleaf_v6/figures/FigS8_list_nestedcv.(pdf|png)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

# ==========================================
# 1) Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
NCV = ROOT / "results_nestedcv" / "nested_cv_summary.xlsx"
OUT = ROOT / "论文" / "overleaf_v6" / "figures" / "FigS8_list_nestedcv"
METHODS = {   # key in nested_cv_summary -> (label, colour, line style)
    "stability pi>=0.4": ("Consensus list (π ≥ 0.4)", "#0072B2", "-"),
    "stability pi>=0.3": ("Consensus list (π ≥ 0.3)", "#56B4E9", "-"),
    "naive top-14": ("Single-run BRS, top 14", "#CC79A7", (0, (5, 1.6))),
    "correlation top-14": ("Correlation filter, top 14", "#009E73", (0, (2.4, 1.4))),
    "best single": ("Best single metabolite", "#E69F00", (0, (1, 1.2))),
}
ROC_METHODS = ["stability pi>=0.4", "naive top-14", "correlation top-14", "best single"]
RANDOM_KEY = "random 14 from prescreen"
INK, INK_2, GRID = "#1A1A1A", "#555555", "#E3E3E3"
FS, FS_PANEL = 7.0, 9.0
FIG_W, FIG_H = 7.0, 2.2

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


def axes_in(l, b, w, h):
    return fig.add_axes([l / FIG_W, b / FIG_H, w / FIG_W, h / FIG_H])


# ==========================================
# 2) Figure
# ==========================================
summ = pd.read_excel(NCV, sheet_name="summary")
oof = pd.read_excel(NCV, sheet_name="oof_scores")
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

# (a) Out-of-fold ROC curves (vertical average over the two repeats)
A_L, A_B, A_SIZE = 0.45, 0.40, 1.62
axA = axes_in(A_L, A_B, A_SIZE, A_SIZE)
style(axA)
axA.plot([0, 1], [0, 1], ls=(0, (1, 1.5)), lw=0.7, color="#9A9A9A", zorder=1)
grid = np.linspace(0, 1, 201)
handles, labels = [], []
for m in ROC_METHODS:
    lab, col, ls = METHODS[m]
    d = oof[oof.method == m]
    tprs = []
    for r, g in d.groupby("repeat"):
        fpr, tpr, _ = roc_curve(g["y"], g["oof_score"], drop_intermediate=False)
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    axA.plot(grid, np.mean(tprs, axis=0), color=col, lw=1.25, ls=ls, zorder=3)
    a = summ.loc[summ.method == m, "pooled_OOF_AUC_mean"].item()
    handles.append(Line2D([0], [0], color=col, lw=1.25, ls=ls))
    labels.append(f"{lab}\n(AUC {a:.2f})")
axA.set_xlim(-0.01, 1.01)
axA.set_ylim(-0.01, 1.01)
axA.set_aspect("equal")
ticks = [0, 0.25, 0.5, 0.75, 1]
axA.set_xticks(ticks)
axA.set_yticks(ticks)
axA.set_xticklabels(["0", ".25", ".5", ".75", "1"])
axA.set_yticklabels(["0", ".25", ".5", ".75", "1"])
axA.set_xlabel("False-positive rate", labelpad=2)
axA.set_ylabel("True-positive rate", labelpad=2)
axA.grid(True, color=GRID, lw=0.5, zorder=0)
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=((A_L + A_SIZE + 0.15) / FIG_W, (A_B + A_SIZE) / FIG_H),
           frameon=False, handlelength=2.2, handletextpad=0.5, labelspacing=0.55, borderaxespad=0, borderpad=0)

# (b) Out-of-fold AUC with 95% CI
B_L, B_W = 5.05, 1.85
axB = axes_in(B_L, A_B, B_W, A_SIZE)
style(axB)
axB.spines["left"].set_visible(False)
rows = list(METHODS) + [RANDOM_KEY]
for k, m in enumerate(rows):
    yy = len(rows) - 1 - k
    if m == RANDOM_KEY:
        s = summ[summ.method.str.startswith(RANDOM_KEY)].iloc[0]
        col, lab = "#8C8C8C", "Random 14 from prescreen"
    else:
        s = summ[summ.method == m].iloc[0]
        lab, col, _ = METHODS[m]
    axB.plot([s["OOF_AUC_CI_low"], s["OOF_AUC_CI_high"]], [yy, yy], color=col, lw=1.1, solid_capstyle="butt", zorder=3)
    axB.scatter([s["pooled_OOF_AUC_mean"]], [yy], s=16, color=col, edgecolor="white", linewidth=0.5, zorder=4,
                marker="o" if m != RANDOM_KEY else "D")
    axB.text(0.385, yy, lab, ha="right", va="center")
axB.axvline(0.5, color="#7A7A7A", lw=0.8, ls=(0, (1, 1.5)), zorder=1)
axB.set_yticks([])
axB.set_ylim(-0.6, len(rows) - 0.4)
axB.set_xlim(0.4, 0.95)
axB.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axB.set_xticklabels([".4", ".5", ".6", ".7", ".8", ".9"])
axB.set_xlabel("Nested-CV AUC (95% CI)", labelpad=2)
axB.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)

for x_in, letter in [(0.02, "(a)"), (3.70, "(b)")]:
    fig.text(x_in / FIG_W, (FIG_H - 0.04) / FIG_H, letter, fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")
fig.savefig(str(OUT) + ".pdf", facecolor="white")
fig.savefig(str(OUT) + ".png", dpi=300, facecolor="white")
print("Saved", str(OUT) + ".pdf/.png")
