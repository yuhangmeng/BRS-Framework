# -*- coding: utf-8 -*-
"""
Fig. 4: mixture assembly (output B). Metabolites are added in order of selection probability, for all strains
together (high BNI for any strain) and for each strain against its own high-inhibition label.

(a) AUC versus the number of metabolites: strict nested cross-validation at fixed size (band: stratified bootstrap
    95% CI) and in-sample (order and AUC on the same 44 accessions); dot: size of the final mixture.
(b) Out-of-fold ROC curves (vertical average of the two repeats) of the first metabolite alone and of the mixture
    whose size was chosen inside each training set.

Inputs: data/preprocessing/*.xlsx, results/stability_selection.xlsx (Step 2), results/nested_cv.xlsx (Step 3)
Output: figures/output/Fig4_mixtures.pdf, .png
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve

from common import RES, STRAINS, auc, labels, pi_single, save, wheat

TARGETS = [("overall", "All strains", "high BNI for any strain"), ("NF", "NF", "AOA"), ("NV", "NV", "AOA"),
           ("NM", "NM", "AOB"), ("NU", "NU", "AOB")]
COLS = {"overall": STRAINS, **{s: [s] for s in STRAINS}}
K_MAX = {"overall": 30, "NF": 15, "NV": 15, "NM": 15, "NU": 15}
K_SHOW = 20
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
IN_SAMPLE = "#a9a8a2"
FS, FS_PANEL = 7.0, 9.0
FIG_W, FIG_H = 7.0, 4.35
FULL = {"6,8C-glycosylated flavone": "6,8-C-Glycosylated\nflavone", "2-Hydroxybenzenepropanoic acid":
        "2-Hydroxybenzenepropanoic\nacid", "N,N-Dimethyl-L-arginine": "N,N-Dimethyl-\nL-arginine",
        "gamma-Glutamylmethionine": "γ-Glutamylmethionine", "N2-Succinyl-L-arginine": "N2-Succinyl-\nL-arginine",
        "Ala-ala": "Ala-Ala"}
mpl.rcParams.update({
    "font.family": "Arial", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "legend.fontsize": FS,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def full_name(n):
    base, _, tag = str(n).strip().partition(" [")
    return FULL.get(base, base) + (" [" + tag if tag else "")


# ---- data ----
NCV = RES / "nested_cv.xlsx"
scur = pd.read_excel(NCV, sheet_name="curves")
mix = pd.read_excel(NCV, sheet_name="mixtures").set_index("target")
final_k = pd.read_excel(NCV, sheet_name="final_mixtures").set_index("target")[
    "k for the final mixture (inner rule on 25 full-data pairs)"]
oof = pd.read_excel(NCV, sheet_name="oof_scores")
M_FIRST, M_MIX = "first metabolite (k = 1)", "mixture, k chosen in training"

# in-sample curve: full-data order (pi, ties by full-data BRS) and AUC on the same 44 accessions
Z, aoi = wheat()
P = pi_single()
pi_full = P[[f"pi_{s}" for s in STRAINS]].set_axis(STRAINS, axis=1)
brs_full = P[[f"BRS_full_{s}" for s in STRAINS]].set_axis(STRAINS, axis=1)


def order_from(cols):
    d = pd.DataFrame({"p": pi_full[cols].max(axis=1), "b": brs_full[cols].max(axis=1).fillna(0)})
    d = d[(d.p > 0) & d.index.isin(Z.columns)]
    return list(d.sort_values(["p", "b"], ascending=False, kind="mergesort").index)


cur = {}
for t, cols in COLS.items():
    o, y = order_from(cols)[:K_MAX[t]], labels(aoi, t)
    cur[t] = pd.DataFrame({"k": range(1, len(o) + 1),
                           "AUC in-sample": [auc(Z[o[:k]].mean(axis=1).values, y) for k in range(1, len(o) + 1)],
                           "first": P.loc[o[0], "set"]})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)


def mean_roc(target, method):
    grid = np.linspace(0, 1, 201)
    d = oof[(oof.target == target) & (oof.method == method)]
    tprs = []
    for _, g in d.groupby("repeat"):
        fpr, tpr, _ = roc_curve(g["y"], g["oof_score"], drop_intermediate=False)
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    return grid, np.mean(tprs, axis=0)


# ---- figure ----
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
L, R, GAP = 0.40, 0.05, 0.17
W = (FIG_W - L - R - GAP * (len(TARGETS) - 1)) / len(TARGETS)
A_B, A_H = 2.65, 1.05
B_B, B_H = 0.96, W


def axes_in(l, b, w, h):
    return fig.add_axes([l / FIG_W, b / FIG_H, w / FIG_W, h / FIG_H])


for j, (t, title, sub) in enumerate(TARGETS):
    x0 = L + j * (W + GAP)
    d = cur[t][cur[t].k <= K_SHOW]
    s_ = scur[(scur.target == t) & (scur.k <= K_SHOW)].sort_values("k")
    k_rec = int(final_k.loc[t])

    ax = axes_in(x0, A_B, W, A_H)
    style(ax)
    ax.axhline(0.5, color=AXIS, lw=0.6, zorder=1)
    ax.fill_between(s_.k, s_["CI low"], s_["CI high"], color=BLUE, alpha=0.10, lw=0, zorder=2)
    ax.plot(d.k, d["AUC in-sample"], color=IN_SAMPLE, lw=1.2, solid_capstyle="round", zorder=3)
    ax.plot(s_.k, s_["nested-CV AUC (fixed k)"], color=BLUE, lw=1.5, solid_capstyle="round", solid_joinstyle="round",
            zorder=4)
    yk = float(s_.loc[s_.k == k_rec, "nested-CV AUC (fixed k)"].iloc[0])
    ax.scatter([k_rec], [yk], s=22, color=BLUE, edgecolor="white", linewidth=1.0, zorder=5)
    ax.scatter([0.30], [0.075], s=14, color=BLUE, edgecolor="white", linewidth=0.8, transform=ax.transAxes,
               zorder=6, clip_on=False)
    ax.text(0.36, 0.075, f"final size k = {k_rec}", transform=ax.transAxes, ha="left", va="center",
            fontsize=FS - 0.8, color=INK, zorder=6)
    ax.set_xlim(0.5, K_SHOW + 0.5)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    if j:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("AUC")
    ax.set_xlabel("Metabolites in mixture", labelpad=1.5)
    ax.set_title(f"{title}\n", fontsize=FS, fontweight="bold", pad=1, loc="left")
    ax.text(0, 1.035, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS - 0.5, color=INK_2)

    axb = axes_in(x0, B_B, W, B_H)
    style(axb)
    axb.plot([0, 1], [0, 1], color=AXIS, lw=0.6, zorder=1)
    key = []
    for method, col in ((M_FIRST, ORANGE), (M_MIX, BLUE)):
        fpr, tpr = mean_roc(t, method)
        auc_k = float(mix.loc[t, "first metabolite AUC" if method == M_FIRST else "nested-CV AUC"])
        axb.plot(fpr, tpr, color=col, lw=1.5 if method == M_MIX else 1.2, zorder=3 if method == M_MIX else 2)
        lab = full_name(cur[t]["first"].iloc[0]) if method == M_FIRST else "mixture"
        key.append((col, f"{lab}  {auc_k:.2f}"))
    axb.set_xlim(-0.01, 1.01)
    axb.set_ylim(-0.01, 1.01)
    axb.set_aspect("equal")
    axb.set_xticks([0, 0.5, 1])
    axb.set_yticks([0, 0.5, 1])
    axb.set_xticklabels(["0", ".5", "1"])
    axb.set_yticklabels(["0", ".5", "1"] if j == 0 else [])
    axb.set_xlabel("False-positive rate", labelpad=1.5)
    if j == 0:
        axb.set_ylabel("True-positive rate")
    y_in = B_B - 0.36
    for col, txt in key:
        n = txt.count("\n") + 1
        fig.add_artist(Line2D([x0 / FIG_W, (x0 + 0.14) / FIG_W], [(y_in - 0.055) / FIG_H] * 2, color=col,
                              lw=1.5, transform=fig.transFigure))
        fig.text((x0 + 0.19) / FIG_W, y_in / FIG_H, txt, ha="left", va="top", fontsize=FS - 0.8, color=INK,
                 linespacing=1.1)
        y_in -= 0.115 * n + 0.04

fig.legend([Line2D([0], [0], color=BLUE, lw=1.5), Patch(facecolor=BLUE, alpha=0.10, lw=0),
            Line2D([0], [0], color=IN_SAMPLE, lw=1.2),
            Line2D([0], [0], ls="none", marker="o", ms=4.5, mfc=BLUE, mec="white", mew=1.0)],
           ["Nested cross-validation, fixed size", "95% CI", "In-sample (same accessions)", "Final size"],
           loc="lower left", ncol=4, bbox_to_anchor=(L / FIG_W, (A_B + A_H + 0.33) / FIG_H), frameon=False,
           handlelength=1.6, columnspacing=1.4, borderaxespad=0, borderpad=0)
fig.legend([Line2D([0], [0], color=ORANGE, lw=1.2), Line2D([0], [0], color=BLUE, lw=1.5)],
           ["First metabolite alone (out-of-fold)", "Mixture, size chosen within the training folds (out-of-fold)"],
           loc="lower left", ncol=2, bbox_to_anchor=(L / FIG_W, (B_B + B_H + 0.07) / FIG_H), frameon=False,
           handlelength=1.6, columnspacing=1.4, borderaxespad=0, borderpad=0)
fig.text(0.02 / FIG_W, (A_B + A_H + 0.48) / FIG_H, "(a)", fontsize=FS_PANEL, fontweight="bold", va="center")
fig.text(0.02 / FIG_W, (B_B + B_H + 0.13) / FIG_H, "(b)", fontsize=FS_PANEL, fontweight="bold", va="center")
save(fig, "Fig4_mixtures")
print(mix[["nested-CV AUC", "CI low", "CI high", "first metabolite AUC", "k per fold"]].round(2).to_string())
print(final_k.to_string())
