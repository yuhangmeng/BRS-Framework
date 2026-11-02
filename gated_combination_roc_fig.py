# -*- coding: utf-8 -*-
"""
Supplementary Fig. S10: ROC curves of gated combinations per strain.

Per strain, against the strain's own high-inhibition label:
    grey   : the strain's most complementary gated pair (largest pi_comp in Table S16) as a fixed set, in-sample
    blue   : nested CV, single metabolite with the largest training AUC in the re-derived gated pool
    orange : nested CV, pair with the largest training AUC in the same pool
Input: supplementary/ST_gated_combination_auc.xlsx (gated_combination_auc.py)
Output: 论文/overleaf_v6/figures/FigS10_gated_roc.(pdf|png)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "supplementary" / "ST_gated_combination_auc.xlsx"
OUT = ROOT / "论文" / "overleaf_v6" / "figures" / "FigS10_gated_roc"
STRAINS = [("NF", "AOA"), ("NV", "AOA"), ("NM", "AOB"), ("NU", "AOB")]
SHORT = {"6,8C-glycosylated flavone [4893]": "GF", "6,8C-glycosylated flavone [3551]": "GF[3551]",
         "2-Hydroxybenzenepropanoic acid [3771]": "2-HBPA", "N,N-Dimethyl-L-arginine": "DMA",
         "gamma-Glutamylmethionine": "γ-Glu-Met", "Protocatechuic acid": "PCA",
         "N2-Succinyl-L-arginine": "N2-Suc-Arg", "Fatty acid - C18H30O3 [7875]": "FA C18H30O3",
         "N-Acetyl-DL-glutamic acid": "NAG"}

# same tokens as Fig. 4 (validated blue/orange pair; grey for in-sample)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
IN_SAMPLE = "#a9a8a2"
FS = 7.0
mpl.rcParams.update({
    "font.family": "Arial", "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
    "ytick.labelsize": FS - 0.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})

F = pd.read_excel(SRC, sheet_name="fixed_sets")
N = pd.read_excel(SRC, sheet_name="nested").set_index(["strain", "method"])
O = pd.read_excel(SRC, sheet_name="nested_oof")

# composite of a fixed set and strain labels, recomputed with the preprocessing of gated_combination_auc.py
lab = pd.read_excel(ROOT / "Data Preprocessing" / "filtered_metabolites.xlsx", sheet_name=0)
met_cols = [c for c in lab.columns if "Inhibition" not in str(c)]
X = pd.read_excel(ROOT / "Data Preprocessing" / "filtered_notfound_removed.xlsx", sheet_name=0,
                  decimal=",")[met_cols].apply(pd.to_numeric, errors="coerce")
X = X.apply(lambda c: c.where(c != 0, c[c != 0].min() / 2))
Z = np.log1p(X)
Z = (Z - Z.mean()) / Z.std()
Z.columns = [str(c).strip() for c in Z.columns]
y_strain = {}
for s, _ in STRAINS:
    a = pd.to_numeric(lab[[c for c in lab.columns if "Inhibition" in str(c) and str(c).endswith(s)][0]],
                      errors="coerce")
    y_strain[s] = (a >= a.quantile(0.75)).astype(int).values


def mean_roc(d):
    grid = np.linspace(0, 1, 201)
    tprs = []
    for _, r in d.groupby("repeat"):
        fpr, tpr, _ = roc_curve(r["y"], r["oof"], drop_intermediate=False)
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    return grid, np.mean(tprs, axis=0)


fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6), facecolor="white")
fig.subplots_adjust(left=0.065, right=0.985, bottom=0.15, top=0.72, wspace=0.28)
fig.legend([Line2D([0], [0], color=IN_SAMPLE, lw=1.2), Line2D([0], [0], color=BLUE, lw=1.5),
            Line2D([0], [0], color=ORANGE, lw=1.5)],
           ["Most complementary pair (fixed, in-sample)",
            "Best single metabolite, chosen in training folds (nested CV)",
            "Best pair, chosen in training folds (nested CV)"],
           loc="upper left", bbox_to_anchor=(0.055, 1.0), ncol=2, frameon=False, fontsize=FS - 0.5,
           handlelength=1.6, columnspacing=1.5, labelspacing=0.3)
for ax, (s, guild) in zip(axes, STRAINS):
    pairs = F[(F.strain == s) & (F.order == "pair")].sort_values(["pi_comp", "AUC_composite"], ascending=False)
    top = pairs.iloc[0]
    keys = top["keys"].split(" + ")
    sc = Z[keys].mean(axis=1).values
    fpr, tpr, _ = roc_curve(y_strain[s], sc)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.6, zorder=1)
    ax.plot(fpr, tpr, color=IN_SAMPLE, lw=1.2, zorder=2)
    for m, col in (("best single", BLUE), ("best pair", ORANGE)):
        x, y = mean_roc(O[(O.strain == s) & (O.method == m)])
        ax.plot(x, y, color=col, lw=1.5, zorder=3)
    labels = [(IN_SAMPLE, f"{top.AUC_composite:.2f}"),
              (BLUE, f"{N.loc[(s, 'best single'), 'AUC_nestedCV']:.2f}"),
              (ORANGE, f"{N.loc[(s, 'best pair'), 'AUC_nestedCV']:.2f}")]
    ax.text(0.97, 0.05 + 0.09 * 3, "AUC", transform=ax.transAxes, fontsize=FS - 1.0, color=INK_2, ha="right")
    for j, (col, txt) in enumerate(labels):
        yy = 0.05 + 0.09 * (2 - j)
        ax.add_line(Line2D([0.70, 0.78], [yy + 0.025, yy + 0.025], color=col, lw=1.5, transform=ax.transAxes))
        ax.text(0.97, yy, txt, transform=ax.transAxes, fontsize=FS - 1.0, color=INK, ha="right")
    name = " + ".join(SHORT.get(n, n) for n in top.combination.split(" + "))
    ax.set_title(f"{s} ({guild})", fontsize=FS, fontweight="bold", loc="left", pad=11)
    ax.text(0, 1.03, f"fixed pair: {name}", transform=ax.transAxes, fontsize=FS - 1.0, color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_aspect("equal")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("False-positive rate")
axes[0].set_ylabel("True-positive rate")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT) + ".pdf", facecolor="white")
fig.savefig(str(OUT) + ".png", dpi=300, facecolor="white")
print("Saved", str(OUT) + ".pdf/.png")
