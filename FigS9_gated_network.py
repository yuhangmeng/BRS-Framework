# -*- coding: utf-8 -*-
"""
Supplementary Fig. S9: hypothesis network of the gated combination test (Step 4).

One panel per strain. Nodes are the members of the strain's gated pool (colour: compound group; size: selection
probability pi for that strain; dagger: single-accession leverage >= 0.7). Edges are the pairs of Table S16:
    solid grey   complementarity, width ~ pi_comp (fraction of half-samples with an additive gain > 0.02)
    dashed arc   interaction, width ~ pi_int (fraction with a gain from the product term > 0.02)
Edges are drawn when pi >= MIN_PI (about four times the label-permutation fractions, which were <= 0.04). Values >= LABEL_PI are printed.

Inputs: supplementary/ST_gated_combinations.xlsx, supplementary/ST_incremental_signature.xlsx
Output: 论文/overleaf_v6/figures/FigS9_gated_network.(pdf|png)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent
GATED = ROOT / "supplementary" / "ST_gated_combinations.xlsx"
INC = ROOT / "supplementary" / "ST_incremental_signature.xlsx"
OUT = ROOT / "论文" / "overleaf_v6" / "figures" / "FigS9_gated_network"
STRAINS = [("NF", "AOA"), ("NV", "AOA"), ("NM", "AOB"), ("NU", "AOB")]
MIN_PI, LABEL_PI = 0.15, 0.20
SHORT = {"6,8C-glycosylated flavone [4893]": "GF [4893]", "6,8C-glycosylated flavone [3129]": "GF [3129]",
         "6,8C-glycosylated flavone [3551]": "GF [3551]", "2-Hydroxybenzenepropanoic acid [3771]": "2-HBPA",
         "N,N-Dimethyl-L-arginine": "DMA", "gamma-Glutamylmethionine": "γ-Glu-Met",
         "Protocatechuic acid": "PCA", "N2-Succinyl-L-arginine": "N2-Suc-Arg",
         "Fatty acid - C18H30O3 [7875]": "FA C18H30O3", "N-Acetyl-DL-glutamic acid": "NAG",
         "N-acetyltryptophan": "N-Ac-Trp"}
GROUP = {"Flavonoids": "phenolic", "Phenylpropanoids": "phenolic", "Amino acids and derivatives": "nitrogen",
         "Lipids": "other"}

# palette: validated categorical slots 1-3 (blue, orange, aqua) + neutral grey; aqua is always direct-labelled
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#a9a8a2"
INK, INK_2 = "#0b0b0b", "#52514e"
FILL = {"phenolic": BLUE, "nitrogen": AQUA, "other": NEUTRAL}
FS = 7.0
mpl.rcParams.update({"font.family": "Arial", "font.size": FS, "text.color": INK, "pdf.fonttype": 42,
                     "ps.fonttype": 42})

pools = pd.read_excel(GATED, sheet_name="pools").set_index("strain")["pool"]
comb = pd.read_excel(GATED, sheet_name="combinations")
wet = pd.read_excel(INC, sheet_name="wetlab_lists")

fig = plt.figure(figsize=(7.0, 2.45), facecolor="white")
ratios = [1.0, 0.72, 1.0, 1.28]
gs = fig.add_gridspec(1, 4, width_ratios=ratios, left=0.01, right=0.99, bottom=0.01, top=0.86, wspace=0.04)
for col, (s, guild) in enumerate(STRAINS):
    ax = fig.add_subplot(gs[0, col])
    names = pools[s].split("; ")
    meta = wet[wet.strain == s].set_index("metabolite_added")
    n = len(names)
    ang = np.deg2rad([180.0, 0.0]) if n == 2 else np.deg2rad(90 - 360 * np.arange(n) / n)
    r_node = 0.7 if n == 2 else 1.0
    pos = {m: r_node * np.array([np.cos(a), np.sin(a)]) for m, a in zip(names, ang)}

    pairs = comb[(comb.strain == s) & (comb.order == "pair")]
    placed = []                                       # label positions, to keep edge labels apart

    def put_label(cands, txt, color):
        free = [c for c in cands if all(np.linalg.norm(c - q) > 0.42 for q in placed)]
        xy = free[0] if free else cands[0]
        placed.append(xy)
        ax.text(*xy, txt, fontsize=FS - 1.2, color=color, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    # interaction arcs first (their label position is fixed), then straight complementarity edges
    for _, r in pairs.iterrows():
        a, b = r.combination.split(" + ")
        p1, p2 = pos[a], pos[b]
        if r.pi_int >= MIN_PI:
            d = p2 - p1
            perp = np.array([d[1], -d[0]])
            mid_c = (p1 + p2) / 2
            rad = 0.5 if np.linalg.norm(mid_c + 0.25 * perp) >= np.linalg.norm(mid_c - 0.25 * perp) else -0.5
            ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                         color=ORANGE, lw=0.4 + 5 * r.pi_int, linestyle=(0, (3, 1.5)),
                                         shrinkA=0, shrinkB=0, zorder=2))
            if r.pi_int >= LABEL_PI:
                put_label([mid_c + 0.5 * rad * perp], f"{r.pi_int:.2f}", ORANGE)   # arcs bow away from the centre
    for _, r in pairs.iterrows():
        a, b = r.combination.split(" + ")
        p1, p2 = pos[a], pos[b]
        if r.pi_comp >= MIN_PI:
            ax.plot(*zip(p1, p2), color=INK_2, lw=0.4 + 5 * r.pi_comp, alpha=0.85, solid_capstyle="round",
                    zorder=1)
            if r.pi_comp >= LABEL_PI:
                put_label([p1 + t * (p2 - p1) for t in (0.36, 0.64, 0.5, 0.25, 0.75)], f"{r.pi_comp:.2f}", INK_2)

    for m in names:
        row = meta.loc[m]
        grp = GROUP.get(str(row["class"]).strip(), "other")
        ax.scatter(*pos[m], s=40 + 520 * row.pi_strain, color=FILL[grp], edgecolor="white", linewidth=1.0,
                   zorder=3)
        lab = SHORT.get(m, m) + ("$^\\dagger$" if row.leverage >= 0.7 else "")
        v = pos[m]
        off = v * 1.30 + np.array([0, 0.22 * np.sign(v[1]) if abs(v[0]) < 0.3 else 0])
        ha = "center" if abs(v[0]) < 0.3 else ("left" if v[0] > 0 else "right")
        if n == 2:
            off = v + np.array([0, -0.42])
            ha = "center"
        ax.text(*off, lab, fontsize=FS - 0.6, ha=ha, va="center", color=INK, zorder=5)
    ax.set_xlim(-2.0 * ratios[col], 2.0 * ratios[col])     # same scale in every panel
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{s} ({guild})", fontsize=FS, fontweight="bold", loc="left", pad=2)
    if not ((pairs.pi_comp >= MIN_PI) | (pairs.pi_int >= MIN_PI)).any():
        ax.text(0, -1.25, f"no pair with π ≥ {MIN_PI:.2f}", fontsize=FS - 1.0, color=INK_2,
                ha="center", va="center")

handles = [Line2D([0], [0], marker="o", ls="", markersize=6, markerfacecolor=BLUE, markeredgecolor="white"),
           Line2D([0], [0], marker="o", ls="", markersize=6, markerfacecolor=AQUA, markeredgecolor="white"),
           Line2D([0], [0], marker="o", ls="", markersize=6, markerfacecolor=NEUTRAL, markeredgecolor="white"),
           Line2D([0], [0], color=INK_2, lw=2.2),
           Line2D([0], [0], color=ORANGE, lw=2.2, ls=(0, (3, 1.5)))]
labels = ["Phenolic (flavonoid, phenylpropanoid)", "Nitrogen-containing (amino-acid derived)", "Lipid",
          "Complementarity (π$_{comp}$)", "Interaction (π$_{int}$)"]
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=3, frameon=False,
           fontsize=FS - 0.5, handlelength=1.8, columnspacing=1.4, labelspacing=0.35)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT) + ".pdf", facecolor="white")
fig.savefig(str(OUT) + ".png", dpi=300, facecolor="white")
print("Saved", str(OUT) + ".pdf/.png")
