# -*- coding: utf-8 -*-
"""
Stability map: how the reproducible single metabolites stand out among all 413 metabolites (one panel per strain).

    x  Pearson correlation of the metabolite (log-transformed, autoscaled) with the strain's AOI% (all 44 accessions);
       dashed line: prescreen threshold r = 0.2
    y  selection probability pi: fraction of 50 complementary half-samples in which the metabolite was in the strain's
       BRS top 10; dashed line: gate threshold 0.4
    filled, labelled   pi >= 0.4 (the reproducible metabolites; colour: compound group; dagger: leverage >= 0.7)
    ring               in the strain's top 10 of the single full-data BRS run
    grey               all other metabolites
Each panel lists the funnel 413 -> r > 0.2 -> pi >= 0.4 and how many full-data top-10 metabolites were not reproducible.

Inputs: Data Preprocessing/filtered_metabolites_log.xlsx, results_stability_uni/stability_summary_uni.xlsx,
        results_v3_log/univariate_BRS.xlsx, name.xlsx
Output: 论文/overleaf_v6/figures/Fig_stability_map.(pdf|png)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Data Preprocessing" / "filtered_metabolites_log.xlsx"
STAB = ROOT / "results_stability_uni" / "stability_summary_uni.xlsx"
FULL = ROOT / "results_v3_log" / "univariate_BRS.xlsx"
NAME_MAP = ROOT / "name.xlsx"
OUT = ROOT / "论文" / "overleaf_v6" / "figures" / "Fig_stability_map"
STRAINS = [("NF", "AOA"), ("NV", "AOA"), ("NM", "AOB"), ("NU", "AOB")]
R_MIN, PI_GATE, LEV_FLAG, TOP_K = 0.2, 0.4, 0.7, 10
N2 = "N2"          # "N2" (as in the text) or "N²"; not yet decided
SHORT = {"6,8C-glycosylated flavone": "6,8-C-Glycosylated\nflavone", "2-Hydroxybenzenepropanoic acid":
         "2-Hydroxybenzenepropanoic\nacid", "N,N-Dimethyl-L-arginine": "N,N-Dimethyl-\nL-arginine",
         "gamma-Glutamylmethionine": "γ-Glutamylmethionine", "Protocatechuic acid": "Protocatechuic acid",
         "N2-Succinyl-L-arginine": N2 + "-Succinyl-\nL-arginine", "Ala-ala": "Ala-Ala"}
GROUP = {"Flavonoids": "phenolic", "Phenylpropanoids": "phenolic", "Amino acids and derivatives": "nitrogen"}

# same tokens as Figs S7/S8: validated categorical slots (blue, aqua) + neutral; every coloured point is labelled
BLUE, AQUA, NEUTRAL = "#2a78d6", "#1baf7a", "#a9a8a2"
INK, INK_2, GRID, AXIS, BG = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#d6d5cf"
SHADE = "#f3f2ef"
FILL = {"phenolic": BLUE, "nitrogen": AQUA, "other": NEUTRAL}
FS = 7.0
mpl.rcParams.update({
    "font.family": "Arial", "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
    "ytick.labelsize": FS - 0.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ==========================================
# 1) Data
# ==========================================
inp = pd.read_excel(INPUT, sheet_name=0, decimal=",")
tcol = {s: [c for c in inp.columns if "Inhibition" in str(c) and str(c).endswith(s)][0] for s, _ in STRAINS}
mets = [c for c in inp.columns if c not in tcol.values()]
keys = [str(c).strip() for c in mets]
X = inp[mets].apply(pd.to_numeric, errors="coerce")
X.columns = keys
pis = pd.read_excel(STAB, sheet_name="pi_single")
pis["key"] = pis["key"].astype(str).str.strip()
pis = pis.set_index("key")
nm = pd.read_excel(NAME_MAP, sheet_name=0, dtype=str).set_index("ID")
name = {k.strip(): str(v).strip() for k, v in nm["Compound name"].items()}
klass = {k.strip(): str(v).strip() for k, v in nm["CHEMICAL CLASS"].items()}
dup = pd.Series(name).value_counts()


def label(k):
    b = SHORT.get(name.get(k, k), name.get(k, k))
    if dup.get(name.get(k, k), 0) > 1 and "_" in k:
        b += f" [{k.split('_')[1]}]"
    return b


# ==========================================
# 2) Figure
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), facecolor="white")
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.075, top=0.88, wspace=0.16, hspace=0.30)
for ax, (s, guild) in zip(axes.ravel(), STRAINS):
    y = pd.to_numeric(inp[tcol[s]], errors="coerce")
    r = X.apply(lambda c: np.corrcoef(c, y)[0, 1])
    pi = pis[f"pi_{s}"].reindex(keys).fillna(0.0)
    lev = pis[f"leverage_{s}"].reindex(keys)
    full = pd.read_excel(FULL, sheet_name=f"Inhibition (pct)-{s}_Uni_All").sort_values("BRS", ascending=False)
    top_full = set(full["Feature"].astype(str).str.strip().head(TOP_K))
    gated = [k for k in keys if pi[k] >= PI_GATE]

    ax.fill_between([R_MIN, 0.95], PI_GATE, 1.0, color=SHADE, lw=0, zorder=0)
    ax.axvline(R_MIN, color=INK_2, lw=0.7, ls=(0, (4, 2)), zorder=1)
    other = [k for k in keys if k not in gated]
    jit = np.random.default_rng(0).uniform(-0.006, 0.006, len(other))   # separate the many points at pi = 0
    ax.scatter(r[other], pi[other] + np.where(pi[other] == 0, jit, 0), s=7, color=BG, lw=0, zorder=2)
    for k in top_full:
        ax.scatter(r[k], pi[k], s=46, facecolor="none", edgecolor=INK, linewidth=0.8, zorder=4)
    # labels to the right of the points, stacked downwards so that they never overlap
    y_prev, h_prev = np.inf, 0.0
    x_col = max([r[k] for k in gated], default=0.5) + 0.07
    for k in sorted(gated, key=lambda k: -pi[k]):
        grp = GROUP.get(klass.get(k, ""), "other")
        ax.scatter(r[k], pi[k], s=34, color=FILL[grp], edgecolor="white", linewidth=0.6, zorder=5)
        txt = label(k) + ("$^\\dagger$" if lev.get(k, 0) >= LEV_FLAG else "")
        h = 0.045 * (txt.count("\n") + 1)
        y_lab = min(pi[k] + 0.012, y_prev - (h_prev + h) / 2 - 0.012)
        y_prev, h_prev = y_lab, h
        x_lab = x_col
        ax.plot([r[k] + 0.018, x_lab - 0.008], [pi[k], y_lab], color=INK_2, lw=0.5, zorder=4)
        ax.text(x_lab, y_lab, txt, fontsize=FS - 0.6, color=INK, ha="left", va="center", zorder=6,
                linespacing=1.1, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

    n_pre = int((r > R_MIN).sum())
    lost = len(top_full - set(gated))
    ax.text(0.02, 0.97, f"413 metabolites\n{n_pre} with r > {R_MIN}\n{len(gated)} with π ≥ {PI_GATE}",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS - 0.6, color=INK_2, linespacing=1.25)
    ax.text(0.02, 0.62, f"full-data top 10:\n{lost} not reproducible", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS - 0.6, color=INK_2, linespacing=1.25)
    ax.set_title(f"{s} ({guild})", fontsize=FS + 0.5, fontweight="bold", loc="left", pad=3)
    ax.set_xlim(-0.55, 0.95)
    ax.spines["bottom"].set_bounds(-0.55, 0.95)
    ax.set_ylim(-0.03, 0.85)
    ax.set_xticks([-0.3, 0, 0.3, 0.6, 0.9])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
fig.canvas.draw()
ren = fig.canvas.get_renderer()
right = 0.95
for ax in axes.ravel():
    for t in ax.texts:
        bb = t.get_window_extent(ren).transformed(ax.transData.inverted())
        right = max(right, bb.x1 + 0.02)
for ax in axes.ravel():
    ax.set_xlim(-0.55, right)
    ax.grid(False, axis="y")
    for yv in ax.get_yticks():
        ax.plot([-0.55, 0.95], [yv, yv], color=GRID, lw=0.5, zorder=0)
    ax.axhline(PI_GATE, xmin=0, xmax=(0.95 + 0.55) / (right + 0.55), color=INK_2, lw=0.7, ls=(0, (4, 2)), zorder=1)
for ax in axes[1]:
    ax.set_xlabel("Correlation with the strain's AOI% (r, all 44 accessions)")
for ax in axes[:, 0]:
    ax.set_ylabel("Selection probability π (50 half-samples)")

handles = [Line2D([0], [0], marker="o", ls="", ms=5.5, mfc=BLUE, mec="white"),
           Line2D([0], [0], marker="o", ls="", ms=5.5, mfc=AQUA, mec="white"),
           Line2D([0], [0], marker="o", ls="", ms=6.5, mfc="none", mec=INK, mew=0.8),
           Line2D([0], [0], marker="o", ls="", ms=3.5, mfc=BG, mec=BG)]
labels = ["Reproducible (π ≥ 0.4), phenolic", "Reproducible (π ≥ 0.4), nitrogen-containing",
          "Top 10 of a single full-data BRS run", "Other metabolites"]
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 0.995), ncol=2, frameon=False,
           fontsize=FS - 0.3, handletextpad=0.4, columnspacing=1.6, labelspacing=0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT) + ".pdf", facecolor="white")
fig.savefig(str(OUT) + ".png", dpi=300, facecolor="white")
print("Saved", str(OUT) + ".pdf/.png")
