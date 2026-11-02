# -*- coding: utf-8 -*-
"""
Fig. 2 (final, Bioinformatics): combinations cannot be searched directly in small samples, and Fig. S7.

Fig. 2
    (a) selection probability pi (max over the four strains) of the 30 most stable sets of each order in the wheat
        panel, with the value expected if top-10 membership were random (q/p) and the gate threshold;
        solid: BRS; dashed: TreeSHAP interaction values (random forest), which score every pair in every
        half-sample (no candidate pool; treeshap_stability.py)
    (b) sample-size simulation (sample_size_simulation.py): recurrence over 50 half-samples of the planted pair's
        interaction gain (direct test), of the planted pair in the TreeSHAP top 10 of all pairs, of the planted
        triad's interaction gain, and of the interaction gain of a true pair without interaction; median over ten
        datasets per size, band = interquartile range
Fig. S7
    (a) pool effect: fraction of half-samples in which the full-data top-10 sets were enumerated at all,
        and in which they reached the top 10 (mean over strains; dots = strains)
    (b) agreement of the bagged consensus ranking between two disjoint groups of 25 half-samples
        (Spearman; dots = strains)

Inputs: results_stability_v2raw/stability_summary.xlsx, results_stability_v2raw/stability_diagnostics.xlsx,
        results_treeshap_stability/treeshap_stability.xlsx, results_sample_size/sample_size_summary.xlsx
Outputs: 论文/overleaf_v6/figures/Fig2_combinations.(pdf|png), FigS7_enumeration.(pdf|png)
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ==========================================
# 1) Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
# half-sample run of all three orders (single metabolites, pairs, triads); it used the earlier autoscaled
# raw-intensity input, see the Methods. The single-metabolite run with the final log input is in Fig. 3.
STAB = ROOT / "results_stability_v2raw" / "stability_summary.xlsx"
DIAG = ROOT / "results_stability_v2raw" / "stability_diagnostics.xlsx"
TSHAP = ROOT / "results_treeshap_stability" / "treeshap_stability.xlsx"     # same 50 half-samples
SIM = ROOT / "results_sample_size" / "sample_size_summary.xlsx"
OUT_DIR = ROOT / "论文" / "overleaf_v6" / "figures"
STRAINS = ["NF", "NV", "NM", "NU"]
ORDERS = ["single", "pair", "triad"]
N_SHOW = 30
PI_SIG = 0.4
PI_BOUND = 0.7
PI_REPRO = 0.7
TOP_K = 10
N_WHEAT = 44

ORDER_COLOR = {"single": "#E69F00", "pair": "#CC79A7", "triad": "#0072B2"}
ORDER_MARKER = {"single": "o", "pair": "s", "triad": "D"}
ORDER_LABEL = {"single": "Single metabolites", "pair": "Pairs", "triad": "Triads"}
STRAIN_MARKER = {"NF": "o", "NV": "^", "NM": "s", "NU": "D"}
SIM_SERIES = [   # column in per_dataset -> (label, colour, line style, marker)
    ("planted pair: pi_int", "Pair: interaction gain", ORDER_COLOR["pair"], "-", "s"),
    ("treeshap_pi_planted_pair_top10", "Pair: TreeSHAP top 10", ORDER_COLOR["pair"], (0, (3, 1.6)), None),
    ("planted triad: pi_int", "Triad: interaction gain", ORDER_COLOR["triad"], "-", "D"),
    ("true pair, no interaction: pi_int", "Non-interacting pair", "#8C8C8C", (0, (1, 1.2)), None),
]
FIG2_H, BOT, H = 2.85, 0.42, 1.85      # legends sit above the panels
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


# ==========================================
# 2) Data
# ==========================================
pi = {o: pd.read_excel(STAB, sheet_name=f"pi_{o}") for o in ORDERS}
bounds = pd.read_excel(STAB, sheet_name="error_bounds")
p_cand = bounds.groupby("order")["p_candidates"].mean().to_dict()
pool = pd.read_excel(DIAG, sheet_name="pool_effect")
cons = pd.read_excel(DIAG, sheet_name="bagged_consensus")
# TreeSHAP: pi per strain of each strain's 50 most stable sets -> max over strains (exact for the top 30)
pi_ts = {o: pd.read_excel(TSHAP, sheet_name=f"pi_{o}").groupby("set")["pi"].max() for o in ORDERS}
sim = pd.read_excel(SIM, sheet_name="per_dataset")

# ==========================================
# 3) Fig. 2
# ==========================================
fig = plt.figure(figsize=(7.0, FIG2_H), facecolor="white")
LEG_Y = (BOT + H + 0.04) / FIG2_H

# (a) Selection probability of the most stable sets in the wheat panel
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
q_over_p = TOP_K / p_cand["single"]          # random top-10 membership among the prescreened singles
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

# (b) Sample-size simulation
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
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig2_combinations.pdf", facecolor="white")
fig.savefig(OUT_DIR / "Fig2_combinations.png", dpi=300, facecolor="white")
plt.close(fig)

# ==========================================
# 4) Fig. S7
# ==========================================
fig = plt.figure(figsize=(4.9, 2.35), facecolor="white")
H = 2.35 - BOT - 0.30

# (a) Pool effect for the full-data top-10 sets
axB = axes_in(fig, 0.45, BOT, 1.65, H)
style(axB)
w = 0.34
for i, o in enumerate(ORDERS):
    d = pool[pool.order == o]
    for j, (col, lab) in enumerate([("fulltop10_mean_enum_rate", "enumerated"), ("fulltop10_mean_pi", "in top 10")]):
        xx = i + (j - 0.5) * (w + 0.04)
        face = mpl.colors.to_rgba(ORDER_COLOR[o], 0.25) if j == 0 else ORDER_COLOR[o]
        axB.bar(xx, d[col].mean(), width=w, color=face, edgecolor=ORDER_COLOR[o], lw=0.7, zorder=2)
        for k, s in enumerate(STRAINS):
            v = d.loc[d.strain == s, col].item()
            axB.scatter(xx + (k - 1.5) * 0.06, v, s=6, marker=STRAIN_MARKER[s], facecolor="white",
                        edgecolor=INK, linewidth=0.5, zorder=3)
axB.set_xticks(range(len(ORDERS)))
axB.set_xticklabels(["Singles", "Pairs", "Triads"])
axB.set_xlim(-0.6, len(ORDERS) - 0.4)
axB.set_ylim(0, 1.0)
axB.set_ylabel("Fraction of half-samples", labelpad=2)
axB.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
hB = [mpl.patches.Patch(facecolor=mpl.colors.to_rgba(INK_2, 0.25), edgecolor=INK_2, lw=0.7),
      mpl.patches.Patch(facecolor=INK_2, edgecolor=INK_2, lw=0.7)]
axB.legend(hB, ["enumerated", "in top 10"], loc="upper right", frameon=False, handlelength=1.0,
           handleheight=0.8, borderaxespad=0.2)

# (b) Agreement of bagged consensus rankings between disjoint half-sample groups
axC = axes_in(fig, 3.30, BOT, 1.50, H)
style(axC)
for i, o in enumerate(ORDERS):
    d = cons[cons.order == o]
    vals = d["spearman_consensus_groupA_vs_B"]
    if vals.notna().sum() == 0:
        axC.text(i, 0.05, "not\nenumerated\nin both\ngroups", ha="center", va="bottom", color=INK_2, linespacing=1.0)
        continue
    axC.plot([i - 0.22, i + 0.22], [vals.mean()] * 2, color=ORDER_COLOR[o], lw=1.6, zorder=2, solid_capstyle="butt")
    for k, s in enumerate(STRAINS):
        v = d.loc[d.strain == s, "spearman_consensus_groupA_vs_B"].item()
        axC.scatter(i + (k - 1.5) * 0.09, v, s=12, marker=STRAIN_MARKER[s], color=ORDER_COLOR[o],
                    edgecolor="white", linewidth=0.4, zorder=3)
axC.axhline(0, color="#9A9A9A", lw=0.6, zorder=1)
axC.set_xticks(range(len(ORDERS)))
axC.set_xticklabels(["Singles", "Pairs", "Triads"])
axC.set_xlim(-0.6, len(ORDERS) - 0.4)
axC.set_ylim(-0.2, 1.0)
axC.set_ylabel("Spearman ρ, consensus ranking\n(half-sample group A vs B)", labelpad=2, linespacing=1.0)
axC.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
hC = [Line2D([0], [0], ls="none", marker=STRAIN_MARKER[s], ms=3.2, mfc="white", mec=INK, mew=0.6) for s in STRAINS]
fig.legend(hC, STRAINS, loc="upper right", ncol=4, bbox_to_anchor=(0.995, 0.995), frameon=False,
           handlelength=0.8, handletextpad=0.2, columnspacing=0.8, borderaxespad=0)
letter(fig, 0.02, 2.35 - 0.04, "(a)")
letter(fig, 2.70, 2.35 - 0.04, "(b)")
fig.savefig(OUT_DIR / "FigS7_enumeration.pdf", facecolor="white")
fig.savefig(OUT_DIR / "FigS7_enumeration.png", dpi=300, facecolor="white")
plt.close(fig)
print("Saved:", OUT_DIR / "Fig2_combinations.pdf", OUT_DIR / "FigS7_enumeration.pdf")
for o in ORDERS:
    print(o, "max pi", pi[o]["pi_max"].max(), "q/p", round(TOP_K / p_cand[o], 3))
