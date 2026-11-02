# -*- coding: utf-8 -*-
"""
Fig. 2 (final, Bioinformatics): BRS heatmaps of the prioritized single metabolites, pairs and triads.

Replaces the old Fig. 4 (Figure4_heatmap*.py). Row selection is unchanged:
    per strain (NF, NV, NM, NU): top-10 sets by BRS from the *_All sheets
    -> union across strains -> mean BRS over the four strains, counting a strain
       whose top-10 list does not contain the set as 0
    -> the N_ROWS sets with the highest mean, ordered by that mean.
These are the 30 candidate sets of Top5_single_pair_triple_BRS.xlsx / Table 3 / Fig. 4.

Fixes vs the old figure
    - sequential single-hue palette (the old coolwarm drew "not in top-10" as deep blue)
    - cells outside a strain's top-10 drawn light grey with an en dash
    - BRS value printed in every coloured cell
    - AOA (NF, NV) / AOB (NM, NU) bracket above the strain columns
    - full metabolite names, wrapped, never truncated

Cross-checks (asserted): per-strain top-10 vs the *_Top10/_Top30 sheets; rows and values vs
supplementary/Supplementary_Data.xlsx (Fig4a/b/c_*_heatmap_ID) and Top5_single_pair_triple_BRS.xlsx.

Outputs
    论文/overleaf_v6/figures/Fig2_heatmap.pdf   (vector)
    论文/overleaf_v6/figures/Fig2_heatmap.png   (300 dpi preview)
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from pathlib import Path

# ==========================================
# 1) Configuration
# ==========================================
RESULTS = Path(os.environ.get("BRS_OUT_DIR", "results"))   # BRS run to plot
PUBLISHED_RUN = RESULTS == Path("results")                 # cross-check against Supplementary_Data only then
BRS_FILES = {   # order -> (workbook, sheet kind, set column)
    "single": (RESULTS / "univariate_BRS.xlsx", "Uni", "Feature"),
    "pair": (RESULTS / "pairwise_BRS.xlsx", "Pairs", "Pair"),
    "triad": (RESULTS / "triple_BRS.xlsx", "Triplets", "Triplet"),
}
SUPP_PATH = r"supplementary/Supplementary_Data.xlsx"
SUPP_SHEETS = {"single": "Fig4a_uni_heatmap_ID", "pair": "Fig4b_pair_heatmap_ID", "triad": "Fig4c_trip_heatmap_ID"}
CANDIDATE_PATH = (RESULTS / "candidate_sets.xlsx" if (RESULTS / "candidate_sets.xlsx").exists()
                  else Path("Top5_single_pair_triple_BRS.xlsx"))   # written by build_candidate_sets.py
CANDIDATE_SHEET = "Sheet1"
NAME_MAP_PATH = r"name.xlsx"

OUT_DIR = Path("论文/overleaf_v6/figures")
OUT_PDF = OUT_DIR / "Fig2_heatmap.pdf"
OUT_PNG = OUT_DIR / "Fig2_heatmap.png"

STRAINS = ["NF", "NV", "NM", "NU"]
GROUPS = [("AOA", ["NF", "NV"]), ("AOB", ["NM", "NU"])]
TOPK_PER_STRAIN = 10
N_ROWS = 10               # rows per panel (None = the full union of the per-strain top-10 lists)

DISPLAY_FIX = {"L-TRYPTOPHAN": "L-Tryptophan", "Uric Acid": "Uric acid", "Ala-ala": "Ala-Ala",
               "Isoshaftoside": "Isoschaftoside", "4-METHYLCATECHOL": "4-Methylcatechol"}

# Sequential single-hue blue (light -> dark), one shared scale for all panels
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
VMIN, VMAX = 0.2, 0.9
NA_FACE = "#EBEBEB"       # not in this strain's top-10
NA_INK = "#8A8A8A"
EN_DASH = "–"

INK = "#1A1A1A"
INK_2 = "#555555"
FS = 7.0
FS_PANEL = 9.0
FIG_W = 7.0

# Cell geometry (inches)
CELL_PITCH = 0.31
GROUP_GAP = 0.07          # extra space between the AOA and AOB columns
CELL_GAP = 0.025          # white gap around every cell
LABEL_PAD = 0.05

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": FS,
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ==========================================
# 2) Per-strain top-10 lists, union, mean-BRS ranking
# ==========================================
def canon(s: str) -> str:
    return " | ".join(sorted(t.strip() for t in str(s).split("|")))


print("--- Step 1: Per-strain top-10 lists from the *_All sheets ---")
panels = {}
for order, (path, kind, col) in BRS_FILES.items():
    xls = pd.ExcelFile(path)
    tops, union_n = {}, None
    for s in STRAINS:
        d = pd.read_excel(xls, f"Inhibition (pct)-{s}_{kind}_All")
        d["key"] = d[col].map(canon)
        if not d["key"].is_unique:
            raise ValueError(f"{path} {s}: duplicated sets after canonicalisation")
        d = d.sort_values("BRS", ascending=False).reset_index(drop=True)
        if np.isclose(d["BRS"].iloc[TOPK_PER_STRAIN - 1], d["BRS"].iloc[TOPK_PER_STRAIN]):
            raise ValueError(f"{path} {s}: tie at rank {TOPK_PER_STRAIN}")
        tops[s] = d.head(TOPK_PER_STRAIN).set_index("key")["BRS"]
        top_sheet = next(sh for sh in xls.sheet_names if sh.startswith(f"Inhibition (pct)-{s}_{kind}_Top"))
        t = pd.read_excel(xls, top_sheet)
        assert list(t[col].map(canon).head(TOPK_PER_STRAIN)) == list(tops[s].index), f"{top_sheet} differs"
    mat = pd.DataFrame(tops)[STRAINS]                       # union of the four lists; NaN = not in top-10
    mat["mean_BRS"] = mat[STRAINS].fillna(0.0).mean(axis=1)
    mat = mat.sort_values("mean_BRS", ascending=False)
    union_n = len(mat)
    shown = mat if N_ROWS is None else mat.head(N_ROWS)
    panels[order] = {"mat": shown.copy(), "union_n": union_n}
    print(f"✅ {order}: union of per-strain top-{TOPK_PER_STRAIN} = {union_n} sets; showing {len(shown)}")


# ==========================================
# 3) Cross-checks against the supplementary data and the candidate file
# ==========================================
print("\n--- Step 2: Cross-checking against Supplementary_Data.xlsx and Top5_single_pair_triple_BRS.xlsx ---")
cand = pd.read_excel(CANDIDATE_PATH, sheet_name=CANDIDATE_SHEET)
cand_order = {canon(v): str(v) for v in cand.iloc[:, 0]}  # canonical key -> ID order used in Table 3
for order, p in panels.items():
    m = p["mat"]
    if not PUBLISHED_RUN:
        assert set(m.index) <= set(cand_order), f"{order}: rows missing from {CANDIDATE_PATH}"
        print(f"✅ {order}: rows match {CANDIDATE_PATH}")
        continue
    sup = pd.read_excel(SUPP_PATH, sheet_name=SUPP_SHEETS[order])
    sup["key"] = sup["feature_id"].map(canon)
    sup = sup.set_index("key")
    if N_ROWS == 10:
        assert list(sup.index) == list(m.index), f"{SUPP_SHEETS[order]}: row set/order differs"
        for s in STRAINS:
            a = m[s].fillna(0.0).values
            b = sup.loc[m.index, f"Inhibition (%)-{s}"].values
            assert np.allclose(a, b, atol=1e-6), f"{SUPP_SHEETS[order]} {s}: values differ"
        assert set(m.index) <= set(cand_order), f"{order}: rows missing from {CANDIDATE_PATH}"
        print(f"✅ {order}: rows, order and BRS values match {SUPP_SHEETS[order]} and {CANDIDATE_PATH}")
    hv = pd.read_excel(BRS_FILES[order][0], sheet_name="heatmap_values_wide")
    hv_keys = set(hv.iloc[:, 0].map(canon))
    if hv_keys != set(m.index):
        print(f"⚠️  {BRS_FILES[order][0]} [heatmap_values_wide] differs from the per-strain top-10 view: "
              f"{len(hv_keys - set(m.index))} of its {len(hv_keys)} rows are not shown here (stale sheet; not used)")


# ==========================================
# 4) Display names
# ==========================================
id2name = pd.read_excel(NAME_MAP_PATH, sheet_name=0, dtype=str).set_index("ID")["Compound name"].str.strip().to_dict()
fig_ids = sorted({i for p in panels.values() for k in p["mat"].index for i in k.split(" | ")})
base_names = {i: DISPLAY_FIX.get(id2name.get(i, i), id2name.get(i, i)) for i in fig_ids}
name_counts = pd.Series(base_names).value_counts()


def display(met_id: str) -> str:
    nm = base_names[met_id]
    return f"{nm} [{met_id.split('_')[1]}]" if name_counts[nm] > 1 else nm


def set_label(key: str) -> str:
    ids = [t.strip() for t in cand_order.get(key, key).split("|")]
    return " + ".join(display(i) for i in ids)


for p in panels.values():
    p["labels"] = [set_label(k) for k in p["mat"].index]


# ==========================================
# 5) Text helpers (measure in inches; wrap without truncation)
# ==========================================
_MEASURE_FIG = plt.figure(dpi=72)
_RENDERER = _MEASURE_FIG.canvas.get_renderer()


def text_size_in(s: str, size: float = FS, linespacing: float = 1.0, weight: str = "normal"):
    t = _MEASURE_FIG.text(0, 0, s, fontsize=size, linespacing=linespacing, fontweight=weight)
    bb = t.get_window_extent(_RENDERER)
    t.remove()
    return bb.width / 72.0, bb.height / 72.0


def wrap_label(text: str, max_in: float, size: float = FS) -> str:
    """Greedy wrap that keeps each constituent of a combination on one line when possible."""
    parts = text.split(" + ")
    pieces = [p + " +" for p in parts[:-1]] + [parts[-1]]
    units = []
    for p in pieces:
        if text_size_in(p, size)[0] <= max_in:
            units.append(p)
        else:
            units.extend(p.split(" "))
    lines, cur = [], ""
    for u in units:
        trial = u if not cur else f"{cur} {u}"
        if text_size_in(trial, size)[0] <= max_in or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = u
    lines.append(cur)
    return "\n".join(lines)


# ==========================================
# 6) Layout (all geometry in inches)
# ==========================================
print("\n--- Step 3: Plotting ---")
col_x = np.array([0.0, CELL_PITCH, 2 * CELL_PITCH + GROUP_GAP, 3 * CELL_PITCH + GROUP_GAP])
CELLS_W = 4 * CELL_PITCH + GROUP_GAP

# Row 1: (a) singles | (b) pairs ; Row 2: legend | (c) triads (cell columns aligned with (b))
A_LABEL_W = max(text_size_in(t)[0] for t in panels["single"]["labels"]) + 0.02
A_CELL_X = 0.05 + A_LABEL_W + LABEL_PAD
B_CELL_X = FIG_W - 0.04 - CELLS_W
B_LABEL_W = B_CELL_X - LABEL_PAD - (A_CELL_X + CELLS_W + 0.28)
C_CELL_X = B_CELL_X
C_LABEL_W = 3.75

panels["single"]["wrapped"] = [wrap_label(t, A_LABEL_W) for t in panels["single"]["labels"]]
panels["pair"]["wrapped"] = [wrap_label(t, B_LABEL_W) for t in panels["pair"]["labels"]]
panels["triad"]["wrapped"] = [wrap_label(t, C_LABEL_W) for t in panels["triad"]["labels"]]


def row_height(*keys):
    n_lines = max(t.count("\n") + 1 for k in keys for t in panels[k]["wrapped"])
    return 0.175 if n_lines == 1 else 0.06 + 0.105 * n_lines


ROW_H1 = row_height("single", "pair")
ROW_H2 = row_height("triad")
HEAD_H = 0.50             # panel title + AOA/AOB bracket + strain labels
ROW_GAP = 0.20
n1 = max(len(panels["single"]["mat"]), len(panels["pair"]["mat"]))
n2 = len(panels["triad"]["mat"])
FIG_H = round(HEAD_H + n1 * ROW_H1 + ROW_GAP + HEAD_H + n2 * ROW_H2 + 0.05, 2)
ROW1_TOP = FIG_H - HEAD_H
ROW2_TOP = ROW1_TOP - n1 * ROW_H1 - ROW_GAP - HEAD_H

cmap = LinearSegmentedColormap.from_list("brs_blue", SEQ_STEPS)
norm = Normalize(vmin=VMIN, vmax=VMAX)


def rel_luminance(rgb):
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")


def fx(x_in):
    return x_in / FIG_W


def fy(y_in):
    return y_in / FIG_H


def draw_panel(order, cell_x, top, row_h, letter, title):
    p = panels[order]
    m = p["mat"]
    n = len(m)
    ax = fig.add_axes([fx(cell_x), fy(top - n * row_h), fx(CELLS_W), fy(n * row_h)])
    ax.set_xlim(0, CELLS_W)
    ax.set_ylim(n * row_h, 0)                        # y data unit = inch, first row on top
    ax.set_axis_off()
    for i, (_, r) in enumerate(m.iterrows()):
        y0 = i * row_h
        for j, s in enumerate(STRAINS):
            v = r[s]
            x0 = col_x[j]
            in_top = pd.notna(v)
            face = cmap(norm(v)) if in_top else NA_FACE
            ax.add_patch(Rectangle((x0 + CELL_GAP / 2, y0 + CELL_GAP / 2), CELL_PITCH - CELL_GAP,
                                   row_h - CELL_GAP, facecolor=face, edgecolor="none"))
            if in_top:
                ink = "white" if rel_luminance(face) < 0.30 else INK
                ax.text(x0 + CELL_PITCH / 2, y0 + row_h / 2, f"{v:.2f}", ha="center", va="center",
                        color=ink, fontsize=FS)
            else:
                ax.text(x0 + CELL_PITCH / 2, y0 + row_h / 2, EN_DASH, ha="center", va="center",
                        color=NA_INK, fontsize=FS)
        fig.text(fx(cell_x - LABEL_PAD), fy(top - y0 - row_h / 2), p["wrapped"][i], ha="right",
                 va="center", linespacing=1.0)
    # strain labels and AOA / AOB brackets
    y_lab = top + 0.03
    for j, s in enumerate(STRAINS):
        fig.text(fx(cell_x + col_x[j] + CELL_PITCH / 2), fy(y_lab), s, ha="center", va="bottom")
    y_br = y_lab + 0.155
    for grp, members in GROUPS:
        j0, j1 = STRAINS.index(members[0]), STRAINS.index(members[-1])
        x0 = cell_x + col_x[j0] + 0.03
        x1 = cell_x + col_x[j1] + CELL_PITCH - 0.03
        fig.add_artist(mpl.lines.Line2D([fx(x0), fx(x0), fx(x1), fx(x1)],
                                        [fy(y_br - 0.035), fy(y_br), fy(y_br), fy(y_br - 0.035)],
                                        color=INK_2, lw=0.6))
        fig.text(fx((x0 + x1) / 2), fy(y_br + 0.02), grp, ha="center", va="bottom", color=INK_2)
    # panel letter + title, left-aligned with the longest row label
    label_left = cell_x - LABEL_PAD - max(text_size_in(t)[0] for t in p["wrapped"])
    fig.text(fx(label_left), fy(top + 0.30), letter, fontsize=FS_PANEL, fontweight="bold",
             ha="left", va="baseline")
    lw = text_size_in(letter, FS_PANEL, weight="bold")[0]
    fig.text(fx(label_left + lw + 0.05), fy(top + 0.30), title, ha="left", va="baseline")


draw_panel("single", A_CELL_X, ROW1_TOP, ROW_H1, "(a)", "Single metabolites")
draw_panel("pair", B_CELL_X, ROW1_TOP, ROW_H1, "(b)", "Pairs")
draw_panel("triad", C_CELL_X, ROW2_TOP, ROW_H2, "(c)", "Triads")

# Legend block (bottom-left): colour bar + "not in top-10" key
LEG_X = 0.12
cb_w, cb_h = 1.55, 0.10
cb_y = ROW2_TOP - 0.42
cax = fig.add_axes([fx(LEG_X), fy(cb_y), fx(cb_w), fy(cb_h)])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.set_ticks([0.2, 0.4, 0.6, 0.8])
cb.ax.tick_params(labelsize=FS, length=2, width=0.5, pad=1.5)
cb.outline.set_linewidth(0.5)
fig.text(fx(LEG_X), fy(cb_y + cb_h + 0.05), "BRS within the strain's top-10 list", ha="left", va="bottom")
na_y = cb_y - 0.48
fig.add_artist(Rectangle((fx(LEG_X), fy(na_y)), fx(CELL_PITCH - CELL_GAP), fy(ROW_H2 - CELL_GAP),
                         facecolor=NA_FACE, edgecolor="none", transform=fig.transFigure))
fig.text(fx(LEG_X + (CELL_PITCH - CELL_GAP) / 2), fy(na_y + (ROW_H2 - CELL_GAP) / 2), EN_DASH,
         ha="center", va="center", color=NA_INK)
fig.text(fx(LEG_X + CELL_PITCH + 0.05), fy(na_y + (ROW_H2 - CELL_GAP) / 2),
         "Not in this strain's top-10 list", ha="left", va="center")


# ==========================================
# 7) Export
# ==========================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, facecolor="white")
fig.savefig(OUT_PNG, dpi=300, facecolor="white")
plt.close(fig)
plt.close(_MEASURE_FIG)
print(f"✅ Figure size: {FIG_W:.2f} x {FIG_H:.2f} in ({FIG_W * 25.4:.0f} x {FIG_H * 25.4:.0f} mm)")
print(f"✅ Saved: {OUT_PDF}")
print(f"✅ Saved: {OUT_PNG}")
for order, p in panels.items():
    m = p["mat"]
    counts = ", ".join(f"{s} {int(m[s].notna().sum())}" for s in STRAINS)
    multi = int((m[STRAINS].notna().sum(axis=1) > 1).sum())
    vals = m[STRAINS].stack()
    print(f"✅ {order}: coloured cells per strain: {counts}; sets in >1 strain's top-10: {multi}/{len(m)}; "
          f"BRS range {vals.min():.2f}-{vals.max():.2f}; union size {p['union_n']}")
