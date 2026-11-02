# -*- coding: utf-8 -*-
"""
Fig. 4 (final, Bioinformatics): single metabolites vs BRS-prioritized combinations.

Replaces the old Figs 6-8, whose AUCs applied log1p to an already z-scored matrix.
The composite score is the corrected one from Table3_mixture_candidates.py:
    raw intensity -> zeros replaced by half the column's minimum non-zero value
    -> log1p -> per-metabolite z-score -> mean within the candidate set.
Labels: high-BNI if any of NF/NV/NM/NU reaches its own 75th percentile (OR rule).

Panels
    (a) empirical ROC curves (no smoothing) of the 3 best combinations and the 3 best singles
    (b) AUC with 95% stratified-bootstrap CI for all 30 candidate sets (CI from
        supplementary/ST_mixture_candidates.xlsx, sheet all_candidates)
    (c) composite-score distributions (z-score scale) in high- vs low-BNI accessions
        for the best single and the top-3 combinations

The recomputed AUCs are asserted against ST_mixture_candidates.xlsx before plotting.

Outputs
    论文/overleaf_v6/figures/Fig4_single_vs_combo.pdf   (vector)
    论文/overleaf_v6/figures/Fig4_single_vs_combo.png   (300 dpi preview)
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, roc_auc_score

# ==========================================
# 1) Configuration
# ==========================================
RAW_PATH = r"Data Preprocessing/filtered_notfound_removed.xlsx"   # raw intensities
LABEL_PATH = r"Data Preprocessing/filtered_metabolites.xlsx"      # BNI columns used by the published labels
RESULTS = Path(os.environ.get("BRS_OUT_DIR", "results"))
CANDIDATE_PATH = (RESULTS / "candidate_sets.xlsx" if (RESULTS / "candidate_sets.xlsx").exists()
                  else Path("Top5_single_pair_triple_BRS.xlsx"))   # written by build_candidate_sets.py
CANDIDATE_SHEET = "Sheet1"
NAME_MAP_PATH = r"name.xlsx"
ST_PATH = r"supplementary/ST_mixture_candidates.xlsx"
ST_SHEET = "all_candidates"

OUT_DIR = Path("论文/overleaf_v6/figures")
OUT_PDF = OUT_DIR / "Fig4_single_vs_combo.pdf"
OUT_PNG = OUT_DIR / "Fig4_single_vs_combo.png"

STRAINS = ["NF", "NV", "NM", "NU"]
MAIN_Q = 0.75
N_ROC_EACH = 3            # best combinations / best singles shown in (a)
N_BOX_COMBOS = 3          # combinations shown next to the best single in (c)
AUC_TOL = 6e-5            # ST sheet is rounded to 4 decimals
EXPECTED = {              # sanity values quoted in the manuscript
    "N-Acetyl-DL-glutamic acid": 0.816,
    "Uric acid + N-Acetyl-DL-glutamic acid + L-Phenylalanine": 0.911,
    "Uric acid + N-Acetyl-DL-glutamic acid + L-Tryptophan": 0.905,
}
JITTER_SEED = 7

DISPLAY_FIX = {"L-TRYPTOPHAN": "L-Tryptophan", "Uric Acid": "Uric acid", "Ala-ala": "Ala-Ala",
               "Isoshaftoside": "Isoschaftoside", "4-METHYLCATECHOL": "4-Methylcatechol"}

# Order colours (Okabe-Ito; validated for CVD, and light / mid / dark in greyscale)
ORDER_COLOR = {"single": "#E69F00", "pair": "#CC79A7", "triad": "#0072B2"}
ORDER_MARKER = {"single": "o", "pair": "s", "triad": "D"}
ORDER_LABEL = {"single": "Single metabolite", "pair": "Pair", "triad": "Triad"}
# Within-order shades for the ROC curves (rank 1 = darkest)
ROC_SHADES = {"single": ["#8C5E00", "#E69F00", "#F0C35A"],
              "pair": ["#8E3F6C", "#CC79A7", "#E3B2CD"],
              "triad": ["#003C63", "#0072B2", "#5BA7D9"]}
SINGLE_DASHES = [(5.0, 1.6), (2.4, 1.4), (1.0, 1.2)]

INK = "#1A1A1A"
INK_2 = "#555555"
GRID = "#E3E3E3"
FS = 7.0                  # minimum font size at final size (pt)
FS_PANEL = 9.0
FIG_W = 7.0               # double-column width (in)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": FS,
    "axes.labelsize": FS,
    "axes.titlesize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "axes.edgecolor": INK_2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ==========================================
# 2) Data and labels (same steps as Table3_mixture_candidates.py)
# ==========================================
def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def half_min_replace(s: pd.Series) -> pd.Series:
    nz = s[(s != 0) & s.notna()]
    if (s == 0).any() and len(nz):
        s = s.copy()
        s[s == 0] = nz.min() / 2.0
    return s


def bni_labels(bni: pd.DataFrame, q: float) -> np.ndarray:
    """OR rule: high-BNI if any strain reaches its own q-quantile."""
    thr = bni.quantile(q)
    return (bni >= thr).any(axis=1).astype(int).values


print("--- Step 1: Loading raw intensities and BNI labels ---")
raw = to_numeric(pd.read_excel(RAW_PATH, sheet_name=0, decimal=","))
df_lab = pd.read_excel(LABEL_PATH, sheet_name=0)
bni_cols = {s: [c for c in df_lab.columns if "Inhibition" in str(c) and s in str(c)][0] for s in STRAINS}
bni = df_lab[[bni_cols[s] for s in STRAINS]].set_axis(STRAINS, axis=1)
met_cols = [c for c in df_lab.columns if c not in bni_cols.values()]

X_log = np.log1p(raw[met_cols].apply(half_min_replace))
Z = (X_log - X_log.mean()) / X_log.std()
Z.columns = [str(c).strip() for c in Z.columns]
y = bni_labels(bni, MAIN_Q)
n_hi, n_lo = int(y.sum()), int(len(y) - y.sum())
print(f"✅ {Z.shape[0]} accessions x {Z.shape[1]} metabolites; high-BNI = {n_hi} / {len(y)} at q = {MAIN_Q}")


# ==========================================
# 3) Candidate sets and display names
# ==========================================
print("\n--- Step 2: Parsing BRS candidate sets ---")
id2name = pd.read_excel(NAME_MAP_PATH, sheet_name=0, dtype=str).set_index("ID")["Compound name"].str.strip().to_dict()
cand = pd.read_excel(CANDIDATE_PATH, sheet_name=CANDIDATE_SHEET)

all_ids = sorted({f.strip() for v in cand.iloc[:, 0] for f in str(v).split("|")})
base_names = {i: DISPLAY_FIX.get(id2name.get(i, i), id2name.get(i, i)) for i in all_ids}
name_counts = pd.Series(base_names).value_counts()


def display(met_id: str) -> str:
    nm = base_names[met_id]
    return f"{nm} [{met_id.split('_')[1]}]" if name_counts[nm] > 1 else nm


candidates = []
for _, row in cand.iterrows():
    ids = [f.strip() for f in str(row.iloc[0]).split("|")]
    missing = [i for i in ids if i not in Z.columns]
    if missing:
        raise KeyError(f"Metabolites not found in the raw matrix: {missing}")
    candidates.append({"ids": ids, "key": " | ".join(ids),
                       "candidate": " + ".join(display(i) for i in ids),
                       "order": {1: "single", 2: "pair", 3: "triad"}[len(ids)]})
print(f"✅ {len(candidates)} candidate sets")


# ==========================================
# 4) Composite scores and AUC; check against ST_mixture_candidates.xlsx
# ==========================================
print("\n--- Step 3: Composite scores, AUC and consistency check ---")
for c in candidates:
    c["score"] = Z[c["ids"]].mean(axis=1).values
    c["AUC"] = float(roc_auc_score(y, c["score"]))
    r = rankdata(c["score"])                                   # Mann-Whitney cross-check
    c["AUC_rank"] = float((r[y == 1].sum() - n_hi * (n_hi + 1) / 2) / (n_hi * n_lo))
    assert abs(c["AUC"] - c["AUC_rank"]) < 1e-12

st = pd.read_excel(ST_PATH, sheet_name=ST_SHEET)
st["key"] = st["metabolite_ids"].astype(str).str.strip()
st = st.set_index("key")
if set(st.index) != {c["key"] for c in candidates}:
    raise ValueError("Candidate sets in ST_mixture_candidates.xlsx differ from Top5_single_pair_triple_BRS.xlsx")

max_diff = 0.0
for c in candidates:
    s = st.loc[c["key"]]
    if s["candidate"] != c["candidate"] or s["order"] != c["order"]:
        raise ValueError(f"Display name / order mismatch: {c['candidate']!r} vs {s['candidate']!r}")
    d = abs(c["AUC"] - float(s["AUC"]))
    max_diff = max(max_diff, d)
    assert d < AUC_TOL, f"AUC mismatch for {c['candidate']}: {c['AUC']:.4f} vs ST {s['AUC']:.4f}"
    c["CI_low"], c["CI_high"] = float(s["AUC_CI_low"]), float(s["AUC_CI_high"])
    c["p_FWER"] = float(s["p_perm_FWER"])
for name, val in EXPECTED.items():   # spot checks; skipped for sets absent from this run's candidates
    got = next((c["AUC"] for c in candidates if c["candidate"] == name), None)
    if got is not None:
        assert round(got, 3) == val, f"{name}: expected {val}, got {got:.4f}"
print(f"✅ All {len(candidates)} recomputed AUCs match {ST_PATH} (max |diff| = {max_diff:.1e})")

res = pd.DataFrame(candidates).sort_values("AUC", ascending=False).reset_index(drop=True)
singles = res[res["order"] == "single"].reset_index(drop=True)
combos = res[res["order"] != "single"].reset_index(drop=True)
best_single = singles.iloc[0]
print(f"✅ Best single: {best_single['candidate']} AUC = {best_single['AUC']:.3f}")
for _, r in combos.head(N_ROC_EACH).iterrows():
    print(f"✅ Combination: {r['candidate']} AUC = {r['AUC']:.3f} [{r['CI_low']:.2f}, {r['CI_high']:.2f}]")


# ==========================================
# 5) Text helpers (measure in inches; wrap without truncation)
# ==========================================
NBSP = " "           # keeps "(AUC 0.911)" on one line
_MEASURE_FIG = plt.figure(dpi=72)
_RENDERER = _MEASURE_FIG.canvas.get_renderer()


def text_size_in(s: str, size: float = FS, linespacing: float = 1.0):
    t = _MEASURE_FIG.text(0, 0, s, fontsize=size, linespacing=linespacing)
    bb = t.get_window_extent(_RENDERER)
    t.remove()
    return bb.width / 72.0, bb.height / 72.0


def text_width_in(s: str, size: float = FS) -> float:
    return text_size_in(s, size)[0]


def wrap_label(text: str, max_in: float, size: float = FS) -> str:
    """Greedy wrap that keeps each constituent of a combination on one line when possible."""
    parts = text.split(" + ")
    pieces = [p + " +" for p in parts[:-1]] + [parts[-1]]
    units = []
    for p in pieces:                                   # split a constituent only if it alone is too wide
        if text_width_in(p, size) <= max_in:
            units.append(p)
        else:
            units.extend(p.split(" "))
    lines, cur = [], ""
    for u in units:
        trial = u if not cur else f"{cur} {u}"
        if text_width_in(trial, size) <= max_in or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = u
    lines.append(cur)
    return "\n".join(lines)


def with_auc(r) -> str:
    return f"{r['candidate']} (AUC{NBSP}{r['AUC']:.3f})"


def frac(fig_h, x_in, y_in):
    return x_in / FIG_W, y_in / fig_h


def add_axes_in(fig, left, bottom, width, height, fig_h):
    """Axes placed in inches from the bottom-left corner."""
    return fig.add_axes([left / FIG_W, bottom / fig_h, width / FIG_W, height / fig_h])


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


# ==========================================
# 6) Layout (all geometry in inches)
# ==========================================
print("\n--- Step 4: Plotting ---")
LEFT_COL_R = 2.62         # right edge of the left column (a, c)

# (a) ROC square + legend underneath
A_LEFT, A_SIZE, A_TOP_PAD = 0.42, 1.80, 0.22
A_XLAB_H = 0.40           # tick labels + axis title below (a)
LEG_W = LEFT_COL_R - 0.05 - 0.36       # text width after the legend handle
roc_rows = []
for k, (_, r) in enumerate(combos.head(N_ROC_EACH).iterrows()):
    roc_rows.append((r, ROC_SHADES[r["order"]][k], "-"))
for k, (_, r) in enumerate(singles.head(N_ROC_EACH).iterrows()):
    roc_rows.append((r, ROC_SHADES["single"][k], (0, SINGLE_DASHES[k])))
leg_labels = [wrap_label(with_auc(r), LEG_W) for r, _, _ in roc_rows]
LEG_SPACING = 0.40        # in font-size units
leg_h = (sum(text_size_in(t, linespacing=1.05)[1] for t in leg_labels)
         + LEG_SPACING * FS / 72 * (len(leg_labels) - 1))

# (c) one block per candidate: wrapped full-name title, then high- and low-BNI boxes
box_rows = pd.concat([singles.head(1), combos.head(N_BOX_COMBOS)], ignore_index=True)
C_AX_LEFT = 0.40
C_TITLE_W = LEFT_COL_R - 0.05
C_GRP_PITCH, C_BOX_H, C_TITLE_GAP, C_BLOCK_GAP = 0.155, 0.115, 0.035, 0.085
c_titles = [wrap_label(with_auc(r), C_TITLE_W) for _, r in box_rows.iterrows()]
c_title_h = [text_size_in(t)[1] for t in c_titles]
c_block_h = [h + C_TITLE_GAP + 2 * C_GRP_PITCH for h in c_title_h]
C_H = sum(c_block_h) + C_BLOCK_GAP * (len(box_rows) - 1)
C_LEG_H = 0.16            # filled/open legend row above (c)
C_BOTTOM = 0.40

left_stack = A_TOP_PAD + A_SIZE + A_XLAB_H + leg_h + 0.30 + C_LEG_H + 0.06 + C_H + C_BOTTOM

# (b) forest rows: variable pitch so that wrapped labels never touch
B_LABEL_W = 2.45
B_AX_W = 1.50
B_AX_LEFT = FIG_W - B_AX_W - 0.08
B_ROW, B_EXTRA_LINE = 0.150, 0.100
B_BOTTOM, B_TOP_PAD = 0.40, 0.42
res["b_label"] = [wrap_label(t, B_LABEL_W) for t in res["candidate"]]
res["b_lines"] = res["b_label"].str.count("\n") + 1
pitch = (B_ROW + B_EXTRA_LINE * (res["b_lines"] - 1)).values
FIG_H = round(max(left_stack, B_BOTTOM + pitch.sum() + B_TOP_PAD), 2)
pitch = pitch * (FIG_H - B_BOTTOM - B_TOP_PAD) / pitch.sum()      # fill the column height
b_height = float(pitch.sum())

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

# ------------------------------------------------------------------
# (a) Empirical ROC curves (no smoothing; ties drawn as diagonal segments)
# ------------------------------------------------------------------
A_BOTTOM = FIG_H - A_TOP_PAD - A_SIZE
axA = add_axes_in(fig, A_LEFT, A_BOTTOM, A_SIZE, A_SIZE, FIG_H)
style_axes(axA)
axA.plot([0, 1], [0, 1], ls=(0, (1, 1.5)), lw=0.7, color="#9A9A9A", zorder=1)
handles = []
for r, col, ls in roc_rows:
    fpr, tpr, _ = roc_curve(y, r["score"], drop_intermediate=False)
    axA.plot(fpr, tpr, color=col, lw=1.25, ls=ls, solid_joinstyle="miter", zorder=3)
    handles.append(Line2D([0], [0], color=col, lw=1.25, ls=ls))
ticks = [0, 0.25, 0.5, 0.75, 1.0]
tick_lbl = ["0", "0.25", "0.5", "0.75", "1"]
axA.set_xlim(-0.01, 1.01)
axA.set_ylim(-0.01, 1.01)
axA.set_aspect("equal")
axA.set_xticks(ticks)
axA.set_yticks(ticks)
axA.set_xticklabels(tick_lbl)
axA.set_yticklabels(tick_lbl)
axA.set_xlabel("False-positive rate (1 − specificity)", labelpad=2)
axA.set_ylabel("True-positive rate (sensitivity)", labelpad=2)
axA.grid(True, color=GRID, lw=0.5, zorder=0)

leg_top = A_BOTTOM - A_XLAB_H
leg = fig.legend(handles, leg_labels, loc="upper left", bbox_to_anchor=frac(FIG_H, 0.02, leg_top),
                 frameon=False, handlelength=2.6, handletextpad=0.5, labelspacing=LEG_SPACING,
                 borderaxespad=0, borderpad=0)
for t in leg.get_texts():
    t.set_linespacing(1.05)

# ------------------------------------------------------------------
# (b) Forest plot of AUC with 95% bootstrap CI (all 30 candidate sets)
# ------------------------------------------------------------------
axB = add_axes_in(fig, B_AX_LEFT, B_BOTTOM, B_AX_W, b_height, FIG_H)
style_axes(axB)
axB.spines["left"].set_visible(False)
y_edges = np.concatenate([[0.0], np.cumsum(pitch)])
y_mid = b_height - (y_edges[:-1] + y_edges[1:]) / 2.0     # top row = highest AUC
for yy, (_, r) in zip(y_mid, res.iterrows()):
    col = ORDER_COLOR[r["order"]]
    axB.plot([r["CI_low"], r["CI_high"]], [yy, yy], color=col, lw=1.0, solid_capstyle="butt", zorder=3)
    axB.scatter([r["AUC"]], [yy], s=15, marker=ORDER_MARKER[r["order"]], color=col,
                edgecolor="white", linewidth=0.5, zorder=4)
axB.set_ylim(0, b_height)
axB.set_yticks(y_mid)
axB.set_yticklabels(res["b_label"], linespacing=1.0)
axB.tick_params(axis="y", length=0, pad=3)
for t in axB.get_yticklabels():
    t.set_va("center")
axB.set_xlim(0.30, 1.0)
axB.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axB.set_xticklabels(["0.3", "", "0.5", "", "0.7", "", "0.9", ""])
axB.set_xlabel("AUC (95% bootstrap CI)", labelpad=2)
axB.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
axB.axvline(0.5, color="#7A7A7A", lw=0.8, ls=(0, (1, 1.5)), zorder=1)
axB.axvline(best_single["AUC"], color=ORDER_COLOR["single"], lw=0.9, ls=(0, (4, 2)), zorder=2)
axB.text(0.5, b_height + 0.04, "chance", ha="center", va="bottom", color=INK_2, clip_on=False)
axB.text(best_single["AUC"], b_height + 0.04, f"best single\n({best_single['AUC']:.3f})", ha="center",
         va="bottom", color=INK_2, linespacing=1.0, clip_on=False)
order_handles = [Line2D([0], [0], color=ORDER_COLOR[o], lw=1.0, marker=ORDER_MARKER[o], markersize=4,
                        markeredgecolor="white", markeredgewidth=0.5) for o in ORDER_COLOR]
fig.legend(order_handles, [ORDER_LABEL[o] for o in ORDER_COLOR], loc="lower left", ncol=3,
           bbox_to_anchor=frac(FIG_H, B_AX_LEFT - B_LABEL_W, FIG_H - 0.27),
           frameon=False, handlelength=1.8, handletextpad=0.4, columnspacing=1.2,
           borderaxespad=0, borderpad=0)

# ------------------------------------------------------------------
# (c) Composite-score distributions, high- vs low-BNI (z-score scale, linear axis)
# ------------------------------------------------------------------
axC = add_axes_in(fig, C_AX_LEFT, C_BOTTOM, LEFT_COL_R - C_AX_LEFT, C_H, FIG_H)
style_axes(axC)
axC.spines["left"].set_visible(False)
axC.set_ylim(0, C_H)                                   # y data unit = inch
rng = np.random.default_rng(JITTER_SEED)
y_ticks, y_ticklabels = [], []
top = C_H
for k, (_, r) in enumerate(box_rows.iterrows()):
    col = ORDER_COLOR[r["order"]]
    fig.text(*frac(FIG_H, 0.05, C_BOTTOM + top), c_titles[k], ha="left", va="top", linespacing=1.0)
    y_hi = top - c_title_h[k] - C_TITLE_GAP - C_GRP_PITCH / 2
    for grp, yy in ((1, y_hi), (0, y_hi - C_GRP_PITCH)):
        vals = r["score"][y == grp]
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lo = vals[vals >= q1 - 1.5 * iqr].min()
        hi = vals[vals <= q3 + 1.5 * iqr].max()
        face = mpl.colors.to_rgba(col, 0.22) if grp == 1 else "white"
        axC.add_patch(mpl.patches.Rectangle((q1, yy - C_BOX_H / 2), q3 - q1, C_BOX_H, facecolor=face,
                                            edgecolor=col, lw=0.8, zorder=2))
        axC.plot([med, med], [yy - C_BOX_H / 2, yy + C_BOX_H / 2], color=col, lw=1.4, zorder=3,
                 solid_capstyle="butt")
        axC.plot([lo, q1], [yy, yy], color=col, lw=0.7, zorder=2)
        axC.plot([q3, hi], [yy, yy], color=col, lw=0.7, zorder=2)
        jit = rng.uniform(-C_BOX_H * 0.42, C_BOX_H * 0.42, size=len(vals))
        if grp == 1:
            axC.scatter(vals, yy + jit, s=6, color=col, edgecolor="none", alpha=0.9, zorder=4)
        else:
            axC.scatter(vals, yy + jit, s=6, facecolor="white", edgecolor=col, linewidth=0.6, zorder=4)
        y_ticks.append(yy)
        y_ticklabels.append("High" if grp == 1 else "Low")
    top -= c_block_h[k] + C_BLOCK_GAP
axC.set_yticks(y_ticks)
axC.set_yticklabels(y_ticklabels)
axC.tick_params(axis="y", length=0, pad=3)
all_scores = np.concatenate(box_rows["score"].values)
axC.set_xlim(np.floor((all_scores.min() - 0.15) * 2) / 2, np.ceil((all_scores.max() + 0.15) * 2) / 2)
axC.axvline(0, color="#9A9A9A", lw=0.6, zorder=1)
axC.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
axC.set_xlabel("Composite score (mean z-score)", labelpad=2)
axC.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
grp_handles = [Line2D([0], [0], ls="none", marker="o", markersize=3.5, color=INK_2, markeredgecolor=INK_2),
               Line2D([0], [0], ls="none", marker="o", markersize=3.5, markerfacecolor="white",
                      markeredgecolor=INK_2, markeredgewidth=0.7)]
c_leg_y = C_BOTTOM + C_H + 0.06
fig.legend(grp_handles, [f"High-BNI accessions (n = {n_hi})", f"Low-BNI (n = {n_lo})"], loc="lower left",
           ncol=2, bbox_to_anchor=frac(FIG_H, 0.30, c_leg_y), frameon=False, handlelength=1.0,
           handletextpad=0.3, columnspacing=1.0, borderaxespad=0, borderpad=0)

# Panel letters
for x_in, y_in, letter in [(0.02, FIG_H - 0.05, "(a)"),
                           (B_AX_LEFT - B_LABEL_W - 0.14, FIG_H - 0.05, "(b)"),
                           (0.02, c_leg_y + 0.16, "(c)")]:
    fig.text(*frac(FIG_H, x_in, y_in), letter, fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")


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
print(f"✅ AUC range over the 30 sets: {res['AUC'].min():.3f}-{res['AUC'].max():.3f}; "
      f"singles {singles['AUC'].min():.3f}-{singles['AUC'].max():.3f}; "
      f"combinations {combos['AUC'].min():.3f}-{combos['AUC'].max():.3f}; "
      f"combinations above the best single: {(combos['AUC'] > best_single['AUC']).sum()} / {len(combos)}")
