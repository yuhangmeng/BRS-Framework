from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
# -*- coding: utf-8 -*-
# Two-metabolite combination scoring (Pairwise BRS) + overview heatmap
# Requirements implemented:
#   1) Map metabolite IDs to names using an external name-mapping Excel file.
#   2) Heatmap shows only Top 10 pairs (by mean BRS across cases).
#   3) All comments are in English.

import os, re, time, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import r2_score
import lightgbm as lgb
import shap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ================== Configuration ==================
EXCEL_PATH   = str(_ROOT / "data" / "preprocessing" / "filtered_metabolites.xlsx")
SHEET_NAME   = "Sheet1"

# Regression targets (cases)
TARGET_COLS  = ["Inhibition (%)-NF", "Inhibition (%)-NV", "Inhibition (%)-NM", "Inhibition (%)-NU"]
TARGET_COLS1 = ["NF", "NV", "NM", "NU"]  # kept for compatibility (not used below)

# Univariate results used to select candidate features for pairing
UNIVARIATE_XLSX = str(_ROOT / "results" / "univariate_BRS.xlsx")

OUT_DIR  = str(_ROOT / "results")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_XLSX = os.path.join(OUT_DIR, "pairwise_BRS.xlsx")
OUT_HEAT = os.path.join(OUT_DIR, "heatmap_PAIR_BRS_overview_top10_named.png")

# Name mapping table (external file): ID -> Compound name
NAME_MAP_PATH   = str(_ROOT / "data" / "name.xlsx")
NAME_MAP_SHEET  = "Sheet1"          # can be 0
ID_COL_IN_MAP   = "ID"
NAME_COL_IN_MAP = "Compound name"  # if your column is "Name", change here

FOLDS, SEED = 5, 42

# Strength weights (three-model ensemble)
W_LGBM, W_ENET, W_MLP = 0.1, 0.5, 0.4

# BRS weights
ALPHA, BETA, GAMMA = 0.2, 0.2, 0.6

# Consistency gating threshold parameter
CONS_Q = 0.5

# Visualization marker threshold
MARK_THRESH = 0.6

# TopK (for per-case Excel output)
TOPK = 10

# Speed / scale controls
PAIR_TOPK_FROM_UNI   = TOPK   # candidates per case taken from univariate TopK
PAIR_GLOBAL_CAP      = 20     # global cap on candidate features (union then truncate)
MAX_PAIRS_PER_CASE   = 1000   # max number of pairs evaluated per case
PAIR_ADD_INTERACTION = True   # include interaction term x1*x2
MLP2_EPOCHS          = 80
MLP2_HIDDEN          = 32
ENET2_BOOT           = 40
LGBM2_ESTIMATORS     = 400
LGBM2_EARLY_STOP     = 50

# Heatmap shows only Top N pairs
HEAT_TOPN = 10

# LightGBM base parameters
LGBM_SAFE = dict(
    boosting_type="gbdt", objective="regression",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=15, max_depth=-1,
    subsample=0.8, colsample_bytree=0.7, subsample_freq=1,
    min_data_in_leaf=1, min_data_in_bin=1, min_gain_to_split=0.0,
    reg_alpha=0.0, reg_lambda=0.0, force_row_wise=True,
    verbosity=-1, random_state=SEED, n_jobs=-1,
)

# ================== Utility functions ==================
def coerce_numeric(s: pd.Series) -> pd.Series:
    """Convert mixed-format numeric series to float; unparseable entries become NaN."""
    if s.dtype.kind in "biufc":
        return s.astype(float)
    x = s.astype(str).str.strip()
    x = x.str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)
    x = x.str.replace(",", ".", regex=False)
    x = x.str.replace(r"[^0-9.\-eE]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")

def minmax_01(v, eps=1e-12):
    """Min-max normalize to [0,1]. If degenerate, return zeros."""
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)

def median_iqr_gate(a: np.ndarray, q=0.5):
    """Threshold = median + q * IQR (used to compute 'hit' indicators)."""
    a = np.asarray(a, float)
    med = np.nanmedian(a)
    q1, q3 = np.nanpercentile(a, 25), np.nanpercentile(a, 75)
    iqr = q3 - q1
    return med + q * iqr

def safe_title(s: str) -> str:
    """Make a safe sheet name prefix."""
    return str(s).replace("%", "pct").replace("/", "_")

def plot_heat(mat_df, title, out_png, mark_thresh=None):
    """Plot heatmap: rows=pairs, cols=cases, values=BRS (0-1)."""
    # Dynamically scale figure height: 0.45 inch per row, minimum 6 inches
    H = max(6, 0.45 * len(mat_df))
    fig, ax = plt.subplots(figsize=(8.2, H))

    im = ax.imshow(mat_df.values, aspect="auto", vmin=0, vmax=1, cmap="coolwarm")

    # Axis labels:
    # Y-axis: keep " | " separator to improve readability
    ylabels = [re.sub(r"\s*\|\s*", " | ", str(s)) for s in mat_df.index]
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9)

    # X-axis: rotate slightly and right-align
    ax.set_xticks(np.arange(mat_df.shape[1]))
    ax.set_xticklabels(list(mat_df.columns), rotation=20, ha="right", fontsize=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_ylabel("Score (0–1)", rotation=90, va="center")

    # Title
    ax.set_title(title, fontsize=13, pad=14)

    # Expand left margin based on label length to avoid clipping
    maxlen = max(len(str(s)) for s in mat_df.index) if len(mat_df.index) else 0
    left = min(0.62, 0.20 + 0.006 * maxlen)
    fig.subplots_adjust(left=left, right=0.96, top=0.92, bottom=0.10)

    # Optional marking for values above a threshold
    if mark_thresh is not None:
        arr = mat_df.values
        ys, xs = np.where(arr >= mark_thresh)
        for y, x in zip(ys, xs):
            ax.text(x, y, "○", ha="center", va="center", fontsize=9)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def load_id2name_map(map_path, sheet_name, id_col="ID", name_col="Compound name"):
    """Load ID->Name mapping from an external Excel file."""
    df = pd.read_excel(map_path, sheet_name=sheet_name)
    if id_col not in df.columns:
        raise KeyError(f"Missing column in name map: {id_col}")
    if name_col not in df.columns:
        raise KeyError(f"Missing column in name map: {name_col}")

    df = df[[id_col, name_col]].copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df[name_col] = df[name_col].astype(str).str.strip()
    return dict(zip(df[id_col], df[name_col]))

def short_id_tag(met_id: str) -> str:
    """Extract a short tag from a rowID-like ID (e.g., rowID_1952_... -> 1952)."""
    parts = str(met_id).split("_")
    return parts[1] if len(parts) >= 2 else str(met_id)[:8]

# LightGBM safe feature names (fixes: "Do not support special JSON characters in feature name.")
def sanitize_colname(name: str) -> str:
    s = str(name)
    s = re.sub(r'[\\\"/\t\n\r\b\f\[\]\{\}:,|]', "_", s)
    s = s.replace("(", "_").replace(")", "_").replace("=", "_").replace("+", "_").replace("*", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "col"

def make_unique(names):
    """Make sanitized feature names unique."""
    used, out = set(), []
    for n in names:
        base = sanitize_colname(n) or "col"
        new = base
        k = 1
        while new in used:
            k += 1
            new = f"{base}_{k}"
        used.add(new)
        out.append(new)
    return out

# ================== Load data ==================
raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, decimal=",").dropna(how="all")

for t in TARGET_COLS:
    if t not in raw.columns:
        raise ValueError(f"Target column not found: {t}")

# Feature columns are all non-target columns
feature_cols = [c for c in raw.columns if c not in TARGET_COLS]
X_num = pd.DataFrame({c: coerce_numeric(raw[c]) for c in feature_cols})
feat_cols = [c for c in X_num.columns if X_num[c].notna().any()]
X_all = X_num[feat_cols].copy()

# ================== Load TopK candidates from univariate results ==================
def load_univariate_topk(univar_xlsx_path: str, target_cols, topk: int):
    """Read per-case TopK univariate features from Excel outputs."""
    xls = pd.ExcelFile(univar_xlsx_path)
    sheets = xls.sheet_names
    mapping = {}
    for case in target_cols:
        prefix = safe_title(case)[:28]
        candidates = [s for s in sheets if s.startswith(prefix) and f"_Top30" in s]
        if not candidates:
            raise ValueError(
                f"Cannot find Top{topk} sheet for case '{case}' in {univar_xlsx_path} (prefix={prefix})"
            )
        df = pd.read_excel(univar_xlsx_path, sheet_name=candidates[0])
        if "Feature" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Feature"})
        mapping[case] = df["Feature"].astype(str).head(topk).tolist()
    return mapping

uni_topk_map = load_univariate_topk(UNIVARIATE_XLSX, TARGET_COLS, TOPK)

# Union of per-case candidates, then truncate to a global cap
uni_top_union = set()
for case in TARGET_COLS:
    uni_top_union.update(uni_topk_map[case])
uni_top = list(uni_top_union)[:PAIR_GLOBAL_CAP]
print(f"[PAIR] Candidate features: {len(uni_top)} (union of univariate Top{TOPK} across cases, capped at {PAIR_GLOBAL_CAP})")

# ================== Pairwise scoring core ==================
def _make_pair_matrix(X_df: pd.DataFrame, f1: str, f2: str, add_inter=False):
    """Build pairwise feature matrix [x1, x2, (x1*x2)] and standardize columns."""
    x1 = X_df[f1].values.reshape(-1, 1)
    x2 = X_df[f2].values.reshape(-1, 1)
    cols = [f1, f2]
    Xp = np.hstack([x1, x2])
    if add_inter:
        Xp = np.hstack([Xp, (x1 * x2)])
        cols += [f"{f1}*{f2}"]

    Xp = (Xp - np.nanmean(Xp, axis=0)) / (np.nanstd(Xp, axis=0) + 1e-12)
    return Xp, cols

def _oof_r2_linear(Xpair: np.ndarray, yarr: np.ndarray, folds=FOLDS):
    """Out-of-fold R^2 using linear regression (fast validity proxy)."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros_like(yarr, dtype=float)
    for tr, va in kf.split(Xpair):
        lr = LinearRegression()
        lr.fit(Xpair[tr], yarr[tr])
        oof[va] = lr.predict(Xpair[va])
    return max(0.0, float(r2_score(yarr, oof)))

class TinyPairMLP(nn.Module):
    """Lightweight pairwise MLP with a gating (attention-like) mechanism."""
    def __init__(self, d, h=MLP2_HIDDEN):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.reg  = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x):
        a = self.gate(x)
        y = self.reg(x * a)
        return y, a

def pair_scores_for_case(case_name: str, X_use: pd.DataFrame, y_use: pd.Series,
                         uni_candidates: list, max_pairs=MAX_PAIRS_PER_CASE):
    """Compute pairwise BRS scores for a single case."""
    feats = [f for f in uni_candidates if f in X_use.columns]
    all_pairs = list(combinations(feats, 2))

    # If too many pairs, prioritize a structured subset then randomly fill the rest
    if len(all_pairs) > max_pairs:
        favored = []
        for i, f1 in enumerate(feats[:min(len(feats), 2 * int(np.sqrt(max_pairs) + 1))]):
            sl = feats[i + 1: i + 1 + min(20, len(feats) - i - 1)]
            favored.extend([tuple(sorted((f1, f2))) for f2 in sl])
        favored = list(dict.fromkeys(favored))
        remain = [p for p in all_pairs if p not in favored]
        rng = np.random.RandomState(SEED)
        rng.shuffle(remain)
        all_pairs = (favored + remain)[:max_pairs]

    print(f"[PAIR {case_name}] Pairs evaluated: {len(all_pairs)}")
    yarr = y_use.values.astype(float)
    records = []

    # Estimate best univariate OOF R^2 as a baseline for validity improvement
    best_uni_r2 = 0.0
    for f in feats[:min(len(feats), 100)]:
        x = X_use[f].values.reshape(-1, 1)
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
        uni_r2 = _oof_r2_linear(x, yarr, folds=FOLDS)
        best_uni_r2 = max(best_uni_r2, uni_r2)

    for idx, (f1, f2) in enumerate(all_pairs, 1):
        Xp, cols = _make_pair_matrix(X_use, f1, f2, add_inter=PAIR_ADD_INTERACTION)

        # (A) ElasticNet stability selection via bootstrapping
        sel_cnt = 0
        rng = np.random.RandomState(SEED)
        for b in range(ENET2_BOOT):
            n = max(3, int(len(yarr) * 0.7))
            sub = rng.choice(len(yarr), size=n, replace=False)
            Xb, yb = Xp[sub], yarr[sub]
            keep_mask = np.isfinite(Xb).all(axis=0)
            if not np.any(keep_mask):
                continue
            model = ElasticNetCV(l1_ratio=[0.7, 0.85, 1.0], cv=3, random_state=SEED + b, max_iter=2000)
            model.fit(Xb[:, keep_mask], yb)
            if np.any(np.abs(model.coef_) > 1e-10):
                sel_cnt += 1
        enet_freq = sel_cnt / max(ENET2_BOOT, 1)

        # (B) LightGBM + SHAP (use safe column names)
        lgbm_score = 0.0
        try:
            safe_cols = make_unique(cols)
            X_df_local = pd.DataFrame(Xp, columns=safe_cols)
            med = X_df_local.median(axis=0, skipna=True)
            X_df_local = X_df_local.fillna(med)

            kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
            shap_sum = np.zeros(X_df_local.shape[1], dtype=float)
            eff = 0

            for tr, va in kf.split(X_df_local):
                model = lgb.LGBMRegressor(**{
                    **LGBM_SAFE,
                    "n_estimators": LGBM2_ESTIMATORS,
                    "learning_rate": 0.05,
                    "num_leaves": 7,
                    "min_data_in_leaf": 3
                })
                model.fit(
                    X_df_local.iloc[tr], y_use.iloc[tr],
                    eval_set=[(X_df_local.iloc[va], y_use.iloc[va])],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(LGBM2_EARLY_STOP, verbose=False)]
                )
                expl = shap.TreeExplainer(model)
                sv = expl.shap_values(X_df_local.iloc[va])
                if isinstance(sv, list):
                    sv = sv[0]
                shap_mean = np.mean(np.abs(sv), axis=0)
                shap_sum += shap_mean
                eff += 1

            if eff > 0:
                lgbm_score = float(np.mean(shap_sum / eff))
        except Exception:
            # Any exception yields a 0 score for robustness
            pass

        # (C) Pairwise MLP attention score (lightweight)
        d = Xp.shape[1]
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        att_coll = []
        for tr, va in kf.split(Xp):
            Xtr = torch.tensor(Xp[tr], dtype=torch.float32)
            ytr = torch.tensor(yarr[tr], dtype=torch.float32).view(-1, 1)
            Xva = torch.tensor(Xp[va], dtype=torch.float32)

            model = TinyPairMLP(d)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            for _ in range(MLP2_EPOCHS):
                opt.zero_grad()
                pred, _ = model(Xtr)
                loss = loss_fn(pred, ytr)
                loss.backward()
                opt.step()

            with torch.no_grad():
                _, a = model(Xva)
                att_coll.append(a.mean(0).numpy())

        att = np.mean(att_coll, axis=0)
        mlp_pair = float(np.mean(minmax_01(np.clip(att, 0, None))))

        # Validity: pair OOF R^2 improvement over the best univariate baseline (>=0)
        pair_r2 = _oof_r2_linear(Xp, yarr, folds=FOLDS)
        validity_raw = max(0.0, pair_r2 - best_uni_r2)

        records.append((f1, f2, mlp_pair, lgbm_score, enet_freq, validity_raw))

        if (idx % 500) == 0:
            print(f"[PAIR {case_name}] Progress {idx}/{len(all_pairs)}")

    # Aggregate -> normalize -> compute BRS
    dfp = pd.DataFrame(records, columns=["f1", "f2", "MLP_raw", "LGBM_raw", "ENet_raw", "Validity_raw"])

    for col in ["MLP_raw", "LGBM_raw", "ENet_raw", "Validity_raw"]:
        dfp[col.replace("_raw", "_01")] = minmax_01(dfp[col].values)

    for col in ["MLP_01", "LGBM_01", "ENet_01"]:
        gate = median_iqr_gate(dfp[col].values, q=CONS_Q)
        dfp[col.replace("_01", "_hit")] = (dfp[col] >= gate).astype(int)

    dfp["Consistency"] = (dfp["MLP_hit"] + dfp["LGBM_hit"] + dfp["ENet_hit"]) / 3.0

    dfp["Strength"] = (W_LGBM * dfp["LGBM_01"] + W_ENET * dfp["ENet_01"] + W_MLP * dfp["MLP_01"])
    dfp["Validity"]  = dfp["Validity_01"]

    dfp["BRS"] = (ALPHA * dfp["Strength"] + BETA * dfp["Consistency"] + GAMMA * dfp["Validity"]).clip(0, 1)

    dfp["Pair"] = dfp["f1"] + " | " + dfp["f2"]
    dfp = dfp[["Pair", "f1", "f2", "MLP_01", "LGBM_01", "ENet_01", "Strength", "Consistency", "Validity", "BRS"]] \
             .sort_values("BRS", ascending=False)

    return dfp

# ================== Main loop: compute pairwise tables per case ==================
pair_tables = {}
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as wp:
    for case in tqdm(TARGET_COLS, desc="Pairwise scoring per case", ncols=100):
        y = coerce_numeric(raw[case])
        mask = y.notna() & X_all.notna().any(axis=1)
        X_use, y_use = X_all.loc[mask].copy(), y.loc[mask].copy()

        df_pair = pair_scores_for_case(case, X_use, y_use, uni_top, max_pairs=MAX_PAIRS_PER_CASE)
        pair_tables[case] = df_pair

        base = safe_title(case)[:28]
        df_pair.to_excel(wp, sheet_name=f"{base}_Pairs_All", index=False)
        df_pair.head(TOPK).to_excel(wp, sheet_name=f"{base}_Pairs_Top{TOPK}", index=False)

print(f"✅ Pairwise results saved: {OUT_XLSX}")

# ================== Pairwise BRS overview heatmap (Top 10, ID->Name) ==================
# Load ID->Name mapping
id2name = load_id2name_map(
    NAME_MAP_PATH,
    NAME_MAP_SHEET,
    id_col=ID_COL_IN_MAP,
    name_col=NAME_COL_IN_MAP
)

# Collect union of Top10 pairs per case (by BRS)
pairs_union = set()
for case, dfp in pair_tables.items():
    pairs_union.update(dfp.head(10)["Pair"].astype(str).tolist())
pairs_union = list(pairs_union)

# Build matrix: rows=pairs, cols=cases
mat_pair_brs = pd.DataFrame(index=pairs_union, columns=TARGET_COLS, dtype=float)
for case, dfp in pair_tables.items():
    s = dfp.set_index("Pair")["BRS"]
    mat_pair_brs.loc[s.index.intersection(mat_pair_brs.index), case] = s

# Select Top pairs by mean BRS and keep only Top HEAT_TOPN
rows_pair_top = (
    mat_pair_brs.mean(axis=1, skipna=True)
    .sort_values(ascending=False)
    .head(HEAT_TOPN)
    .index.tolist()
)
mat_pair_top = mat_pair_brs.loc[rows_pair_top].fillna(0.0)

# Map pair string "ID1 | ID2" to "Name1 | Name2" for the heatmap index
# If names are duplicated, append short ID tags for disambiguation.
id_list_for_count = []
for pair in mat_pair_top.index.astype(str):
    parts = [p.strip() for p in pair.split("|")]
    if len(parts) >= 2:
        id_list_for_count.extend([parts[0], parts[1]])

mapped_names = [id2name.get(mid, mid) for mid in id_list_for_count]
name_counts = pd.Series(mapped_names).value_counts().to_dict()

def map_id_to_display(mid: str) -> str:
    nm = id2name.get(mid, mid)
    if name_counts.get(nm, 0) > 1:
        return f"{nm} [{short_id_tag(mid)}]"
    return nm

new_index = []
for pair in mat_pair_top.index.astype(str):
    parts = [p.strip() for p in pair.split("|")]
    if len(parts) >= 2:
        p1, p2 = parts[0], parts[1]
        new_index.append(f"{map_id_to_display(p1)} | {map_id_to_display(p2)}")
    else:
        # Fallback: keep as-is
        new_index.append(pair)

mat_pair_top_named = mat_pair_top.copy()
mat_pair_top_named.index = new_index

plot_heat(
    mat_pair_top_named,
    f"BRS Score for Two-Metabolite Combinations (Top {HEAT_TOPN})",
    OUT_HEAT,
    mark_thresh=MARK_THRESH
)

# Save the heatmap numeric values (wide format) and keep both ID and Name
heat_wide = mat_pair_top.copy()
heat_wide.insert(0, "Pair_ID", heat_wide.index.astype(str))

pair_name_list = []
for pair in heat_wide["Pair_ID"].astype(str):
    parts = [p.strip() for p in pair.split("|")]
    if len(parts) >= 2:
        p1, p2 = parts[0], parts[1]
        pair_name_list.append(f"{id2name.get(p1, p1)} | {id2name.get(p2, p2)}")
    else:
        pair_name_list.append(pair)

heat_wide.insert(1, "Pair_Name", pair_name_list)
heat_wide["Mean_BRS"] = mat_pair_top.mean(axis=1)

if os.path.exists(OUT_XLSX):
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        heat_wide.to_excel(w, sheet_name="heatmap_values_wide", index=False)
else:
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        heat_wide.to_excel(w, sheet_name="heatmap_values_wide", index=False)

print("\n✅ Done:")
print(f" - Pairwise tables: {OUT_XLSX}")
print(f" - Heatmap (Top {HEAT_TOPN}, named): {OUT_HEAT}")
