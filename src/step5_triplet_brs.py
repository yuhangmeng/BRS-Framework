from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
# -*- coding: utf-8 -*-
"""
Standalone script: Triplet (3-metabolite) three-method ensemble + BRS scoring + overview heatmap
(with LightGBM feature-name sanitization)

MODIFICATIONS (biologically-motivated, keeping 3rd-order interaction):
1) 3rd-order interaction uses centered-deviation product:
      (x1-μ1)(x2-μ2)(x3-μ3)
   and pairwise interactions use (xi-μi)(xj-μj)
2) Apply tanh saturation to interaction block to reduce multiplicative blow-up:
      Z_inter <- tanh(Z_inter / TANH_SCALE)
3) Validity is marginal gain over the best embedded pair (OOF R²):
      Validity_raw = max(0, R2_triplet - max(R2_pair12, R2_pair13, R2_pair23))

Inputs:
  1) Raw data table: EXCEL_PATH (metabolite features + 4 target columns)
  2) Univariate result table: UNIVARIATE_XLSX (from the "univariate three-method + BRS" script)
  3) External name mapping table: NAME_MAP_PATH (ID -> Compound name)

Outputs:
  - triple_BRS.xlsx
  - heat_triplet_top10_named.png
"""

import os, re, warnings
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
from pathlib import Path

# ================== Configuration ==================
EXCEL_PATH   = str(_ROOT / "data" / "preprocessing" / "filtered_metabolites.xlsx")
SHEET_NAME   = "Sheet1"

TARGET_COLS   = ["Inhibition (%)-NF", "Inhibition (%)-NV", "Inhibition (%)-NM", "Inhibition (%)-NU"]
TARGET_COLS1  = ["NF", "NV", "NM", "NU"]  # compatibility (unused)

UNIVARIATE_XLSX = str(_ROOT / "results" / "univariate_BRS.xlsx")

OUT_DIR  = str(_ROOT / "results")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_XLSX = os.path.join(OUT_DIR, "triple_BRS.xlsx")

NAME_MAP_PATH   = str(_ROOT / "data" / "name.xlsx")
NAME_MAP_SHEET  = "Sheet1"
ID_COL_IN_MAP   = "ID"
NAME_COL_IN_MAP = "Compound name"

FOLDS, SEED = 5, 42

# Strength weights (three-model ensemble)
W_LGBM, W_ENET, W_MLP = 0.1, 0.5, 0.4

# BRS weights
ALPHA, BETA, GAMMA = 0.2, 0.2, 0.6

# Consistency gating threshold parameter
CONS_Q = 0.5

MARK_THRESH = 0.6
TOPK = 10

# Speed controls
TRIPLET_TOPK_FROM_UNI = TOPK
TRIPLET_GLOBAL_CAP    = 10
MAX_TRIPLETS_PER_CASE = 500

# Include interactions (pairwise + third-order kept)
TRIP_ADD_INTERACTIONS = True

# Interaction saturation (biological "saturation" prior)
APPLY_TANH_SAT = True
TANH_SCALE    = 3.0   # typical 2~5; smaller -> stronger saturation

# Model budgets
MLP3_EPOCHS           = 80
MLP3_HIDDEN           = 32
ENET3_BOOT            = 30
LGBM3_ESTIMATORS      = 320
LGBM3_EARLY_STOP      = 50

HEAT_TOPN = 10
OUT_HEAT_TRIP = Path(OUT_DIR) / "heat_triplet_top10_named.png"

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
    if s.dtype.kind in "biufc":
        return s.astype(float)
    x = s.astype(str).str.strip()
    x = x.str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)
    x = x.str.replace(",", ".", regex=False)
    x = x.str.replace(r"[^0-9.\-eE]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")

def minmax_01(v, eps=1e-12):
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)

def median_iqr_gate(a: np.ndarray, q=0.5):
    a = np.asarray(a, float)
    med = np.nanmedian(a)
    q1, q3 = np.nanpercentile(a, 25), np.nanpercentile(a, 75)
    iqr = q3 - q1
    return med + q * iqr

def safe_title(s: str) -> str:
    return str(s).replace("%", "pct").replace("/", "_")

def plot_heat(mat_df, title, out_png, mark_thresh=None):
    H = max(6, 0.45 * len(mat_df))
    fig, ax = plt.subplots(figsize=(8.2, H))
    im = ax.imshow(mat_df.values, aspect="auto", vmin=0, vmax=1, cmap="coolwarm")

    ylabels = [re.sub(r"\s*\|\s*", " | ", str(s)) for s in mat_df.index]
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9)

    ax.set_xticks(np.arange(mat_df.shape[1]))
    ax.set_xticklabels(list(mat_df.columns), rotation=20, ha="right", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_ylabel("Score (0–1)", rotation=90, va="center")

    ax.set_title(title, fontsize=13, pad=14)

    maxlen = max((len(str(s)) for s in mat_df.index), default=0)
    left = min(0.62, 0.20 + 0.006 * maxlen)
    fig.subplots_adjust(left=left, right=0.96, top=0.92, bottom=0.10)

    if mark_thresh is not None:
        arr = mat_df.values
        ys, xs = np.where(arr >= mark_thresh)
        for y, x in zip(ys, xs):
            ax.text(x, y, "○", ha="center", va="center", fontsize=9)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def load_id2name_map(map_path, sheet_name, id_col="ID", name_col="Compound name"):
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
    parts = str(met_id).split("_")
    return parts[1] if len(parts) >= 2 else str(met_id)[:8]

# LightGBM safe feature names
def sanitize_colname(name: str) -> str:
    s = str(name)
    s = re.sub(r'[\\\"/\t\n\r\b\f\[\]\{\}:,|]', "_", s)
    s = s.replace("(", "_").replace(")", "_").replace("=", "_").replace("+", "_").replace("*", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "col"

def make_unique(names):
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

# ================== Load raw data ==================
raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, decimal=",").dropna(how="all")
for t in TARGET_COLS:
    if t not in raw.columns:
        raise ValueError(f"Target column not found: {t}")

feature_cols = [c for c in raw.columns if c not in TARGET_COLS]
X_num = pd.DataFrame({c: coerce_numeric(raw[c]) for c in feature_cols})
feat_cols = [c for c in X_num.columns if X_num[c].notna().any()]
X_all = X_num[feat_cols].copy()

# ================== Load TopK candidates from univariate results ==================
def load_univariate_topk(univar_xlsx_path: str, target_cols, topk: int):
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

uni_topk_map = load_univariate_topk(UNIVARIATE_XLSX, TARGET_COLS, TRIPLET_TOPK_FROM_UNI)

uni_top_union = set()
for case in TARGET_COLS:
    uni_top_union.update(uni_topk_map[case])
uni_top = list(uni_top_union)[:TRIPLET_GLOBAL_CAP]
print(f"[TRIPLET] Candidate features: {len(uni_top)} (union of univariate Top{TRIPLET_TOPK_FROM_UNI} across cases, capped at {TRIPLET_GLOBAL_CAP})")

# ================== Core feature builders (centered interactions + tanh saturation) ==================
def _zscore_cols(M: np.ndarray, eps=1e-12) -> np.ndarray:
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0) + eps
    return (M - mu) / sd

def _apply_tanh_saturation(M: np.ndarray, start_col: int, scale: float):
    """Apply tanh saturation to columns >= start_col."""
    if M.shape[1] <= start_col:
        return M
    Z = M[:, start_col:]
    Z = np.tanh(Z / float(scale))
    M[:, start_col:] = Z
    return M

def _make_trip_matrix(X_df: pd.DataFrame, f1: str, f2: str, f3: str, add_inter=False):
    """
    Triplet design:
      base: x1, x2, x3
      interactions (centered deviations): c1*c2, c1*c3, c2*c3, c1*c2*c3  (kept)
    then optional tanh saturation on interaction block
    then z-score all columns
    """
    x1 = X_df[f1].values.reshape(-1, 1)
    x2 = X_df[f2].values.reshape(-1, 1)
    x3 = X_df[f3].values.reshape(-1, 1)

    # centered deviations (biologically: deviation from baseline)
    c1 = x1 - np.nanmean(x1)
    c2 = x2 - np.nanmean(x2)
    c3 = x3 - np.nanmean(x3)

    cols = [f1, f2, f3]
    Xt = np.hstack([x1, x2, x3])

    if add_inter:
        Xt = np.hstack([Xt, c1*c2, c1*c3, c2*c3, c1*c2*c3])
        cols += [f"{f1}*{f2}", f"{f1}*{f3}", f"{f2}*{f3}", f"{f1}*{f2}*{f3}"]

    # tanh saturation on interaction block only
    if add_inter and APPLY_TANH_SAT:
        Xt = _apply_tanh_saturation(Xt, start_col=3, scale=TANH_SCALE)

    Xt = _zscore_cols(Xt)
    return Xt, cols

def _make_pair_matrix(X_df: pd.DataFrame, fa: str, fb: str):
    """
    Pair design for baseline validity:
      base: xa, xb
      interaction: (xa-μa)(xb-μb)  (centered deviation product)
    then tanh saturation on interaction column
    then z-score
    """
    xa = X_df[fa].values.reshape(-1, 1)
    xb = X_df[fb].values.reshape(-1, 1)

    ca = xa - np.nanmean(xa)
    cb = xb - np.nanmean(xb)

    Xp = np.hstack([xa, xb, ca*cb])

    if APPLY_TANH_SAT:
        Xp = _apply_tanh_saturation(Xp, start_col=2, scale=TANH_SCALE)

    Xp = _zscore_cols(Xp)
    return Xp

# ================== Scoring helpers ==================
def _oof_r2_linear(Xmat: np.ndarray, yarr: np.ndarray, folds=FOLDS):
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros_like(yarr, dtype=float)
    for tr, va in kf.split(Xmat):
        lr = LinearRegression()
        lr.fit(Xmat[tr], yarr[tr])
        oof[va] = lr.predict(Xmat[va])
    return max(0.0, float(r2_score(yarr, oof)))

class TinyTripletMLP(nn.Module):
    def __init__(self, d, h=MLP3_HIDDEN):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.reg  = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x):
        a = self.gate(x)
        y = self.reg(x * a)
        return y, a

def _build_triplet_list(feats: list, max_tris: int):
    all_tris = list(combinations(feats, 3))
    if len(all_tris) <= max_tris:
        return all_tris

    M = min(len(feats), max(9, int(round(max_tris ** (1/3))) * 5))
    favored = []
    for i, _ in enumerate(feats[:M]):
        for j in range(i + 1, min(i + 1 + 15, len(feats))):
            for k in range(j + 1, min(j + 1 + 8, len(feats))):
                favored.append(tuple(sorted((feats[i], feats[j], feats[k]))))

    seen = set()
    favored_dedup = []
    for t in favored:
        if t not in seen:
            seen.add(t)
            favored_dedup.append(t)
        if len(favored_dedup) >= max_tris:
            return favored_dedup[:max_tris]

    remain = [t for t in all_tris if t not in seen]
    rng = np.random.RandomState(SEED)
    rng.shuffle(remain)
    return (favored_dedup + remain)[:max_tris]

# ================== Triplet scoring per case ==================
def triplet_scores_for_case(case_name: str, X_use: pd.DataFrame, y_use: pd.Series,
                            uni_candidates: list, max_tris=MAX_TRIPLETS_PER_CASE):
    feats = [f for f in uni_candidates if f in X_use.columns]
    tri_list = _build_triplet_list(feats, max_tris=max_tris)
    print(f"[TRIPLET {case_name}] Triplets evaluated: {len(tri_list)}")

    yarr = y_use.values.astype(float)
    records = []

    # Cache pairwise OOF R² to avoid recomputation
    pair_r2_cache = {}

    def pair_r2(fa, fb):
        key = tuple(sorted((fa, fb)))
        if key in pair_r2_cache:
            return pair_r2_cache[key]
        Xp = _make_pair_matrix(X_use, key[0], key[1])
        r2 = _oof_r2_linear(Xp, yarr, folds=FOLDS)
        pair_r2_cache[key] = r2
        return r2

    for idx, (f1, f2, f3) in enumerate(tri_list, 1):
        Xt, cols = _make_trip_matrix(X_use, f1, f2, f3, add_inter=TRIP_ADD_INTERACTIONS)

        # (A) ElasticNet stability selection via bootstrapping
        sel_cnt = 0
        rng = np.random.RandomState(SEED)
        for b in range(ENET3_BOOT):
            n = max(3, int(len(yarr) * 0.7))
            sub = rng.choice(len(yarr), size=n, replace=False)
            Xb, yb = Xt[sub], yarr[sub]
            keep_mask = np.isfinite(Xb).all(axis=0)
            if not np.any(keep_mask):
                continue
            model = ElasticNetCV(l1_ratio=[0.7, 0.85, 1.0], cv=3, random_state=SEED + b, max_iter=2000)
            model.fit(Xb[:, keep_mask], yb)
            if np.any(np.abs(model.coef_) > 1e-10):
                sel_cnt += 1
        enet_freq = sel_cnt / max(ENET3_BOOT, 1)

        # (B) LightGBM + SHAP (use safe column names)
        lgbm_score = 0.0
        try:
            safe_cols = make_unique(cols)
            X_df_local = pd.DataFrame(Xt, columns=safe_cols)
            med = X_df_local.median(axis=0, skipna=True)
            X_df_local = X_df_local.fillna(med)

            kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
            shap_sum = np.zeros(X_df_local.shape[1], dtype=float)
            eff = 0

            for tr, va in kf.split(X_df_local):
                model = lgb.LGBMRegressor(**{
                    **LGBM_SAFE,
                    "n_estimators": LGBM3_ESTIMATORS,
                    "learning_rate": 0.05,
                    "num_leaves": 7,
                    "min_data_in_leaf": 3
                })
                model.fit(
                    X_df_local.iloc[tr], y_use.iloc[tr],
                    eval_set=[(X_df_local.iloc[va], y_use.iloc[va])],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(LGBM3_EARLY_STOP, verbose=False)]
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
            pass

        # (C) Triplet MLP attention score (lightweight)
        d = Xt.shape[1]
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        att_coll = []
        for tr, va in kf.split(Xt):
            Xtr = torch.tensor(Xt[tr], dtype=torch.float32)
            ytr = torch.tensor(yarr[tr], dtype=torch.float32).view(-1, 1)
            Xva = torch.tensor(Xt[va], dtype=torch.float32)

            model = TinyTripletMLP(d)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            for _ in range(MLP3_EPOCHS):
                opt.zero_grad()
                pred, _ = model(Xtr)
                loss = loss_fn(pred, ytr)
                loss.backward()
                opt.step()

            with torch.no_grad():
                _, a = model(Xva)
                att_coll.append(a.mean(0).numpy())

        att = np.mean(att_coll, axis=0)
        mlp_trip = float(np.mean(minmax_01(np.clip(att, 0, None))))

        # ---- Validity: marginal gain over best embedded pair (OOF R²) ----
        tri_r2 = _oof_r2_linear(Xt, yarr, folds=FOLDS)
        best_pair_r2 = max(pair_r2(f1, f2), pair_r2(f1, f3), pair_r2(f2, f3))
        validity_raw = max(0.0, tri_r2 - best_pair_r2)

        records.append((f1, f2, f3, mlp_trip, lgbm_score, enet_freq, validity_raw))

        if (idx % 500) == 0:
            print(f"[TRIPLET {case_name}] Progress {idx}/{len(tri_list)}")

    dft = pd.DataFrame(records, columns=["f1", "f2", "f3", "MLP_raw", "LGBM_raw", "ENet_raw", "Validity_raw"])

    for col in ["MLP_raw", "LGBM_raw", "ENet_raw", "Validity_raw"]:
        dft[col.replace("_raw", "_01")] = minmax_01(dft[col].values)

    for col in ["MLP_01", "LGBM_01", "ENet_01"]:
        gate = median_iqr_gate(dft[col].values, q=CONS_Q)
        dft[col.replace("_01", "_hit")] = (dft[col] >= gate).astype(int)

    dft["Consistency"] = (dft["MLP_hit"] + dft["LGBM_hit"] + dft["ENet_hit"]) / 3.0
    dft["Strength"] = (W_LGBM * dft["LGBM_01"] + W_ENET * dft["ENet_01"] + W_MLP * dft["MLP_01"])
    dft["Validity"]  = dft["Validity_01"]

    dft["BRS"] = (ALPHA * dft["Strength"] + BETA * dft["Consistency"] + GAMMA * dft["Validity"]).clip(0, 1)

    dft["Triplet"] = dft["f1"] + " | " + dft["f2"] + " | " + dft["f3"]
    dft = dft[["Triplet", "f1", "f2", "f3",
               "MLP_01", "LGBM_01", "ENet_01",
               "Strength", "Consistency", "Validity", "BRS"]].sort_values("BRS", ascending=False)
    return dft

# ================== Main loop: compute triplet tables per case ==================
triplet_tables = {}
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as wp:
    for case in tqdm(TARGET_COLS, desc="Triplet scoring per case", ncols=100):
        y = coerce_numeric(raw[case])
        mask = y.notna() & X_all.notna().any(axis=1)
        X_use, y_use = X_all.loc[mask].copy(), y.loc[mask].copy()

        dft = triplet_scores_for_case(case, X_use, y_use, uni_top, max_tris=MAX_TRIPLETS_PER_CASE)
        triplet_tables[case] = dft

        base = safe_title(case)[:28]
        dft.to_excel(wp, sheet_name=f"{base}_Triplets_All", index=False)
        dft.head(TOPK).to_excel(wp, sheet_name=f"{base}_Triplets_Top{TOPK}", index=False)

print(f"✅ Triplet results saved: {OUT_XLSX}")

# ================== Overview heatmap (Top 10, ID->Name) ==================
id2name = load_id2name_map(
    NAME_MAP_PATH,
    NAME_MAP_SHEET,
    id_col=ID_COL_IN_MAP,
    name_col=NAME_COL_IN_MAP
)

trip_union = set()
for case, dft in triplet_tables.items():
    trip_union.update(dft.head(10)["Triplet"].astype(str).tolist())
trip_union = list(trip_union)

mat_trip_brs = pd.DataFrame(index=trip_union, columns=TARGET_COLS, dtype=float)
for case, dft in triplet_tables.items():
    s = dft.set_index("Triplet")["BRS"]
    mat_trip_brs.loc[s.index.intersection(mat_trip_brs.index), case] = s

rows_trip_top = (
    mat_trip_brs.mean(axis=1, skipna=True)
    .sort_values(ascending=False)
    .head(HEAT_TOPN)
    .index.tolist()
)
mat_trip_top = mat_trip_brs.loc[rows_trip_top].fillna(0.0)

# name disambiguation for heatmap labels
id_list_for_count = []
for trip in mat_trip_top.index.astype(str):
    parts = [p.strip() for p in trip.split("|")]
    if len(parts) >= 3:
        id_list_for_count.extend([parts[0], parts[1], parts[2]])

mapped_names = [id2name.get(mid, mid) for mid in id_list_for_count]
name_counts = pd.Series(mapped_names).value_counts().to_dict()

def map_id_to_display(mid: str) -> str:
    nm = id2name.get(mid, mid)
    if name_counts.get(nm, 0) > 1:
        return f"{nm} [{short_id_tag(mid)}]"
    return nm

new_index = []
for trip in mat_trip_top.index.astype(str):
    parts = [p.strip() for p in trip.split("|")]
    if len(parts) >= 3:
        p1, p2, p3 = parts[0], parts[1], parts[2]
        new_index.append(f"{map_id_to_display(p1)} | {map_id_to_display(p2)} | {map_id_to_display(p3)}")
    else:
        new_index.append(trip)

mat_trip_top_named = mat_trip_top.copy()
mat_trip_top_named.index = new_index

plot_heat(
    mat_trip_top_named,
    f"BRS Score for Three-Metabolite Combinations (Top {HEAT_TOPN})",
    OUT_HEAT_TRIP,
    mark_thresh=MARK_THRESH
)

# Save heatmap numeric values (wide)
heat_wide = mat_trip_top.copy()
heat_wide.insert(0, "Triplet_ID", heat_wide.index.astype(str))

trip_name_list = []
for trip in heat_wide["Triplet_ID"].astype(str):
    parts = [p.strip() for p in trip.split("|")]
    if len(parts) >= 3:
        p1, p2, p3 = parts[0], parts[1], parts[2]
        trip_name_list.append(f"{id2name.get(p1, p1)} | {id2name.get(p2, p2)} | {id2name.get(p3, p3)}")
    else:
        trip_name_list.append(trip)

heat_wide.insert(1, "Triplet_Name", trip_name_list)
heat_wide["Mean_BRS"] = mat_trip_top.mean(axis=1)

if os.path.exists(OUT_XLSX):
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        heat_wide.to_excel(w, sheet_name="heatmap_values_wide", index=False)
else:
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        heat_wide.to_excel(w, sheet_name="heatmap_values_wide", index=False)

print(f"✅ Triplet heatmap saved: {OUT_HEAT_TRIP}")
