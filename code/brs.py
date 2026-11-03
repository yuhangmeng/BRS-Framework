# -*- coding: utf-8 -*-
"""
Step 1: Balanced Relevance Score (BRS) of single metabolites, separately for each phenotype (AOI% of each strain).

    BRS = 0.2 Strength + 0.2 Consistency + 0.6 Validity
    Strength    = 0.1 LightGBM gain + 0.5 |Elastic Net coefficient| + 0.4 perceptron attention (min-max normalized)
    Consistency = fraction of the three learners whose normalized score is >= median + 0.5 IQR
    Validity    = squared Pearson correlation with the phenotype
Metabolites are prescreened by Pearson r > 0.2 with the phenotype (at most 100 per phenotype).

Environment variables (used by stability_selection.py and nested_cv.py):
    BRS_INPUT        input workbook (default: data/preprocessing/filtered_metabolites_log.xlsx)
    BRS_OUT_DIR      output folder (default: results/brs_full_data)
    BRS_SAMPLE_FILE  optional file with row indices: restricts the analysis to these samples
Output: <BRS_OUT_DIR>/univariate_BRS.xlsx (one sheet per phenotype) and a DONE marker.
"""
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = os.environ.get("BRS_INPUT", str(ROOT / "data" / "preprocessing" / "filtered_metabolites_log.xlsx"))
SHEET_NAME = "Sheet1"

TARGET_COLS = ["Inhibition (%)-NF", "Inhibition (%)-NV", "Inhibition (%)-NM", "Inhibition (%)-NU"]

OUT_DIR = os.environ.get("BRS_OUT_DIR", str(ROOT / "results" / "brs_full_data"))
os.makedirs(OUT_DIR, exist_ok=True)
OUT_XLSX = os.path.join(OUT_DIR, "univariate_BRS.xlsx")

# ===== Speed-related parameters =====
FOLDS, SEED  = 3, 42          # CV
torch.manual_seed(SEED)       # MLP initialisation is otherwise different on every run
np.random.seed(SEED)
MLP_EPOCHS   = 30             # MLP
MLP_HIDDEN   = 32
LGBM_ESTIMATORS    = 50       # LightGBM
LGBM_EARLY_STOP    = 20

MAX_FEATS_PER_CASE = int(os.environ.get("BRS_MAX_FEATS", 100))   # Maximum number of features to include in BRS for each case (the most relevant portion)
R_MIN = float(os.environ.get("BRS_R_MIN", 0.2))                   # prescreen: keep metabolites with Pearson r > R_MIN

# Three-model weights for Strength
W_LGBM, W_ENET, W_MLP = 0.1, 0.5, 0.4
# BRS weights
ALPHA, BETA, GAMMA = 0.2, 0.2, 0.6
# Consistency threshold
CONS_Q = 0.5

# TopK for per-case output
TOPK = 30

# Base LightGBM parameters
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


def sanitize_colname(name: str) -> str:
    s = str(name)
    s = re.sub(r'[\\\"/\t\n\r\b\f\[\]\{\}:,|]', '_', s)
    s = s.replace('(', '_').replace(')', '_').replace('=', '_').replace('+', '_').replace('*', '_').replace(' ', '_')
    s = re.sub(r'_+', '_', s).strip('_')
    return s if s else 'col'


def make_unique(names):
    used, out = set(), []
    for n in names:
        base = sanitize_colname(n) or 'col'
        new = base
        k = 1
        while new in used:
            k += 1
            new = f"{base}_{k}"
        used.add(new)
        out.append(new)
    return out


def _fast_uni_r2(x: np.ndarray, yarr: np.ndarray) -> float:

    x = np.asarray(x, float).ravel()
    y = np.asarray(yarr, float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return 0.0
    r = np.corrcoef(x[mask], y[mask])[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(max(0.0, r ** 2))


# ========= Single-feature MLP model =========
class TinyUniMLP(nn.Module):
    def __init__(self, h=MLP_HIDDEN):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.reg  = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x):
        a = self.gate(x)
        y = self.reg(x * a)
        return y, a


# ========= Global ENet: assign one linear score to all candidate features =========
def compute_enet_scores(X_use: pd.DataFrame, y_use: pd.Series, feats: list) -> dict:
    if len(feats) == 0:
        return {}

    X_sel = X_use[feats].copy()
    for c in X_sel.columns:
        med = X_sel[c].median(skipna=True)
        X_sel[c] = X_sel[c].fillna(med)

    X_mat = X_sel.values.astype(float)
    yarr = y_use.values.astype(float)

    mask = np.isfinite(X_mat).all(axis=1) & np.isfinite(yarr)
    X_mat, yarr = X_mat[mask], yarr[mask]
    if X_mat.shape[0] < 5:
        return {f: 0.0 for f in feats}

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_mat)

    try:
        enet = ElasticNetCV(
            l1_ratio=[0.7, 0.85, 1.0],
            cv=3,
            random_state=SEED,
            n_jobs=-1,
            max_iter=2000
        )
        enet.fit(X_std, yarr)
        coefs = np.abs(enet.coef_)
    except Exception as e:
        print(f"[ENet] Fallback to 0 because of error: {e}")
        return {f: 0.0 for f in feats}

    return {f: float(c) for f, c in zip(feats, coefs)}


# ================== Core single-feature BRS function ==================
def metabolite_scores_for_case(case_name: str,
                               X_use: pd.DataFrame,
                               y_use: pd.Series,
                               candidate_feats: list,
                               enet_scores_raw: dict):
    feats = [f for f in candidate_feats if f in X_use.columns]
    yarr = y_use.values.astype(float)
    records = []

    for idx, f in enumerate(feats, 1):
        x = X_use[f].values.reshape(-1, 1).astype(float)
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

        enet_raw = float(enet_scores_raw.get(f, 0.0))

        # (B) LightGBM gain
        lgbm_score = 0.0
        try:
            safe_cols = make_unique([f])
            X_df_local = pd.DataFrame(x, columns=safe_cols)
            X_df_local = X_df_local.fillna(X_df_local.median(axis=0, skipna=True))

            kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
            gain_sum, eff = 0.0, 0
            for tr, va in kf.split(X_df_local):
                model = lgb.LGBMRegressor(**{
                    **LGBM_SAFE,
                    "n_estimators": LGBM_ESTIMATORS,
                    "learning_rate": 0.05,
                    "num_leaves": 7,
                    "min_data_in_leaf": 3
                })
                model.fit(
                    X_df_local.iloc[tr], y_use.iloc[tr],
                    eval_set=[(X_df_local.iloc[va], y_use.iloc[va])],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False)]
                )
                gain = model.booster_.feature_importance(importance_type="gain")[0]
                gain_sum += gain
                eff += 1
            if eff > 0:
                lgbm_score = float(gain_sum / eff)
        except Exception:
            pass


        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        att_coll = []
        for tr, va in kf.split(x):
            Xtr = torch.tensor(x[tr], dtype=torch.float32)
            ytr = torch.tensor(yarr[tr], dtype=torch.float32).view(-1, 1)
            Xva = torch.tensor(x[va], dtype=torch.float32)

            model = TinyUniMLP(h=MLP_HIDDEN)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            for _ in range(MLP_EPOCHS):
                opt.zero_grad()
                pred, _ = model(Xtr)
                loss = loss_fn(pred, ytr)
                loss.backward()
                opt.step()

            with torch.no_grad():
                _, a = model(Xva)
                att_coll.append(a.mean(0).numpy())

        att = np.mean(att_coll, axis=0)
        mlp_score = float(np.clip(att, 0, None)[0])

        # (D) Validity: corr^2
        validity_raw = _fast_uni_r2(x, yarr)

        records.append((f, mlp_score, lgbm_score, enet_raw, validity_raw))

        if (idx % 100) == 0:
            print(f"[UNI {case_name}]  {idx}/{len(feats)}")

    dfu = pd.DataFrame(records,
                       columns=["Feature", "MLP_raw", "LGBM_raw", "ENet_raw", "Validity_raw"])

    for col in ["MLP_raw", "LGBM_raw", "ENet_raw", "Validity_raw"]:
        dfu[col.replace("_raw", "_01")] = minmax_01(dfu[col].values)

    for col in ["MLP_01", "LGBM_01", "ENet_01"]:
        gate = median_iqr_gate(dfu[col].values, q=CONS_Q)
        dfu[col.replace("_01", "_hit")] = (dfu[col] >= gate).astype(int)
    dfu["Consistency"] = (dfu["MLP_hit"] + dfu["LGBM_hit"] + dfu["ENet_hit"]) / 3.0

    dfu["Strength"] = (
        W_LGBM * dfu["LGBM_01"] +
        W_ENET * dfu["ENet_01"] +
        W_MLP  * dfu["MLP_01"]
    )
    dfu["Validity"] = dfu["Validity_01"]

    dfu["BRS"] = (
        ALPHA * dfu["Strength"] +
        BETA  * dfu["Consistency"] +
        GAMMA * dfu["Validity"]
    ).clip(0, 1)

    dfu = dfu[["Feature", "MLP_01", "LGBM_01", "ENet_01",
               "Strength", "Consistency", "Validity", "BRS"]].sort_values("BRS", ascending=False)

    return dfu


# ================== Load data ==================
raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, decimal=',').dropna(how="all")
if os.environ.get("BRS_SAMPLE_FILE"):   # optional row subset, used by the subsampling stability analysis
    raw = raw.iloc[np.loadtxt(os.environ["BRS_SAMPLE_FILE"], dtype=int)].reset_index(drop=True)
for t in TARGET_COLS:
    if t not in raw.columns:
        raise ValueError(f"Target column not found: {t}")

feature_cols = [c for c in raw.columns if c not in TARGET_COLS]
X_num = pd.DataFrame({c: coerce_numeric(raw[c]) for c in feature_cols})
feat_cols = [c for c in X_num.columns if X_num[c].notna().any()]
X_all = X_num[feat_cols].copy()

print(f"Total features: {len(feat_cols)}")


# ================== main ==================
uni_tables = {}
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as wu:
    for case in tqdm(TARGET_COLS, desc="use case calculate BRS (fast_v2)", ncols=100):
        y = coerce_numeric(raw[case])
        mask = y.notna() & X_all.notna().any(axis=1)
        X_use, y_use = X_all.loc[mask].copy(), y.loc[mask].copy()

        # ==== Step 1 ====
        corrs_pos = []
        yv = y_use.values.astype(float)

        for f in feat_cols:
            xv = X_use[f].values.astype(float)
            mask2 = np.isfinite(xv) & np.isfinite(yv)
            if mask2.sum() < 5:
                continue
            r = np.corrcoef(xv[mask2], yv[mask2])[0, 1]
            if np.isfinite(r) and r > R_MIN:
                corrs_pos.append((f, r))

        if len(corrs_pos) == 0:
            print(f"[{case}] No candidate features were found.")
            cand_feats = []
        else:
            corrs_sorted = sorted(corrs_pos, key=lambda x: x[1], reverse=True)
            cand_feats = [f for f, r in corrs_sorted[:MAX_FEATS_PER_CASE]]

        print(
            f"[{case}] Positively correlated features: {len(corrs_pos)}, "
            f"features used for BRS: {len(cand_feats)}"
        )

        # ==== Step 2: compute ENet scores for all candidate features at once ====
        enet_scores_raw = compute_enet_scores(X_use, y_use, cand_feats)

        # ==== Step 3: compute MLP, LGBM, Validity, and BRS for each candidate feature ====
        df_uni = metabolite_scores_for_case(
            case,
            X_use,
            y_use,
            candidate_feats=cand_feats,
            enet_scores_raw=enet_scores_raw
        )
        uni_tables[case] = df_uni

        base = safe_title(case)[:28]
        df_uni.to_excel(wu, sheet_name=f"{base}_Uni_All", index=False)
        df_uni.head(TOPK).to_excel(wu, sheet_name=f"{base}_Uni_Top{TOPK}", index=False)

print(f"Single-feature BRS results saved: {OUT_XLSX}")
Path(OUT_DIR, "DONE").write_text("done\n", encoding="utf-8")
