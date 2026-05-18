from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
# -*- coding: utf-8 -*-
"""
Comprehensive Benchmark for BRS Framework
==========================================
This script provides five categories of validation experiments:

  (A) Multi-scenario synthetic benchmark
      - Varying noise levels, sample sizes, and signal strengths
      - Recovery metrics: Recall@5/10/20, pair rank, triplet rank

  (B) Ablation study
      - Full BRS vs. removing each component (MLP, LightGBM, ElasticNet)
      - Full BRS vs. removing Consistency or Validity terms

  (C) Weight sensitivity analysis
      - Grid search over (alpha, beta, gamma) and model weights
      - Jaccard overlap of top-10 rankings across weight settings

  (D) Additional baselines
      - Permutation Importance (sklearn), Mutual Information
      - Compared alongside LASSO, ElasticNet, RF, LightGBM+SHAP

  (E) Triplet recovery benchmark
      - Evaluates whether methods can recover the planted triplet

Outputs:
  results/benchmark_comprehensive/
    scenario_table.csv          -- multi-scenario results
    scenario_heatmap.png/pdf    -- heatmap of Recall@K across scenarios
    ablation_table.csv          -- ablation results
    ablation_figure.png/pdf
    sensitivity_table.csv       -- weight sensitivity
    sensitivity_figure.png/pdf
    full_benchmark_table.csv    -- all methods, all metrics (default scenario)
    full_benchmark_figure.png/pdf
"""
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

warnings.filterwarnings("ignore")

# ====================== Global Config ======================
OUT_DIR = Path(str(_ROOT / "results" / "benchmark_comprehensive"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
RNG = np.random.default_rng(SEED)

# BRS default hyper-parameters (matching Main Method.py)
BRS_PRESCREEN_R = 0.2
BRS_MLP_EPOCHS = 30
BRS_MLP_HIDDEN = 32
BRS_FOLDS = 3
BRS_LGBM_PARAMS = dict(n_estimators=50, learning_rate=0.05, num_leaves=7,
                       min_data_in_leaf=3, verbose=-1)
BRS_W_LGBM, BRS_W_ENET, BRS_W_MLP = 0.1, 0.5, 0.4
BRS_ALPHA, BRS_BETA, BRS_GAMMA = 0.2, 0.2, 0.6
BRS_CONS_Q = 0.5

# Pair scoring hyper-parameters (matching Main Method(pair).py)
BRS_PAIR_FOLDS = 5
BRS_PAIR_MLP_EPOCHS = 80
BRS_PAIR_LGBM_EST = 400
BRS_PAIR_ENET_BOOT = 40

# Triplet scoring
BRS_TRIPLET_MLP_EPOCHS = 80
BRS_TRIPLET_LGBM_EST = 320
BRS_TRIPLET_ENET_BOOT = 30
TANH_SCALE = 3.0

# Stability
K_TOP = 10
STABILITY_B = 20
TOP_UNI_FOR_COMBO = 20

# Matplotlib style
plt.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
})

COLORS = {
    "BRS": "#4C78A8",
    "LASSO": "#F58518",
    "ElasticNet": "#54A24B",
    "RandomForest": "#E45756",
    "LightGBM+SHAP": "#72B7B2",
    "PermutationImp": "#B279A2",
    "MutualInfo": "#FF9DA6",
}

# ====================== Data Generation ======================

def generate_synthetic(n_samples=50, n_noise=195, noise_std=0.4,
                       coef_lin=(1.8, -1.5, 1.3),
                       coef_nonlin=(2.2, 1.8),
                       coef_pair=2.4, coef_triplet=2.8,
                       seed=42):
    """Generate synthetic dataset with known univariate, pair, and triplet signals."""
    rng = np.random.default_rng(seed)
    n_total = 5 + n_noise

    X = rng.normal(0, 1, (n_samples, n_total))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    feat_names = (["X_lin_1", "X_lin_2", "X_lin_3", "X_nonlin_1", "X_nonlin_2"]
                  + [f"X_noise_{i+1}" for i in range(n_noise)])

    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    xn1, xn2 = X[:, 3], X[:, 4]

    # Linear main effects
    z_lin = coef_lin[0]*x1 + coef_lin[1]*x2 + coef_lin[2]*x3

    # Nonlinear main effects
    tmp1 = (xn1 - 0.8)**2
    z_nl1 = coef_nonlin[0] * (tmp1 - tmp1.mean())
    tmp2 = np.maximum(0, xn2 - 0.1)
    z_nl2 = coef_nonlin[1] * (tmp2 - tmp2.mean())

    # Synergistic effects
    z_pair = coef_pair * (x1 * xn1)
    z_triplet = coef_triplet * (x2 * x3 * xn2)

    eps = rng.normal(0, noise_std, n_samples)
    y = z_lin + z_nl1 + z_nl2 + z_pair + z_triplet + eps

    true_uni = {"X_lin_1", "X_lin_2", "X_lin_3", "X_nonlin_1", "X_nonlin_2"}
    true_pair = ("X_lin_1", "X_nonlin_1")
    true_triplet = ("X_lin_2", "X_lin_3", "X_nonlin_2")

    return X, y, feat_names, true_uni, true_pair, true_triplet


# ====================== MLP Models ======================

class TinyUniMLP(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        self.reg = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        a = self.gate(x)
        return self.reg(x * a), a


class TinyComboMLP(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.reg = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        a = self.gate(x)
        return self.reg(x * a), a


# ====================== BRS Scoring Engine ======================

def _mlp_gate(x_col, y, epochs=BRS_MLP_EPOCHS, folds=BRS_FOLDS):
    x = torch.tensor(x_col, dtype=torch.float32).view(-1, 1)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    gates = []
    for tr, va in kf.split(x):
        model = TinyUniMLP()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            pred, _ = model(x[tr])
            loss = nn.functional.mse_loss(pred, yt[tr])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            _, a = model(x[va])
            gates.append(float(a.mean()))
    return float(np.mean(gates))


def _minmax(d):
    if not d:
        return d
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    rng = (hi - lo) if hi > lo else 1.0
    return {k: float((v - lo) / rng) for k, v in d.items()}


def _minmax_vec(v):
    v = np.asarray(v, dtype=float)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def _consistency_hits(vals, q=BRS_CONS_Q):
    """Binary hit indicator using median + q*IQR threshold."""
    if isinstance(vals, dict):
        v = np.array(list(vals.values()), dtype=float)
    else:
        v = np.asarray(vals, dtype=float)
    med = np.median(v)
    q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
    thr = med + q * (q3 - q1)
    if isinstance(vals, dict):
        return {k: (1.0 if vals[k] >= thr else 0.0) for k in vals}
    return (v >= thr).astype(float)


def brs_univariate(X, y, feat_names,
                   w_lgbm=BRS_W_LGBM, w_enet=BRS_W_ENET, w_mlp=BRS_W_MLP,
                   alpha=BRS_ALPHA, beta=BRS_BETA, gamma=BRS_GAMMA,
                   use_mlp=True, use_lgbm=True, use_enet=True,
                   use_consistency=True, use_validity=True):
    """
    Full BRS univariate scoring with optional ablation switches.

    Returns: (ranked_names, scores_array)
    """
    n_feat = X.shape[1]

    # Prescreen by |Pearson r|
    r_abs = np.zeros(n_feat)
    for j in range(n_feat):
        try:
            r_abs[j] = abs(pearsonr(X[:, j], y)[0])
        except Exception:
            r_abs[j] = 0.0
    prescreen = np.where(r_abs > BRS_PRESCREEN_R)[0]
    if len(prescreen) > 100:
        order = np.argsort(r_abs[prescreen])[::-1][:100]
        prescreen = prescreen[order]
    if len(prescreen) == 0:
        prescreen = np.argsort(r_abs)[::-1][:30]

    # MLP gate importance
    mlp_raw = {}
    if use_mlp:
        for j in prescreen:
            mlp_raw[j] = _mlp_gate(X[:, j], y)
    else:
        for j in prescreen:
            mlp_raw[j] = 0.0

    # LightGBM gain importance
    lgbm_raw = {}
    if use_lgbm:
        for j in prescreen:
            kf = KFold(n_splits=BRS_FOLDS, shuffle=True, random_state=42)
            gains = []
            for tr, va in kf.split(X):
                m = lgb.LGBMRegressor(**BRS_LGBM_PARAMS)
                try:
                    m.fit(X[tr, j:j+1], y[tr],
                          eval_set=[(X[va, j:j+1], y[va])],
                          callbacks=[lgb.early_stopping(20, verbose=False)])
                    gains.append(float(m.booster_.feature_importance(
                        importance_type="gain")[0]))
                except Exception:
                    gains.append(0.0)
            lgbm_raw[j] = float(np.mean(gains))
    else:
        for j in prescreen:
            lgbm_raw[j] = 0.0

    # ElasticNet coefficient
    enet_raw = {}
    if use_enet:
        Xp = X[:, prescreen]
        Xp_std = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)
        enet = ElasticNetCV(l1_ratio=[0.7, 0.85, 1.0], cv=3, max_iter=2000,
                            random_state=42)
        enet.fit(Xp_std, y)
        for k, j in enumerate(prescreen):
            enet_raw[j] = float(abs(enet.coef_[k]))
    else:
        for j in prescreen:
            enet_raw[j] = 0.0

    # Validity
    validity_raw = {}
    for j in prescreen:
        validity_raw[j] = float(r_abs[j]**2) if use_validity else 0.0

    # Normalize
    mlp_n = _minmax(mlp_raw)
    lgbm_n = _minmax(lgbm_raw)
    enet_n = _minmax(enet_raw)
    val_n = _minmax(validity_raw)

    # Consistency hits
    h_mlp = _consistency_hits(mlp_n)
    h_lgbm = _consistency_hits(lgbm_n)
    h_enet = _consistency_hits(enet_n)

    # Compute BRS
    # Count active models for weight renormalization
    active_w = []
    if use_lgbm: active_w.append(("lgbm", w_lgbm))
    if use_enet: active_w.append(("enet", w_enet))
    if use_mlp: active_w.append(("mlp", w_mlp))

    if active_w:
        total_w = sum(w for _, w in active_w)
    else:
        total_w = 1.0

    scores = np.zeros(n_feat)
    for j in prescreen:
        strength = 0.0
        if use_lgbm:
            strength += (w_lgbm / total_w) * lgbm_n[j]
        if use_enet:
            strength += (w_enet / total_w) * enet_n[j]
        if use_mlp:
            strength += (w_mlp / total_w) * mlp_n[j]

        cons = 0.0
        if use_consistency:
            n_models = sum([use_mlp, use_lgbm, use_enet])
            if n_models > 0:
                cons = sum([
                    h_mlp.get(j, 0) if use_mlp else 0,
                    h_lgbm.get(j, 0) if use_lgbm else 0,
                    h_enet.get(j, 0) if use_enet else 0,
                ]) / n_models

        v = val_n.get(j, 0.0) if use_validity else 0.0

        # Renormalize alpha/beta/gamma if a term is disabled
        a, b, g = alpha, beta, gamma
        if not use_consistency:
            b = 0.0
        if not use_validity:
            g = 0.0
        total_abg = a + b + g
        if total_abg > 0:
            scores[j] = (a/total_abg) * strength + (b/total_abg) * cons + (g/total_abg) * v
        else:
            scores[j] = strength

    order = np.argsort(scores)[::-1]
    return [feat_names[j] for j in order], scores


# ====================== Baseline Methods ======================

def _std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def lasso_rank(X, y, feat_names, **kw):
    m = LassoCV(cv=5, max_iter=5000, random_state=42).fit(_std(X), y)
    scores = np.abs(m.coef_)
    return [feat_names[j] for j in np.argsort(scores)[::-1]], scores


def enet_rank(X, y, feat_names, **kw):
    m = ElasticNetCV(l1_ratio=[0.7, 0.85, 1.0], cv=5, max_iter=5000,
                     random_state=42).fit(_std(X), y)
    scores = np.abs(m.coef_)
    return [feat_names[j] for j in np.argsort(scores)[::-1]], scores


def rf_rank(X, y, feat_names, **kw):
    m = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    m.fit(X, y)
    scores = m.feature_importances_
    return [feat_names[j] for j in np.argsort(scores)[::-1]], scores


def lgbm_shap_rank(X, y, feat_names, **kw):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_data_in_leaf=3, random_state=42, verbose=-1)
    m.fit(X, y)
    if SHAP_OK:
        sv = shap.TreeExplainer(m).shap_values(X)
        scores = np.abs(sv).mean(0)
    else:
        scores = m.booster_.feature_importance(importance_type="gain").astype(float)
    return [feat_names[j] for j in np.argsort(scores)[::-1]], scores


def permutation_imp_rank(X, y, feat_names, **kw):
    """Permutation importance using RandomForest as base estimator."""
    m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    m.fit(X, y)
    result = permutation_importance(m, X, y, n_repeats=20, random_state=42,
                                   n_jobs=-1)
    scores = result.importances_mean
    scores = np.clip(scores, 0, None)  # clip negative to 0
    return [feat_names[j] for j in np.argsort(scores)[::-1]], scores


def mutual_info_rank(X, y, feat_names, **kw):
    """Mutual information regression (non-parametric, captures nonlinear)."""
    scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    return [feat_names[j] for j in np.argsort(scores)[::-1]], scores


ALL_METHODS = {
    "BRS": lambda X, y, fn, **kw: brs_univariate(X, y, fn),
    "LASSO": lasso_rank,
    "ElasticNet": enet_rank,
    "RandomForest": rf_rank,
    "LightGBM+SHAP": lgbm_shap_rank,
    "PermutationImp": permutation_imp_rank,
    "MutualInfo": mutual_info_rank,
}


# ====================== Metrics ======================

def recall_at_k(ranked, k, truth_set):
    if not isinstance(truth_set, set):
        truth_set = set(truth_set)
    return len(set(ranked[:k]) & truth_set) / len(truth_set)


def _pair_design(xi, xj):
    M = np.column_stack([xi, xj, xi * xj]).astype(float)
    return (M - M.mean(0)) / (M.std(0) + 1e-8)


def _triplet_design(xi, xj, xk):
    """Triplet design matrix with centered interaction terms + tanh saturation."""
    ci = xi - xi.mean()
    cj = xj - xj.mean()
    ck = xk - xk.mean()
    inter = np.column_stack([ci*cj, ci*ck, cj*ck, ci*cj*ck])
    inter = np.tanh(inter / TANH_SCALE)
    M = np.column_stack([xi, xj, xk, inter])
    return (M - M.mean(0)) / (M.std(0) + 1e-8)


def _oof_r2(Xp, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.zeros_like(y, dtype=float)
    for tr, va in kf.split(Xp):
        lr = LinearRegression().fit(Xp[tr], y[tr])
        oof[va] = lr.predict(Xp[va])
    return max(0.0, float(r2_score(y, oof)))


# ---- BRS pair scoring (full three-model ensemble) ----

def _brs_combo_raw(Xd, y, best_baseline_r2, n_boot, mlp_epochs, lgbm_est, folds):
    """Generic BRS combo scoring for pair or triplet design matrix."""
    d = Xd.shape[1]
    n = len(y)

    # (A) ElasticNet bootstrap stability
    sel_cnt = 0
    rng_b = np.random.RandomState(42)
    for b in range(n_boot):
        sub = rng_b.choice(n, size=max(3, int(n*0.7)), replace=False)
        try:
            m = ElasticNetCV(l1_ratio=[0.7, 0.85, 1.0], cv=3,
                             random_state=42+b, max_iter=2000)
            m.fit(Xd[sub], y[sub])
            if np.any(np.abs(m.coef_) > 1e-10):
                sel_cnt += 1
        except Exception:
            pass
    enet_raw = sel_cnt / max(n_boot, 1)

    # (B) LightGBM + SHAP
    lgbm_raw = 0.0
    try:
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        shap_sum = np.zeros(d)
        eff = 0
        for tr, va in kf.split(Xd):
            m = lgb.LGBMRegressor(n_estimators=lgbm_est, learning_rate=0.05,
                                  num_leaves=7, min_data_in_leaf=3,
                                  random_state=42, verbose=-1)
            m.fit(Xd[tr], y[tr], eval_set=[(Xd[va], y[va])],
                  eval_metric="l2",
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            if SHAP_OK:
                sv = shap.TreeExplainer(m).shap_values(Xd[va])
                if isinstance(sv, list):
                    sv = sv[0]
                shap_sum += np.mean(np.abs(sv), axis=0)
                eff += 1
            else:
                shap_sum += m.booster_.feature_importance(
                    importance_type="gain").astype(float)
                eff += 1
        if eff > 0:
            lgbm_raw = float(np.mean(shap_sum / eff))
    except Exception:
        pass

    # (C) MLP attention
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    att_coll = []
    for tr, va in kf.split(Xd):
        Xtr = torch.tensor(Xd[tr], dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.float32).view(-1, 1)
        Xva = torch.tensor(Xd[va], dtype=torch.float32)
        model = TinyComboMLP(d)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(mlp_epochs):
            opt.zero_grad()
            pred, _ = model(Xtr)
            loss = nn.functional.mse_loss(pred, ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            _, a = model(Xva)
            att_coll.append(a.mean(0).numpy())
    att = np.clip(np.mean(att_coll, axis=0), 0, None)
    lo, hi = att.min(), att.max()
    mlp_raw = float(((att - lo)/(hi - lo)).mean()) if hi > lo else 0.0

    # (D) Validity
    combo_r2 = _oof_r2(Xd, y, folds)
    validity_raw = max(0.0, combo_r2 - best_baseline_r2)

    return mlp_raw, lgbm_raw, enet_raw, validity_raw


def _brs_combo_rank(X, y, feat_names, ranked_uni, combos_true, combo_size,
                    top_k=TOP_UNI_FOR_COMBO):
    """
    Score all combos formed from top-k univariate features using full BRS.
    Returns rank of the true combo (1-indexed), or None if not in pool.
    """
    top = ranked_uni[:top_k]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    # Check if all members of true combo are in the top pool
    for member in combos_true:
        if member not in top:
            return None

    # Best univariate / pair baseline
    if combo_size == 2:
        best_base_r2 = 0.0
        for f in top:
            xi = X[:, name_to_idx[f]].astype(float)
            xi = (xi - xi.mean()) / (xi.std() + 1e-8)
            best_base_r2 = max(best_base_r2, _oof_r2(xi.reshape(-1, 1), y))
        n_boot = BRS_PAIR_ENET_BOOT
        mlp_ep = BRS_PAIR_MLP_EPOCHS
        lgbm_e = BRS_PAIR_LGBM_EST
        folds = BRS_PAIR_FOLDS
        design_fn = lambda i, j: _pair_design(X[:, i], X[:, j])
    else:  # triplet
        # Best pair R^2 as baseline
        best_base_r2 = 0.0
        pair_indices = list(combinations(range(len(top)), 2))
        # Sample if too many
        if len(pair_indices) > 100:
            pair_indices = list(RNG.choice(len(pair_indices), 100, replace=False))
            pair_indices = [list(combinations(range(len(top)), 2))[i] for i in pair_indices]
        for a, b in pair_indices:
            Xp = _pair_design(X[:, name_to_idx[top[a]]],
                              X[:, name_to_idx[top[b]]])
            best_base_r2 = max(best_base_r2, _oof_r2(Xp, y))
        n_boot = BRS_TRIPLET_ENET_BOOT
        mlp_ep = BRS_TRIPLET_MLP_EPOCHS
        lgbm_e = BRS_TRIPLET_LGBM_EST
        folds = BRS_PAIR_FOLDS
        design_fn = lambda i, j, k=None: None  # placeholder

    # Enumerate combos
    if combo_size == 2:
        combo_list = list(combinations(range(len(top)), 2))
    else:
        combo_list = list(combinations(range(len(top)), 3))
        if len(combo_list) > 500:
            sel = RNG.choice(len(combo_list), 500, replace=False)
            # Make sure true combo is included
            true_idx = None
            true_sorted = tuple(sorted(combos_true))
            for ci, combo in enumerate(combo_list):
                names_sorted = tuple(sorted(top[c] for c in combo))
                if names_sorted == true_sorted:
                    true_idx = ci
                    break
            sel = list(sel)
            if true_idx is not None and true_idx not in sel:
                sel[0] = true_idx
            combo_list = [combo_list[i] for i in sel]

    combos_named = []
    raws = []
    for combo in combo_list:
        indices = [name_to_idx[top[c]] for c in combo]
        names = tuple(sorted(top[c] for c in combo))
        combos_named.append(names)

        if combo_size == 2:
            Xd = _pair_design(X[:, indices[0]], X[:, indices[1]])
        else:
            Xd = _triplet_design(X[:, indices[0]], X[:, indices[1]],
                                 X[:, indices[2]])

        raw = _brs_combo_raw(Xd, y, best_base_r2, n_boot, mlp_ep, lgbm_e, folds)
        raws.append(raw)

    raws = np.array(raws)
    mlp01 = _minmax_vec(raws[:, 0])
    lgbm01 = _minmax_vec(raws[:, 1])
    enet01 = _minmax_vec(raws[:, 2])
    val01 = _minmax_vec(raws[:, 3])

    cons = (_consistency_hits(mlp01) + _consistency_hits(lgbm01)
            + _consistency_hits(enet01)) / 3.0
    strength = BRS_W_LGBM*lgbm01 + BRS_W_ENET*enet01 + BRS_W_MLP*mlp01
    brs = BRS_ALPHA*strength + BRS_BETA*cons + BRS_GAMMA*val01

    order = np.argsort(brs)[::-1]
    ranking = [combos_named[k] for k in order]

    true_sorted = tuple(sorted(combos_true))
    try:
        return ranking.index(true_sorted) + 1
    except ValueError:
        return None


# ---- Baseline combo scorers ----

def _baseline_combo_rank(X, y, feat_names, ranked_uni, combo_true,
                         combo_size, scorer_fn, top_k=TOP_UNI_FOR_COMBO):
    """Generic combo ranking for baseline methods using interaction importance."""
    top = ranked_uni[:top_k]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    for member in combo_true:
        if member not in top:
            return None

    combo_list = list(combinations(range(len(top)), combo_size))
    if len(combo_list) > 500:
        sel = list(RNG.choice(len(combo_list), 500, replace=False))
        true_sorted = tuple(sorted(combo_true))
        true_idx = None
        for ci, combo in enumerate(combo_list):
            if tuple(sorted(top[c] for c in combo)) == true_sorted:
                true_idx = ci
                break
        if true_idx is not None and true_idx not in sel:
            sel[0] = true_idx
        combo_list = [combo_list[i] for i in sel]

    scored = []
    for combo in combo_list:
        indices = [name_to_idx[top[c]] for c in combo]
        names = tuple(sorted(top[c] for c in combo))
        cols = [X[:, idx] for idx in indices]
        s = scorer_fn(cols, y)
        scored.append((names, s))

    scored.sort(key=lambda t: t[1], reverse=True)
    true_sorted = tuple(sorted(combo_true))
    try:
        return [p for p, _ in scored].index(true_sorted) + 1
    except ValueError:
        return None


def _pair_score_lasso(cols, y):
    Xd = _pair_design(cols[0], cols[1])
    m = LassoCV(cv=3, max_iter=3000, random_state=42).fit(Xd, y)
    return float(abs(m.coef_[2]))


def _pair_score_enet(cols, y):
    Xd = _pair_design(cols[0], cols[1])
    m = ElasticNetCV(l1_ratio=[0.7, 0.85, 1.0], cv=3, max_iter=3000,
                     random_state=42).fit(Xd, y)
    return float(abs(m.coef_[2]))


def _pair_score_rf(cols, y):
    Xd = _pair_design(cols[0], cols[1])
    m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    m.fit(Xd, y)
    return float(m.feature_importances_[2])


def _pair_score_lgbm(cols, y):
    Xd = _pair_design(cols[0], cols[1])
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7,
                          min_data_in_leaf=3, random_state=42, verbose=-1)
    m.fit(Xd, y)
    if SHAP_OK:
        sv = shap.TreeExplainer(m).shap_values(Xd)
        if isinstance(sv, list): sv = sv[0]
        return float(np.abs(sv).mean(0)[2])
    return float(m.booster_.feature_importance(importance_type="gain")[2])


def _triplet_score_rf(cols, y):
    Xd = _triplet_design(cols[0], cols[1], cols[2])
    m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    m.fit(Xd, y)
    # Use sum of interaction feature importances (cols 3-6)
    return float(m.feature_importances_[3:].sum())


def _triplet_score_lgbm(cols, y):
    Xd = _triplet_design(cols[0], cols[1], cols[2])
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7,
                          min_data_in_leaf=3, random_state=42, verbose=-1)
    m.fit(Xd, y)
    if SHAP_OK:
        sv = shap.TreeExplainer(m).shap_values(Xd)
        if isinstance(sv, list): sv = sv[0]
        return float(np.abs(sv).mean(0)[3:].sum())
    return float(m.booster_.feature_importance(importance_type="gain")[3:].sum())


BASELINE_PAIR_SCORERS = {
    "LASSO": _pair_score_lasso,
    "ElasticNet": _pair_score_enet,
    "RandomForest": _pair_score_rf,
    "LightGBM+SHAP": _pair_score_lgbm,
}

BASELINE_TRIPLET_SCORERS = {
    "RandomForest": _triplet_score_rf,
    "LightGBM+SHAP": _triplet_score_lgbm,
}


# ====================== Bootstrap Stability ======================

def bootstrap_stability(X, y, feat_names, method_fn, true_signals=None,
                        k=K_TOP, B=STABILITY_B):
    """
    Returns (jaccard_k, signal_recovery_rate).
    - jaccard_k: mean pairwise Jaccard of top-k lists across B bootstrap resamples
    - signal_recovery_rate: mean fraction of true_signals found in top-k
      across bootstrap resamples (None if true_signals not provided)
    """
    n = X.shape[0]
    top_sets = []
    for b in range(B):
        idx = RNG.choice(n, size=max(2, int(0.8*n)), replace=False)
        try:
            ranked, _ = method_fn(X[idx], y[idx], feat_names)
            top_sets.append(set(ranked[:k]))
        except Exception:
            top_sets.append(set())
    # Pairwise Jaccard
    j_vals = []
    for i in range(len(top_sets)):
        for j in range(i+1, len(top_sets)):
            a, b = top_sets[i], top_sets[j]
            if a or b:
                j_vals.append(len(a & b) / len(a | b))
    jaccard = float(np.mean(j_vals)) if j_vals else 0.0

    # Signal recovery rate
    srr = None
    if true_signals is not None:
        true_set = set(true_signals)
        recoveries = []
        for s in top_sets:
            if s:
                recoveries.append(len(s & true_set) / len(true_set))
        srr = float(np.mean(recoveries)) if recoveries else 0.0

    return jaccard, srr


# ====================== (A) Multi-Scenario Benchmark ======================

def run_multi_scenario():
    """Test BRS and baselines across varying data conditions."""
    print("=" * 70)
    print("(A) MULTI-SCENARIO SYNTHETIC BENCHMARK")
    print("=" * 70)

    scenarios = [
        # (label, n_samples, n_noise, noise_std, coef_scale)
        ("Default",           50,  195, 0.4,  1.0),
        ("Low noise",         50,  195, 0.2,  1.0),
        ("High noise",        50,  195, 0.8,  1.0),
        ("Very high noise",   50,  195, 1.2,  1.0),
        ("Small sample (30)", 30,  195, 0.4,  1.0),
        ("Large sample (100)",100, 195, 0.4,  1.0),
        ("Large sample (200)",200, 195, 0.4,  1.0),
        ("Many noise (500)",  50,  495, 0.4,  1.0),
        ("Many noise (1000)", 50,  995, 0.4,  1.0),
        ("Weak signal",       50,  195, 0.4,  0.5),
        ("Strong signal",     50,  195, 0.4,  2.0),
    ]

    methods_for_scenario = {
        "BRS": lambda X, y, fn: brs_univariate(X, y, fn),
        "LASSO": lasso_rank,
        "ElasticNet": enet_rank,
        "RandomForest": rf_rank,
        "LightGBM+SHAP": lgbm_shap_rank,
        "PermutationImp": permutation_imp_rank,
        "MutualInfo": mutual_info_rank,
    }

    rows = []
    for label, ns, nn_, nstd, cscale in scenarios:
        print(f"\n--- Scenario: {label} (n={ns}, noise_feats={nn_}, "
              f"noise_std={nstd}, signal_scale={cscale}) ---")

        X, y, fn, true_u, true_p, true_t = generate_synthetic(
            n_samples=ns, n_noise=nn_, noise_std=nstd,
            coef_lin=tuple(c*cscale for c in (1.8, -1.5, 1.3)),
            coef_nonlin=tuple(c*cscale for c in (2.2, 1.8)),
            coef_pair=2.4*cscale, coef_triplet=2.8*cscale,
            seed=42,
        )

        for mname, mfn in methods_for_scenario.items():
            ranked, _ = mfn(X, y, fn)
            r5 = recall_at_k(ranked, 5, true_u)
            r10 = recall_at_k(ranked, 10, true_u)
            r20 = recall_at_k(ranked, 20, true_u)
            print(f"  {mname:20s}  R@5={r5:.2f}  R@10={r10:.2f}  R@20={r20:.2f}")
            rows.append(dict(
                scenario=label, method=mname,
                n_samples=ns, n_noise=nn_, noise_std=nstd,
                signal_scale=cscale,
                recall_5=r5, recall_10=r10, recall_20=r20,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "scenario_table.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'scenario_table.csv'}")

    # --- Heatmap figure ---
    _plot_scenario_heatmap(df)
    return df


def _plot_scenario_heatmap(df):
    """Create heatmap: rows = scenarios, columns = methods, cell = Recall@10."""
    pivot = df.pivot_table(index="scenario", columns="method",
                           values="recall_10", aggfunc="first")
    # Reorder scenarios by the order they appear
    scenario_order = df["scenario"].unique()
    pivot = pivot.reindex(scenario_order)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            color = "white" if v > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Recall@10", shrink=0.8)
    ax.set_title("Univariate Feature Recovery (Recall@10) Across Scenarios",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scenario_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "scenario_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'scenario_heatmap.png'}")


# ====================== (B) Ablation Study ======================

def run_ablation():
    """Test BRS with each component removed."""
    print("\n" + "=" * 70)
    print("(B) ABLATION STUDY")
    print("=" * 70)

    X, y, fn, true_u, true_p, true_t = generate_synthetic()

    ablation_configs = {
        "BRS (full)":          dict(),
        "w/o MLP":             dict(use_mlp=False),
        "w/o LightGBM":        dict(use_lgbm=False),
        "w/o ElasticNet":      dict(use_enet=False),
        "w/o Consistency":     dict(use_consistency=False),
        "w/o Validity":        dict(use_validity=False),
        "Strength only":       dict(use_consistency=False, use_validity=False),
        "MLP only":            dict(use_lgbm=False, use_enet=False,
                                    use_consistency=False),
        "LightGBM only":       dict(use_mlp=False, use_enet=False,
                                    use_consistency=False),
        "ElasticNet only":     dict(use_mlp=False, use_lgbm=False,
                                    use_consistency=False),
    }

    rows = []
    for config_name, kwargs in ablation_configs.items():
        ranked, _ = brs_univariate(X, y, fn, **kwargs)
        r5 = recall_at_k(ranked, 5, true_u)
        r10 = recall_at_k(ranked, 10, true_u)
        r20 = recall_at_k(ranked, 20, true_u)
        print(f"  {config_name:25s}  R@5={r5:.2f}  R@10={r10:.2f}  R@20={r20:.2f}"
              f"  top5={ranked[:5]}")
        rows.append(dict(config=config_name, recall_5=r5, recall_10=r10,
                         recall_20=r20, top5="; ".join(ranked[:5])))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ablation_table.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'ablation_table.csv'}")

    # --- Ablation bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["recall_5"], w, label="Recall@5", color="#4C78A8")
    ax.bar(x, df["recall_10"], w, label="Recall@10", color="#F58518")
    ax.bar(x + w, df["recall_20"], w, label="Recall@20", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Recall of 5 planted features")
    ax.set_ylim(0, 1.15)
    ax.set_title("Ablation Study: Contribution of Each BRS Component",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on Recall@10 bars
    for i, v in enumerate(df["recall_10"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "ablation_figure.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "ablation_figure.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'ablation_figure.png'}")
    return df


# ====================== (C) Weight Sensitivity Analysis ======================

def run_sensitivity():
    """Grid search over BRS weight parameters and measure ranking stability."""
    print("\n" + "=" * 70)
    print("(C) WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)

    X, y, fn, true_u, true_p, true_t = generate_synthetic()

    # Grid over (alpha, beta, gamma) -- they sum to 1
    abg_grid = []
    for a in np.arange(0.1, 0.8, 0.1):
        for b in np.arange(0.1, 0.8, 0.1):
            g = round(1.0 - a - b, 2)
            if 0.05 <= g <= 0.85:
                abg_grid.append((round(a, 2), round(b, 2), g))

    # Also grid over model weights
    model_w_grid = [
        (0.1, 0.5, 0.4),   # default
        (0.33, 0.33, 0.34), # equal
        (0.5, 0.3, 0.2),
        (0.2, 0.2, 0.6),
        (0.1, 0.1, 0.8),   # MLP-heavy
        (0.1, 0.8, 0.1),   # ENet-heavy
        (0.8, 0.1, 0.1),   # LGBM-heavy
    ]

    # --- Part 1: BRS weight sensitivity (alpha, beta, gamma) ---
    print(f"\n  Testing {len(abg_grid)} (alpha, beta, gamma) combinations...")
    abg_rows = []
    all_top10_sets = []
    for a, b, g in abg_grid:
        ranked, _ = brs_univariate(X, y, fn, alpha=a, beta=b, gamma=g)
        r5 = recall_at_k(ranked, 5, true_u)
        r10 = recall_at_k(ranked, 10, true_u)
        top10_set = set(ranked[:10])
        all_top10_sets.append(top10_set)
        abg_rows.append(dict(alpha=a, beta=b, gamma=g,
                             recall_5=r5, recall_10=r10,
                             top5="; ".join(ranked[:5])))

    # Compute pairwise Jaccard between all top-10 sets
    jaccards = []
    for i in range(len(all_top10_sets)):
        for j in range(i+1, len(all_top10_sets)):
            s1, s2 = all_top10_sets[i], all_top10_sets[j]
            if s1 or s2:
                jaccards.append(len(s1 & s2) / len(s1 | s2))
    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

    df_abg = pd.DataFrame(abg_rows)
    print(f"  BRS weight (a,b,g) sensitivity:")
    print(f"    Recall@10  mean={df_abg['recall_10'].mean():.3f}  "
          f"std={df_abg['recall_10'].std():.3f}  "
          f"min={df_abg['recall_10'].min():.2f}  "
          f"max={df_abg['recall_10'].max():.2f}")
    print(f"    Mean pairwise Jaccard of top-10: {mean_jaccard:.3f}")

    # --- Part 2: Model weight sensitivity (w_lgbm, w_enet, w_mlp) ---
    print(f"\n  Testing {len(model_w_grid)} model weight combinations...")
    mw_rows = []
    for wl, we, wm in model_w_grid:
        ranked, _ = brs_univariate(X, y, fn, w_lgbm=wl, w_enet=we, w_mlp=wm)
        r5 = recall_at_k(ranked, 5, true_u)
        r10 = recall_at_k(ranked, 10, true_u)
        mw_rows.append(dict(w_lgbm=wl, w_enet=we, w_mlp=wm,
                            recall_5=r5, recall_10=r10,
                            top5="; ".join(ranked[:5])))
    df_mw = pd.DataFrame(mw_rows)
    print(f"  Model weight sensitivity:")
    print(f"    Recall@10  mean={df_mw['recall_10'].mean():.3f}  "
          f"std={df_mw['recall_10'].std():.3f}")

    # Save both tables
    df_all = pd.concat([
        df_abg.assign(weight_type="BRS (a,b,g)"),
        df_mw.assign(weight_type="Model (wL,wE,wM)"),
    ], ignore_index=True)
    df_all.to_csv(OUT_DIR / "sensitivity_table.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'sensitivity_table.csv'}")

    # --- Figure: scatter of Recall@10 vs gamma ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    ax = axes[0]
    ax.scatter(df_abg["gamma"], df_abg["recall_10"], c="#4C78A8", alpha=0.6, s=30)
    ax.axhline(y=df_abg["recall_10"].mean(), color="gray", ls="--", lw=1,
               label=f"mean={df_abg['recall_10'].mean():.2f}")
    # Mark the default (0.2, 0.2, 0.6)
    default_row = df_abg[(df_abg["alpha"] == 0.2) & (df_abg["beta"] == 0.2)]
    if len(default_row) > 0:
        ax.scatter(default_row["gamma"], default_row["recall_10"],
                   c="red", s=100, zorder=5, marker="*",
                   label="Default (0.2, 0.2, 0.6)")
    ax.set_xlabel("γ (Validity weight)")
    ax.set_ylabel("Recall@10")
    ax.set_title("(a) BRS weight sensitivity (α, β, γ)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.08)

    ax = axes[1]
    labels = [f"({r['w_lgbm']},{r['w_enet']},{r['w_mlp']})"
              for _, r in df_mw.iterrows()]
    x = np.arange(len(df_mw))
    bars = ax.bar(x, df_mw["recall_10"],
                  color=["#E45756" if i == 0 else "#4C78A8"
                         for i in range(len(df_mw))])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Recall@10")
    ax.set_title("(b) Model weight sensitivity (wL, wE, wM)")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(df_mw["recall_10"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "sensitivity_figure.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "sensitivity_figure.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'sensitivity_figure.png'}")

    return df_all, mean_jaccard


# ====================== (D+E) Full Benchmark with Combo Recovery ======================

def run_full_benchmark():
    """
    Complete benchmark on default scenario:
    - All methods: univariate Recall@5/10/20
    - Pair recovery rank
    - Triplet recovery rank
    - Bootstrap stability
    """
    print("\n" + "=" * 70)
    print("(D+E) FULL BENCHMARK: UNIVARIATE + PAIR + TRIPLET RECOVERY")
    print("=" * 70)

    X, y, fn, true_u, true_p, true_t = generate_synthetic()

    methods = {
        "BRS": lambda X, y, fn: brs_univariate(X, y, fn),
        "LASSO": lasso_rank,
        "ElasticNet": enet_rank,
        "RandomForest": rf_rank,
        "LightGBM+SHAP": lgbm_shap_rank,
        "PermutationImp": permutation_imp_rank,
        "MutualInfo": mutual_info_rank,
    }

    rows = []
    for mname, mfn in methods.items():
        print(f"\n[{mname}]")
        t0 = time.time()
        ranked, _ = mfn(X, y, fn)
        r5 = recall_at_k(ranked, 5, true_u)
        r10 = recall_at_k(ranked, 10, true_u)
        r20 = recall_at_k(ranked, 20, true_u)
        print(f"  Recall@5/10/20 = {r5:.2f} / {r10:.2f} / {r20:.2f}")
        print(f"  top-5: {ranked[:5]}")

        # Pair recovery
        if mname == "BRS":
            pr = _brs_combo_rank(X, y, fn, ranked, true_p, combo_size=2)
        elif mname in BASELINE_PAIR_SCORERS:
            pr = _baseline_combo_rank(X, y, fn, ranked, true_p, 2,
                                      BASELINE_PAIR_SCORERS[mname])
        else:
            pr = None
        print(f"  Pair rank: {pr if pr else 'n/a'}")

        # Triplet recovery
        if mname == "BRS":
            tr = _brs_combo_rank(X, y, fn, ranked, true_t, combo_size=3)
        elif mname in BASELINE_TRIPLET_SCORERS:
            tr = _baseline_combo_rank(X, y, fn, ranked, true_t, 3,
                                      BASELINE_TRIPLET_SCORERS[mname])
        else:
            tr = None
        print(f"  Triplet rank: {tr if tr else 'n/a'}")

        # Stability (top-5 Jaccard + signal recovery rate)
        print(f"  Stability...", end=" ", flush=True)
        jac5, srr = bootstrap_stability(X, y, fn, mfn,
                                        true_signals=true_u, k=5)
        print(f"Jaccard@5={jac5:.3f}  SignalRecovery={srr:.3f}")

        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)")

        rows.append(dict(
            method=mname,
            recall_5=r5, recall_10=r10, recall_20=r20,
            pair_rank=pr if pr else np.nan,
            triplet_rank=tr if tr else np.nan,
            stability_jaccard5=jac5,
            signal_recovery=srr,
            top5="; ".join(ranked[:5]),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "full_benchmark_table.csv", index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nSaved: {OUT_DIR / 'full_benchmark_table.csv'}")

    # --- Figure: 4-panel benchmark ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    method_names = df["method"].tolist()
    n_m = len(method_names)
    x = np.arange(n_m)
    colors_list = [COLORS.get(m, "#999999") for m in method_names]

    # (a) Univariate recall
    ax = axes[0, 0]
    w = 0.25
    ax.bar(x - w, df["recall_5"], w, label="Recall@5", color="#4C78A8")
    ax.bar(x, df["recall_10"], w, label="Recall@10", color="#F58518")
    ax.bar(x + w, df["recall_20"], w, label="Recall@20", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) Univariate feature recovery")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    # (b) Pair rank
    ax = axes[0, 1]
    pr_vals = df["pair_rank"].fillna(999).values
    pr_capped = np.minimum(pr_vals, 200)
    bars = ax.bar(method_names, pr_capped, color=colors_list)
    for i, (bar, raw) in enumerate(zip(bars, pr_vals)):
        label = "n/a" if raw >= 999 else str(int(raw))
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                label, ha="center", fontsize=9)
    ax.set_ylabel("Rank of true pair (lower = better)")
    ax.set_title("(b) Synergistic pair recovery")
    ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (c) Triplet rank
    ax = axes[1, 0]
    tr_vals = df["triplet_rank"].fillna(999).values
    tr_capped = np.minimum(tr_vals, 200)
    bars = ax.bar(method_names, tr_capped, color=colors_list)
    for i, (bar, raw) in enumerate(zip(bars, tr_vals)):
        label = "n/a" if raw >= 999 else str(int(raw))
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                label, ha="center", fontsize=9)
    ax.set_ylabel("Rank of true triplet (lower = better)")
    ax.set_title("(c) Synergistic triplet recovery")
    ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (d) Signal Recovery Rate
    ax = axes[1, 1]
    ax.bar(method_names, df["signal_recovery"], color=colors_list)
    ax.set_ylabel("Signal Recovery Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"(d) Bootstrap signal recovery (B={STABILITY_B}, top-5)")
    ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(df["signal_recovery"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.suptitle("Comprehensive Benchmark: BRS vs. Baselines on Synthetic Data",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "full_benchmark_figure.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "full_benchmark_figure.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'full_benchmark_figure.png'}")
    return df


# ====================== Main ======================

def main():
    t_total = time.time()

    # (A) Multi-scenario
    df_scenario = run_multi_scenario()

    # (B) Ablation
    df_ablation = run_ablation()

    # (C) Weight sensitivity
    df_sens, mean_jac = run_sensitivity()

    # (D+E) Full benchmark with pair + triplet
    df_full = run_full_benchmark()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nMulti-scenario: {len(df_scenario)} rows "
          f"({df_scenario['scenario'].nunique()} scenarios x "
          f"{df_scenario['method'].nunique()} methods)")
    print(f"Ablation: {len(df_ablation)} configurations tested")
    print(f"Sensitivity: mean pairwise Jaccard across weight grid = {mean_jac:.3f}")
    print(f"\nFull benchmark (default scenario):")
    print(df_full[["method", "recall_5", "recall_10", "recall_20",
                    "pair_rank", "triplet_rank", "stability_jaccard5",
                    "signal_recovery"]].to_string(index=False))

    print(f"\nTotal elapsed: {time.time() - t_total:.1f}s")
    print(f"All outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
