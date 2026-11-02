# -*- coding: utf-8 -*-
"""
Pipeline-level benchmark on the wheat panel (Supplementary Table S21, sheet 'wheat_single'): does stability selection
with BRS give more reproducible single-metabolite lists than stability selection with simpler scorers, and what does
stability selection add over a single run?

Scorers (per strain, on the same prescreen as BRS: Pearson r > 0.2, at most 100 metabolites, log-scale input):
    Pearson      top 10 by r
    Elastic Net  ElasticNetCV as in BRS (l1_ratio 0.7/0.85/1.0, 3-fold CV) on the standardized prescreened
                 metabolites; top 10 by |coefficient| among the non-zero coefficients
    BRS          top 10 by BRS (existing runs; not repeated)
All scorers use the same half-samples: the 50 complementary half-samples of the full-data stability selection
(results_stability_uni/) and the 20 training half-samples of every outer fold of the nested cross-validation
(results_nestedcv/). The consensus list is defined as for BRS (max-strain pi >= 0.4; fallback top 14 by pi), the
single-run comparator is the top 14 of one run on the training set (max over strains), and lists are scored with
the composite score of the paper (nested-CV AUC against the high-BNI label).

Output: supplementary/ST_pipeline_benchmark.xlsx (sheet wheat_single)
"""
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ROOT = Path(__file__).resolve().parent
BRS_INPUT = ROOT / "Data Preprocessing" / "filtered_metabolites_log.xlsx"
STAB = ROOT / "results_stability_uni"
NCV = ROOT / "results_nestedcv"
FULL_BRS = ROOT / "results_v3_log" / "univariate_BRS.xlsx"
OUT_XLSX = ROOT / "supplementary" / "ST_pipeline_benchmark.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
TARGETS = {s: f"Inhibition (%)-{s}" for s in STRAINS}
R_MIN, CAP, TOP_K, PI_THR, FALLBACK_N, SEED = 0.2, 100, 10, 0.4, 14, 42
N_SPLITS, N_REPEATS, Q = 5, 2, 0.75

# ==========================================
# 1) Data
# ==========================================
raw = pd.read_excel(BRS_INPUT, sheet_name="Sheet1", decimal=",").dropna(how="all")
feat = [c for c in raw.columns if c not in TARGETS.values()]
XM = raw[feat].apply(pd.to_numeric, errors="coerce").values.astype(float)
FEAT = [str(c).strip() for c in feat]
YM = {s: pd.to_numeric(raw[TARGETS[s]], errors="coerce").values.astype(float) for s in STRAINS}
n = len(raw)

# composite score and labels exactly as in nested_cv_signature.py
lab = pd.read_excel(ROOT / "Data Preprocessing" / "filtered_metabolites.xlsx", sheet_name=0)
bni = lab[[[c for c in lab.columns if "Inhibition" in str(c) and str(c).endswith(s)][0] for s in STRAINS]]
bni = bni.apply(pd.to_numeric, errors="coerce").set_axis(STRAINS, axis=1)
Y_ALL = (bni >= bni.quantile(Q)).any(axis=1).astype(int).values
_raw = pd.read_excel(ROOT / "Data Preprocessing" / "filtered_notfound_removed.xlsx", sheet_name=0, decimal=",")
_raw = _raw[[c for c in lab.columns if "Inhibition" not in str(c)]].apply(pd.to_numeric, errors="coerce")
_raw = _raw.apply(lambda s: s.where(s != 0, s[s != 0].min() / 2))
Z_ALL = np.log1p(_raw)
Z_ALL = (Z_ALL - Z_ALL.mean()) / Z_ALL.std()
Z_ALL.columns = [str(c).strip() for c in Z_ALL.columns]
assert len(Z_ALL) == n


def auc(sc, yy):
    r = rankdata(sc)
    n1 = yy.sum()
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))


def prescreen(idx, s):
    X, y = XM[idx], YM[s][idx]
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    Xc, yc = X - np.nanmean(X, 0), y - y.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nansum(Xc * yc[:, None], 0) / np.sqrt(np.nansum(Xc ** 2, 0) * (yc ** 2).sum())
    r = np.where(np.isfinite(r), r, -np.inf)
    keep = [j for j in np.argsort(-r) if r[j] > R_MIN][:CAP]
    return keep, r, X, y


def score(idx, s, scorer):
    """Ranked metabolite keys (best first) for one strain on the samples idx."""
    keep, r, X, y = prescreen(idx, s)
    if scorer == "Pearson" or not keep:
        return [FEAT[j] for j in keep]
    Xs = StandardScaler().fit_transform(np.nan_to_num(X[:, keep], nan=np.nanmean(X[:, keep])))
    coef = np.abs(ElasticNetCV(l1_ratio=[0.7, 0.85, 1.0], cv=3, random_state=SEED, max_iter=2000).fit(Xs, y).coef_)
    order = sorted(range(len(keep)), key=lambda i: (-coef[i], -r[keep[i]]))
    return [FEAT[keep[i]] for i in order if coef[i] > 0]


def brs_ranked(xlsx):
    x = pd.ExcelFile(xlsx)
    return {s: list(pd.read_excel(x, f"Inhibition (pct)-{s}_Uni_All").sort_values("BRS", ascending=False)["Feature"]
                    .astype(str).str.strip()) for s in STRAINS}


def pi_from(rankings):
    """rankings: list over half-samples of {strain: ranked keys}."""
    cnt = {}
    for rk in rankings:
        for s in STRAINS:
            for m in rk[s][:TOP_K]:
                cnt.setdefault(m, dict.fromkeys(STRAINS, 0))[s] += 1
    return pd.DataFrame(cnt).T.reindex(columns=STRAINS).fillna(0) / len(rankings)


def rankings_for(scorer, dirs, idx_list):
    if scorer == "BRS":
        return [brs_ranked(d / "univariate_BRS.xlsx") for d in dirs]
    return [{s: score(idx, s, scorer) for s in STRAINS} for idx in idx_list]


def single_run_order(rk, scorer, brs_xlsx=None):
    """Full ranking of one run, combining strains by the best score (BRS) or the best rank (Pearson, Elastic Net)."""
    if scorer == "BRS":
        x = pd.ExcelFile(brs_xlsx)
        best = pd.concat([pd.read_excel(x, f"Inhibition (pct)-{s}_Uni_All").assign(
            Feature=lambda d: d["Feature"].astype(str).str.strip()).set_index("Feature")["BRS"] for s in STRAINS], axis=1)
        return list(best.max(axis=1).sort_values(ascending=False).index)
    best = {}
    for s in STRAINS:
        for i, m in enumerate(rk[s]):
            best[m] = min(best.get(m, np.inf), i)
    return sorted(best, key=best.get)


def single_run_top(rk, scorer, brs_xlsx=None):
    return single_run_order(rk, scorer, brs_xlsx)[:FALLBACK_N]


def jaccard_mean(lists):
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in combinations(lists, 2)]))


SCORERS = ["Pearson", "Elastic Net", "BRS"]

# ==========================================
# 2) Full data: 50 half-samples
# ==========================================
subs = sorted(STAB.glob("sub_*"))
idx_subs = [np.loadtxt(d / "samples.txt", dtype=int) for d in subs]
full_rows = []
for sc in SCORERS:
    rk = rankings_for(sc, subs, idx_subs)
    pi = pi_from(rk)
    full_run = brs_ranked(FULL_BRS) if sc == "BRS" else {s: score(np.arange(n), s, sc) for s in STRAINS}
    not_repro = sum(sum(pi[s].get(m, 0) < PI_THR for m in full_run[s][:TOP_K]) for s in STRAINS)
    half = len(rk) // 2 // 2 * 2            # split by complementary pairs: first half of the pairs vs the rest
    pa, pb = pi_from(rk[:half]), pi_from(rk[half:])
    rho = []
    for s in STRAINS:
        u = sorted(set(pa.index[pa[s] > 0]) | set(pb.index[pb[s] > 0]))
        rho.append(spearmanr(pa[s].reindex(u).fillna(0), pb[s].reindex(u).fillna(0)).correlation)
    rec = {"scorer": sc + " + stability selection", "selected (pi >= 0.4), all strains": int((pi >= PI_THR).sum().sum())}
    rec.update({f"selected {s}": int((pi[s] >= PI_THR).sum()) for s in STRAINS})
    rec.update({f"max pi {s}": float(pi[s].max()) for s in STRAINS})
    rec["single-run top 10 not reproducible (of 40)"] = int(not_repro)
    rec["pi agreement between two disjoint sets of half-samples (mean Spearman over strains)"] = float(np.nanmean(rho))
    full_rows.append(rec)
    print(sc, "full data done")

# ==========================================
# 3) Nested cross-validation (same outer folds and training half-samples)
# ==========================================
outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
             .split(np.zeros(n), Y_ALL))
ncv_rows = []
for sc in SCORERS:
    oof_ss, oof_single = np.full((N_REPEATS, n), np.nan), np.full((N_REPEATS, n), np.nan)
    oof_ss14 = np.full((N_REPEATS, n), np.nan)
    lists_ss, lists_single, lists_ss14 = [], [], []
    for f, (tr, te) in enumerate(outer):
        fd = NCV / f"fold_{f:02d}"
        halves = sorted(fd.glob("half_*"))
        rk = rankings_for(sc, halves, [np.loadtxt(h / "samples.txt", dtype=int) for h in halves])
        pm = pi_from(rk).max(axis=1).sort_values(ascending=False)
        m = [x for x in pm.index[pm >= PI_THR] if x in Z_ALL.columns] or \
            [x for x in pm.index if x in Z_ALL.columns][:FALLBACK_N]
        full_tr = None if sc == "BRS" else {s: score(tr, s, sc) for s in STRAINS}
        order1 = [x for x in single_run_order(full_tr, sc, fd / "full" / "univariate_BRS.xlsx") if x in Z_ALL.columns]
        m1 = order1[:FALLBACK_N]
        # fixed-size consensus: top 14 by pi, ties broken by the single-run order of the training set
        pos = {x: i for i, x in enumerate(order1)}
        m14 = sorted([x for x in pm.index if x in Z_ALL.columns and pm[x] > 0],
                     key=lambda x: (-pm[x], pos.get(x, np.inf)))[:FALLBACK_N]
        rep = f // N_SPLITS
        oof_ss[rep, te] = Z_ALL[m].mean(axis=1).values[te]
        oof_single[rep, te] = Z_ALL[m1].mean(axis=1).values[te]
        oof_ss14[rep, te] = Z_ALL[m14].mean(axis=1).values[te]
        lists_ss.append(m)
        lists_single.append(m1)
        lists_ss14.append(m14)
    for name, oof, lists in ((sc + " + stability selection (pi >= 0.4)", oof_ss, lists_ss),
                             (sc + " + stability selection (top 14 by pi)", oof_ss14, lists_ss14),
                             (sc + ", single run (top 14)", oof_single, lists_single)):
        ncv_rows.append({"method": name, "nested-CV AUC": float(np.mean([auc(oof[r], Y_ALL) for r in range(N_REPEATS)])),
                         "list size (mean over folds)": float(np.mean([len(x) for x in lists])),
                         "membership Jaccard across the 10 outer folds": jaccard_mean(lists)})
    print(sc, "nested CV done")

F, N = pd.DataFrame(full_rows), pd.DataFrame(ncv_rows)
readme = pd.DataFrame([
    ("full_data", "50 complementary half-samples of all 44 accessions (results_stability_uni); pi = fraction of "
                  "half-samples in which a metabolite is in the strain's top 10."),
    ("nested_cv", "10 outer folds of nested_cv_signature.py; lists are built on the training accessions only "
                  "(20 half-samples) and scored on the held-out accessions with the composite score."),
    ("Elastic Net", "only metabolites with a non-zero coefficient count as selected (at most 10 per half-sample)."),
], columns=["item", "description"])
mode = "a" if OUT_XLSX.exists() else "w"
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", mode=mode, **({"if_sheet_exists": "replace"} if mode == "a" else {})) as w:
    readme.to_excel(w, sheet_name="README_wheat", index=False)
    F.round(3).to_excel(w, sheet_name="wheat_full_data", index=False)
    N.round(3).to_excel(w, sheet_name="wheat_nested_cv", index=False)
pd.set_option("display.width", 250)
print(F.round(3).T.to_string(), "\n")
print(N.round(3).to_string(index=False))
print("Saved", OUT_XLSX)
