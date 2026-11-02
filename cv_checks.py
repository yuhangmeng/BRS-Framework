# -*- coding: utf-8 -*-
"""
Checks of the nested cross-validation (Supplementary Table S20). Uses only the outputs of nested_cv_signature.py
(results_nestedcv/: outer folds, 20 training half-samples each with their sample lists and BRS tables) and of the
full-data stability selection (results_stability_uni/); no BRS run is repeated.

    mixture_size  Step 3 chooses the recommended mixture size k on the nested-CV curve (incremental_signature.py),
                  so the AUC at that k is optimistic. Here k is also chosen inside each outer training set: the
                  disjoint training half-samples are paired, metabolites are ranked by BRS on one half and the
                  top-k composite is scored on the other (both directions, all pairs); k is the smallest size within
                  0.02 of the inner maximum, and the outer test accessions are scored with the pi order of the full
                  training set at that k. A pre-specified rule (k = number of metabolites with training pi >= 0.4)
                  is reported as well.
    scaling       composite scores with half-minimum imputation and autoscaling computed on all 44 accessions (as in
                  the paper) or on the training accessions only.
    thresholds    size and nested-CV AUC of the consensus list for selection thresholds 0.3-0.6.

Output: supplementary/ST_cv_checks.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parent
NCV = ROOT / "results_nestedcv"
STAB_XLSX = ROOT / "results_stability_uni" / "stability_summary_uni.xlsx"
OUT_XLSX = ROOT / "supplementary" / "ST_cv_checks.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
N_SPLITS, N_REPEATS, SEED, TOP_K, Q = 5, 2, 42, 10, 0.75     # as in nested_cv_signature.py
K_MAX = {"overall": 30, "NF": 15, "NV": 15, "NM": 15, "NU": 15}  # as in incremental_signature.py
PARSIMONY, PI_THR, FALLBACK_N = 0.02, 0.4, 14
THRESHOLDS = [0.3, 0.4, 0.5, 0.6]

# ==========================================
# 1) Data and labels (as in nested_cv_signature.py)
# ==========================================
lab = pd.read_excel(ROOT / "Data Preprocessing" / "filtered_metabolites.xlsx", sheet_name=0)
bcols = {s: [c for c in lab.columns if "Inhibition" in str(c) and str(c).endswith(s)][0] for s in STRAINS}
bni = lab[[bcols[s] for s in STRAINS]].apply(pd.to_numeric, errors="coerce").set_axis(STRAINS, axis=1)
Y = {"overall": (bni >= bni.quantile(Q)).any(axis=1).astype(int).values}
Y.update({s: (bni[s] >= bni[s].quantile(Q)).astype(int).values for s in STRAINS})
COLS = {"overall": STRAINS, **{s: [s] for s in STRAINS}}
met_cols = [c for c in lab.columns if "Inhibition" not in str(c)]
RAW = pd.read_excel(ROOT / "Data Preprocessing" / "filtered_notfound_removed.xlsx", sheet_name=0,
                    decimal=",")[met_cols].apply(pd.to_numeric, errors="coerce")
RAW.columns = [str(c).strip() for c in RAW.columns]
n = len(RAW)


def zscore(train_idx):
    """Half-minimum imputation, log1p and autoscaling with all constants taken from train_idx."""
    Xt = RAW.iloc[train_idx]
    X = RAW.where(RAW != 0, Xt.where(Xt != 0).min() / 2, axis=1)
    L = np.log1p(X)
    return (L - L.iloc[train_idx].mean()) / L.iloc[train_idx].std()


Z_ALL = zscore(np.arange(n))


def auc(score, yy):
    r = rankdata(score)
    n1 = yy.sum()
    n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pooled_auc(oof, yy):
    return float(np.mean([auc(oof[r], yy) for r in range(N_REPEATS)]))


def brs_table(xlsx):
    x = pd.ExcelFile(xlsx)
    return pd.concat([pd.read_excel(x, f"Inhibition (pct)-{s}_Uni_All").assign(
        Feature=lambda d: d["Feature"].astype(str).str.strip()).set_index("Feature")["BRS"].rename(s)
        for s in STRAINS], axis=1)


def order_from(pi, brs, cols):
    """As in incremental_signature.py: metabolites with pi > 0 by max pi over cols, ties by BRS."""
    p = pi[cols].max(axis=1)
    b = brs.reindex(pi.index)[cols].max(axis=1).fillna(0)
    d = pd.DataFrame({"p": p, "b": b})
    d = d[(d.p > 0) & d.index.isin(Z_ALL.columns)]
    return list(d.sort_values(["p", "b"], ascending=False, kind="mergesort").index)


# ==========================================
# 2) Outer folds
# ==========================================
outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
             .split(np.zeros(n), Y["overall"]))
folds = []
for f, (tr, te) in enumerate(outer):
    fd = NCV / f"fold_{f:02d}"
    cnt, hs = {}, []
    for h in sorted(fd.glob("half_*")):
        idx = np.loadtxt(h / "samples.txt", dtype=int)
        b = brs_table(h / "univariate_BRS.xlsx")
        for s in STRAINS:
            for m in b[s].dropna().sort_values(ascending=False).index[:TOP_K]:
                cnt.setdefault(m, dict.fromkeys(STRAINS, 0))[s] += 1
        hs.append((idx, b))
    pairs = []
    for i in range(0, len(hs), 2):   # half_{2p} and half_{2p+1} are disjoint halves of the training set
        assert not set(hs[i][0]) & set(hs[i + 1][0]) and set(hs[i][0]) | set(hs[i + 1][0]) <= set(tr)
        pairs.append((hs[i], hs[i + 1]))
    folds.append(dict(tr=tr, te=te, rep=f // N_SPLITS, pairs=pairs, Z_tr=zscore(tr),
                      pi=pd.DataFrame(cnt).T.reindex(columns=STRAINS).fillna(0) / len(hs),
                      brs=brs_table(fd / "full" / "univariate_BRS.xlsx")))


def inner_k(F, target):
    yy, cols, kmax = Y[target], COLS[target], K_MAX[target]
    curves = []
    for (ia, ba), (ib, bb) in F["pairs"]:
        for (i_fit, b_fit), i_val in (((ia, ba), ib), ((ib, bb), ia)):
            if yy[i_val].sum() in (0, len(i_val)):
                continue
            o = [m for m in b_fit[cols].max(axis=1).dropna().sort_values(ascending=False).index if m in Z_ALL.columns]
            Zt = zscore(i_fit)
            curves.append([auc(Zt[o[:k]].mean(axis=1).values[i_val], yy[i_val]) for k in range(1, kmax + 1)])
    c = np.nanmean(np.array(curves), axis=0)
    return int(np.where(c >= c.max() - PARSIMONY)[0].min()) + 1


# ==========================================
# 3) Mixture size and scaling
# ==========================================
size_rows, scale_rows = [], []
for target in ["overall"] + STRAINS:
    yy, cols, kmax = Y[target], COLS[target], K_MAX[target]
    blank = lambda: np.full((N_REPEATS, n), np.nan)
    by_k = {(k, sc): blank() for k in range(1, kmax + 1) for sc in ("all", "train")}
    inner, rule = {"all": blank(), "train": blank()}, {"all": blank(), "train": blank()}
    ks_inner, ks_rule, empty = [], [], 0
    for F in folds:
        o, te, r = order_from(F["pi"], F["brs"], cols), F["te"], F["rep"]
        Zs = {"all": Z_ALL, "train": F["Z_tr"]}
        for k in range(1, kmax + 1):
            for sc, Zx in Zs.items():
                by_k[(k, sc)][r, te] = Zx[o[:k]].mean(axis=1).values[te]
        ki = inner_k(F, target)
        kr = int((F["pi"][cols].max(axis=1) >= PI_THR).sum())
        empty += kr == 0
        kr = max(kr, 1)
        ks_inner.append(ki)
        ks_rule.append(kr)
        for sc, Zx in Zs.items():
            inner[sc][r, te] = Zx[o[:ki]].mean(axis=1).values[te]
            rule[sc][r, te] = Zx[o[:kr]].mean(axis=1).values[te]
    curve = {sc: np.array([pooled_auc(by_k[(k, sc)], yy) for k in range(1, kmax + 1)]) for sc in ("all", "train")}
    k_out = int(np.where(curve["all"] >= curve["all"].max() - PARSIMONY)[0].min()) + 1
    size_rows.append({
        "target": target, "AUC first metabolite (k = 1)": curve["all"][0],
        "recommended k (chosen on the nested-CV curve)": k_out, "AUC at recommended k (optimistic)": curve["all"][k_out - 1],
        "k chosen in the inner loop: min": min(ks_inner), "median": float(np.median(ks_inner)), "max": max(ks_inner),
        "AUC with k chosen in the inner loop": pooled_auc(inner["all"], yy),
        "rule k = #(training pi >= 0.4): min": min(ks_rule), "rule: max": max(ks_rule),
        "folds without pi >= 0.4 (k set to 1)": empty, "AUC with the rule": pooled_auc(rule["all"], yy)})
    scale_rows += [{"set": f"{target}: recommended k = {k_out}", "AUC, constants from all accessions": curve["all"][k_out - 1],
                    "AUC, constants from training accessions": curve["train"][k_out - 1]},
                   {"set": f"{target}: k chosen in the inner loop", "AUC, constants from all accessions": pooled_auc(inner["all"], yy),
                    "AUC, constants from training accessions": pooled_auc(inner["train"], yy)}]

# ==========================================
# 4) Consensus list: scaling and selection threshold
# ==========================================
pis = pd.read_excel(STAB_XLSX, sheet_name="pi_single")
p_full = pis.set_index(pis["key"].astype(str).str.strip())[[f"pi_{s}" for s in STRAINS]].max(axis=1)
thr_rows = []
for thr in THRESHOLDS:
    oof = {"all": np.full((N_REPEATS, n), np.nan), "train": np.full((N_REPEATS, n), np.nan)}
    sizes, empty = [], 0
    for F in folds:
        pm = F["pi"].max(axis=1).sort_values(ascending=False)
        m = [x for x in pm.index[pm >= thr] if x in Z_ALL.columns]
        if not m:                                    # fallback as in nested_cv_signature.py
            empty += 1
            m = [x for x in pm.index if x in Z_ALL.columns][:FALLBACK_N]
        sizes.append(len(m))
        oof["all"][F["rep"], F["te"]] = Z_ALL[m].mean(axis=1).values[F["te"]]
        oof["train"][F["rep"], F["te"]] = F["Z_tr"][m].mean(axis=1).values[F["te"]]
    thr_rows.append({"selection threshold": thr, "metabolites selected on all accessions": int((p_full >= thr).sum()),
                     "list size per outer fold: min": min(sizes), "max": max(sizes), "folds without a metabolite": empty,
                     "nested-CV AUC": pooled_auc(oof["all"], Y["overall"]),
                     "nested-CV AUC, constants from training accessions": pooled_auc(oof["train"], Y["overall"])})
    if thr == PI_THR:
        scale_rows.insert(0, {"set": "consensus list (pi >= 0.4)", "AUC, constants from all accessions": pooled_auc(oof["all"], Y["overall"]),
                              "AUC, constants from training accessions": pooled_auc(oof["train"], Y["overall"])})

S, C, T = pd.DataFrame(size_rows), pd.DataFrame(scale_rows), pd.DataFrame(thr_rows)
readme = pd.DataFrame([
    ("mixture_size", "AUC at the recommended k is read off the same nested-CV curve that chose k and is optimistic; "
                     "'k chosen in the inner loop' selects k from the training half-samples only (rank on one half by BRS, "
                     "score on the other) and is the unbiased estimate for the size-selection procedure."),
    ("scaling", "half-minimum imputation and autoscaling of the composite score from all 44 accessions (paper) or "
                "from the training accessions of each outer fold; BRS itself standardizes within each training set."),
    ("thresholds", "consensus list = metabolites with training-fold max-strain pi >= threshold (fallback: top 14 by pi)."),
    ("labels", "high/low labels (75th percentile of AOI%) are a fixed endpoint defined on all accessions; they are used "
               "only for evaluation, since BRS and stability selection use the continuous AOI%."),
], columns=["sheet", "description"])
OUT_XLSX.parent.mkdir(exist_ok=True)
with pd.ExcelWriter(OUT_XLSX) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    S.round(3).to_excel(w, sheet_name="mixture_size", index=False)
    C.round(3).to_excel(w, sheet_name="scaling", index=False)
    T.round(3).to_excel(w, sheet_name="thresholds", index=False)
pd.set_option("display.width", 250)
for d in (S, C, T):
    print(d.round(3).to_string(index=False), "\n")
print("Saved", OUT_XLSX)
