# -*- coding: utf-8 -*-
"""
Incremental (nested) metabolite signatures: add metabolites one at a time in order of stability-selection
probability and track the AUC of the composite score, overall and for each strain.

Ordering
    overall      : pi_max (largest selection probability over the four strains)
    strain c     : pi_c
    ties are broken by the full-data (or training-set) BRS for the same strain(s).
Evaluation
    in-sample    : order from the 50 half-samples on all 44 accessions, AUC on the same accessions
    nested CV    : in every outer fold of results_nestedcv/ the order is recomputed from that fold's 20 inner
                   half-samples (training accessions only), and the top-k composite is scored on the held-out
                   accessions; out-of-fold scores pooled within each repeat, AUC averaged over the two repeats
    greedy       : comparator that adds, at each step, the metabolite that maximizes the training AUC
                   (forward selection on labels), evaluated the same two ways
Labels
    overall      : high-BNI if any strain reaches its 75th percentile (as in the manuscript)
    strain c     : high if that strain's AOI% reaches its own 75th percentile
                   (Spearman correlation with the continuous AOI% is reported as well)

Outputs: supplementary/ST_incremental_signature.xlsx
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold

# ==========================================
# 1) Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "Data Preprocessing" / "filtered_notfound_removed.xlsx"
LABEL_PATH = ROOT / "Data Preprocessing" / "filtered_metabolites.xlsx"
STAB_XLSX = ROOT / os.environ.get("BRS_STAB_XLSX", "results_stability_uni/stability_summary_uni.xlsx")   # pi_single
NCV_DIR = ROOT / "results_nestedcv"
BIO_XLSX = ROOT / "supplementary" / "ST_single_compound_bioassays.xlsx"
NAME_MAP = ROOT / "name.xlsx"
OUT_XLSX = ROOT / "supplementary" / "ST_incremental_signature.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
N_SPLITS, N_REPEATS, SEED = 5, 2, 42          # must match nested_cv_signature.py
TOP_K = 10
K_MAX_ALL, K_MAX_STRAIN = 30, 15
MAIN_Q = 0.75
N_BOOT = 2000
DISPLAY_FIX = {"L-TRYPTOPHAN": "L-Tryptophan", "Uric Acid": "Uric acid", "Ala-ala": "Ala-Ala",
               "Isoshaftoside": "Isoschaftoside"}

# ==========================================
# 2) Data and labels (same preprocessing as nested_cv_signature.py)
# ==========================================
lab = pd.read_excel(LABEL_PATH, sheet_name=0)
bcols = {s: [c for c in lab.columns if "Inhibition" in str(c) and str(c).endswith(s)][0] for s in STRAINS}
bni = lab[[bcols[s] for s in STRAINS]].apply(pd.to_numeric, errors="coerce").set_axis(STRAINS, axis=1)
y_all = (bni >= bni.quantile(MAIN_Q)).any(axis=1).astype(int).values
y_strain = {s: (bni[s] >= bni[s].quantile(MAIN_Q)).astype(int).values for s in STRAINS}
met_cols = [c for c in lab.columns if "Inhibition" not in str(c)]
X = pd.read_excel(RAW_PATH, sheet_name=0, decimal=",")[met_cols].apply(pd.to_numeric, errors="coerce")
X = X.apply(lambda s: s.where(s != 0, s[s != 0].min() / 2))
Z = np.log1p(X)
Z = (Z - Z.mean()) / Z.std()
Z.columns = [str(c).strip() for c in Z.columns]
n = len(y_all)

nm = pd.read_excel(NAME_MAP, sheet_name=0, dtype=str).set_index("ID")
names = {k.strip(): DISPLAY_FIX.get(v.strip(), v.strip()) for k, v in nm["Compound name"].items()}
classes = {k.strip(): str(v).strip() for k, v in nm["CHEMICAL CLASS"].items()}
dup = pd.Series(names).value_counts()


def display(k):
    b = names.get(k, k)
    return f"{b} [{k.split('_')[1]}]" if dup.get(b, 0) > 1 and "_" in k else b


def auc(score, yy):
    r = rankdata(score)
    n1 = yy.sum()
    n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def composite(ids):
    return Z[list(ids)].mean(axis=1).values


# ==========================================
# 3) Orders from selection probabilities
# ==========================================
def order_from(pi: pd.DataFrame, brs: pd.DataFrame, cols):
    """Metabolites with pi>0 sorted by max pi over cols, ties by max BRS over cols, then by id."""
    p = pi[cols].max(axis=1)
    b = brs.reindex(pi.index)[cols].max(axis=1).fillna(0)
    d = pd.DataFrame({"p": p, "b": b})
    d = d[(d.p > 0) & d.index.isin(Z.columns)]
    return list(d.sort_values(["p", "b"], ascending=False, kind="mergesort").index)


# full data: pi from the 50 half-samples, BRS from the full-data run
pis = pd.read_excel(STAB_XLSX, sheet_name="pi_single")
pis["key"] = pis["key"].astype(str).str.strip()
PI_FULL = pis.set_index("key")[[f"pi_{s}" for s in STRAINS]].set_axis(STRAINS, axis=1)
BRS_FULL = pis.set_index("key")[[f"BRS_full_{s}" for s in STRAINS if f"BRS_full_{s}" in pis.columns]]
BRS_FULL = BRS_FULL.set_axis([c.replace("BRS_full_", "") for c in BRS_FULL.columns], axis=1).reindex(columns=STRAINS)


def fold_pi_brs(fold_dir: Path):
    """Selection probabilities and training-set BRS for one outer fold."""
    cnt, halves = {}, sorted(fold_dir.glob("half_*"))
    for h in halves:
        xls = pd.ExcelFile(h / "univariate_BRS.xlsx")
        for s in STRAINS:
            t = pd.read_excel(xls, f"Inhibition (pct)-{s}_Uni_All").sort_values("BRS", ascending=False)
            for m in t["Feature"].astype(str).str.strip().head(TOP_K):
                cnt.setdefault(m, dict.fromkeys(STRAINS, 0))[s] += 1
    pi = pd.DataFrame(cnt).T.reindex(columns=STRAINS).fillna(0) / len(halves)
    xls = pd.ExcelFile(fold_dir / "full" / "univariate_BRS.xlsx")
    brs = pd.concat([pd.read_excel(xls, f"Inhibition (pct)-{s}_Uni_All").assign(
        Feature=lambda d: d["Feature"].astype(str).str.strip()).set_index("Feature")["BRS"].rename(s)
        for s in STRAINS], axis=1)
    return pi, brs


def greedy_order(rows, yy, k_max, pool):
    """Forward selection: add the metabolite that maximizes the training AUC of the composite."""
    chosen, zsum = [], np.zeros(len(rows))
    Zr = Z.loc[:, pool].values[rows]
    for _ in range(k_max):
        best, best_auc = None, -1
        for j, m in enumerate(pool):
            if m in chosen:
                continue
            a = auc((zsum + Zr[:, j]) / (len(chosen) + 1), yy[rows])
            if a > best_auc:
                best, best_auc = j, a
        chosen.append(pool[best])
        zsum += Zr[:, best]
    return chosen


# ==========================================
# 4) Curves
# ==========================================
outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
             .split(np.zeros(n), y_all))
fold_cache = [fold_pi_brs(NCV_DIR / f"fold_{f:02d}") for f in range(len(outer))]
prescreen_pool = list(PI_FULL.index[PI_FULL.max(axis=1) > 0].intersection(Z.columns))
OOF_ROWS = []          # out-of-fold composite scores of the stability order, for ROC curves
PARSIMONY = 0.02       # recommended size: smallest k whose nested-CV AUC is within this of the maximum


def curves(target: str, yy: np.ndarray, cols, k_max: int):
    """In-sample and nested-CV AUC of the top-k composite for stability order and greedy order."""
    rows = []
    order_full = order_from(PI_FULL, BRS_FULL, cols)[:k_max]
    greedy_full = greedy_order(np.arange(n), yy, k_max, prescreen_pool)
    oof = {(meth, k): np.full((N_REPEATS, n), np.nan) for meth in ("stability", "greedy") for k in range(1, k_max + 1)}
    for f, (tr, te) in enumerate(outer):
        rep = f // N_SPLITS
        pi_f, brs_f = fold_cache[f]
        o_stab = order_from(pi_f, brs_f, cols)
        pool_f = list(pi_f.index[pi_f.max(axis=1) > 0].intersection(Z.columns))
        o_greedy = greedy_order(tr, yy, k_max, pool_f)
        for meth, o in (("stability", o_stab), ("greedy", o_greedy)):
            for k in range(1, k_max + 1):
                ids = o[:k] if len(o) >= k else o
                oof[(meth, k)][rep, te] = composite(ids)[te]
    for k in range(1, k_max + 1):
        O = oof[("stability", k)]
        OOF_ROWS.extend({"target": target, "k": k, "repeat": r, "sample": i, "y": int(yy[i]), "oof_score": O[r, i]}
                        for r in range(N_REPEATS) for i in range(n))
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
    boots = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(N_BOOT)]
    for k in range(1, k_max + 1):
        rec = {"target": target, "k": k}
        for meth, o in (("stability", order_full), ("greedy", greedy_full)):
            ids = o[:k]
            sc = composite(ids)
            rec[f"{meth}_added"] = display(ids[-1]) if len(ids) == k else ""
            rec[f"{meth}_added_key"] = ids[-1] if len(ids) == k else ""
            rec[f"{meth}_AUC_in_sample"] = auc(sc, yy)
            O = oof[(meth, k)]
            rec[f"{meth}_AUC_nestedCV"] = np.mean([auc(O[r], yy) for r in range(N_REPEATS)])
            if meth == "stability":
                bs = [np.mean([auc(O[r, b], yy[b]) for r in range(N_REPEATS)]) for b in boots[:500]]
                rec["stability_CV_CI_low"], rec["stability_CV_CI_high"] = np.percentile(bs, [2.5, 97.5])
                if target != "overall":
                    rec["stability_spearman_AOI"] = spearmanr(sc, bni[target]).correlation
                    rec["stability_spearman_AOI_nestedCV"] = np.mean(
                        [spearmanr(O[r], bni[target]).correlation for r in range(N_REPEATS)])
        if len(order_full) >= k:
            rec["pi_of_added"] = PI_FULL.loc[order_full[k - 1], cols].max()
        rows.append(rec)
    return pd.DataFrame(rows)


res = [curves("overall", y_all, STRAINS, K_MAX_ALL)]
for s in STRAINS:
    res.append(curves(s, y_strain[s], [s], K_MAX_STRAIN))
C = pd.concat(res, ignore_index=True)
REC = []
for t, d in C.groupby("target", sort=False):
    best = d["stability_AUC_nestedCV"].max()
    k_rec = int(d.loc[d["stability_AUC_nestedCV"] >= best - PARSIMONY, "k"].min())
    r = d[d.k == k_rec].iloc[0]
    REC.append({"target": t, "recommended_k": k_rec, "AUC_nestedCV_at_k": r["stability_AUC_nestedCV"],
                "CI_low": r["stability_CV_CI_low"], "CI_high": r["stability_CV_CI_high"],
                "AUC_nestedCV_k1": d[d.k == 1]["stability_AUC_nestedCV"].item(), "max_AUC_nestedCV": best,
                "k_at_max": int(d.loc[d["stability_AUC_nestedCV"].idxmax(), "k"]),
                "members": "; ".join(d.sort_values("k")["stability_added"].head(k_rec))})
REC = pd.DataFrame(REC)

# ==========================================
# 5) Wet-lab list per strain (stability order) with EC50 where tested
# ==========================================
# single-compound assays of the consortium are not part of the manuscript and not distributed: optional
bio = pd.read_excel(BIO_XLSX, sheet_name="all_results") if BIO_XLSX.exists() else None


def ec50(key, strain):
    r = bio[(bio.metabolite == names.get(key, key)) & (bio.strain == strain)]
    if r.empty:
        return "n.t."
    r = r.iloc[0]
    return f"{r.EC50_uM:.0f}" if r.status == "EC50" else "n.i."


wet = []
for s in STRAINS:
    d = C[C.target == s]
    best_cv = d["stability_AUC_nestedCV"].max()
    for _, r in d.iterrows():
        k = int(r.k)
        key = r["stability_added_key"]
        if not key:
            continue
        lev = pis.set_index("key").get(f"leverage_{s}")
        wet.append({"strain": s, "step": k, "metabolite_added": display(key), "class": classes.get(key, ""),
                    "pi_strain": PI_FULL.loc[key, s], "in_signature (pi>=0.4)": PI_FULL.loc[key].max() >= 0.4,
                    "leverage": (lev.get(key) if lev is not None else np.nan),
                    **({"EC50_uM_this_strain": ec50(key, s)} if bio is not None else {}),
                    "cumulative_AUC_in_sample": r["stability_AUC_in_sample"],
                    "cumulative_AUC_nestedCV": r["stability_AUC_nestedCV"],
                    "within_0.02_of_best_CV": r["stability_AUC_nestedCV"] >= best_cv - 0.02, "key": key})
W = pd.DataFrame(wet)

with pd.ExcelWriter(OUT_XLSX) as w:
    pd.DataFrame([
        ("order", "metabolites added by selection probability pi (overall: max over strains; strain: pi of that strain); "
                  "ties broken by BRS"),
        ("in-sample AUC", "order and AUC from the same 44 accessions (optimistic)"),
        ("nested-CV AUC", "order recomputed in each outer fold from 20 inner half-samples of the training accessions; "
                          "top-k composite scored on held-out accessions (5 folds x 2 repeats)"),
        ("greedy", "comparator: forward selection maximizing training AUC"),
        ("strain labels", "high if the strain's AOI% reaches its 75th percentile"),
    ], columns=["field", "description"]).to_excel(w, sheet_name="README", index=False)
    C.round(4).to_excel(w, sheet_name="curves", index=False)
    REC.round(4).to_excel(w, sheet_name="recommended_size", index=False)
    W.round(3).to_excel(w, sheet_name="wetlab_lists", index=False)
    pd.DataFrame(OOF_ROWS).to_excel(w, sheet_name="oof_scores", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
show = ["target", "k", "stability_added", "pi_of_added", "stability_AUC_in_sample", "stability_AUC_nestedCV",
        "greedy_AUC_in_sample", "greedy_AUC_nestedCV"]
print(C[show].round(3).to_string(index=False))
print(REC.round(3).to_string(index=False))
print(f"Saved {OUT_XLSX}")
