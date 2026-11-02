# -*- coding: utf-8 -*-
"""
AUC of the combinations from the gated combination test (Step 4), against each strain's own high-inhibition label
(AOI% >= the strain's 75th percentile), evaluated in two ways.

    fixed    every pair and triad of the full-data gated pools (Table S16) as a fixed set:
             - in-sample AUC of the composite score (mean of autoscaled log intensities), stratified bootstrap 95% CI,
               and the paired difference from the better member;
             - cross-validated AUC (repeated stratified 5-fold) of logistic regressions with additive terms and with
               the product terms of the BRS design, to see whether the interaction helps classification.
             Optimistic: the gated pools were derived on all accessions.
    nested   the procedure "pick the best pair from the gated pool", with selection inside the training folds:
             same outer folds and inner half-samples as nested_cv_signature.py; in every outer fold the pool is
             re-derived from the training half-samples (pi >= 0.4 for the strain, completed to at least POOL_MIN
             metabolites by decreasing pi), the pair and the single metabolite with the largest training AUC are
             chosen, and their composites (and the logistic models of the chosen pair) score the held-out accessions.

Output: supplementary/ST_gated_combination_auc.xlsx (Fig. S8: gated_combination_roc_fig.py)
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

# ==========================================
# 1) Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "Data Preprocessing" / "filtered_notfound_removed.xlsx"
LABEL_PATH = ROOT / "Data Preprocessing" / "filtered_metabolites.xlsx"
STAB_XLSX = ROOT / "results_stability_uni" / "stability_summary_uni.xlsx"
GATED_XLSX = ROOT / "supplementary" / "ST_gated_combinations.xlsx"
INC_XLSX = ROOT / "supplementary" / "ST_incremental_signature.xlsx"
NCV_DIR = ROOT / "results_nestedcv"
NAME_MAP = ROOT / "name.xlsx"
OUT_XLSX = ROOT / "supplementary" / "ST_gated_combination_auc.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
N_SPLITS, N_REPEATS, SEED = 5, 2, 42          # outer folds: must match nested_cv_signature.py
CV_REPEATS = 20                               # repeats for the fixed-set logistic models
MAIN_Q, PI_GATE, POOL_MIN = 0.75, 0.4, 4
TOP_K = 10
N_BOOT = 2000
TANH_SCALE = 3.0
DISPLAY_FIX = {"L-TRYPTOPHAN": "L-Tryptophan", "Uric Acid": "Uric acid", "Ala-ala": "Ala-Ala",
               "Isoshaftoside": "Isoschaftoside"}

# ==========================================
# 2) Data and labels (same preprocessing as incremental_signature.py)
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
dup = pd.Series(names).value_counts()


def display(k):
    b = names.get(k, k)
    return f"{b} [{k.split('_')[1]}]" if dup.get(b, 0) > 1 and "_" in k else b


def auc(score, yy):
    r = rankdata(score)
    n1 = yy.sum()
    n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def composite(ids, rows=slice(None)):
    return Z[list(ids)].values[rows].mean(axis=1)


def design(ids, interaction):
    """Additive columns, or the BRS design (pair: raw product; triad: tanh of centred products)."""
    cols = [Z[i].values for i in ids]
    if not interaction:
        return np.column_stack(cols)
    if len(ids) == 2:
        return np.column_stack(cols + [cols[0] * cols[1]])
    c = [v - v.mean() for v in cols]
    g = lambda z: np.tanh(z / TANH_SCALE)
    return np.column_stack(cols + [g(c[0] * c[1]), g(c[0] * c[2]), g(c[1] * c[2]), g(c[0] * c[1] * c[2])])


def logit_oof(M, yy, tr, te):
    mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-8
    m = LogisticRegression(max_iter=2000).fit((M[tr] - mu) / sd, yy[tr])
    return m.decision_function((M[te] - mu) / sd)


def strat_boot(yy, rng, b=N_BOOT):
    pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(b)]


# ==========================================
# 3) Fixed sets: the full-data gated pools
# ==========================================
pis = pd.read_excel(STAB_XLSX, sheet_name="pi_single")
pis["key"] = pis["key"].astype(str).str.strip()
PI_FULL = pis.set_index("key")[[f"pi_{s}" for s in STRAINS]].set_axis(STRAINS, axis=1)
wet = pd.read_excel(INC_XLSX, sheet_name="wetlab_lists")
rec_k = pd.read_excel(INC_XLSX, sheet_name="recommended_size").set_index("target")["recommended_k"]
gated = pd.read_excel(GATED_XLSX, sheet_name="combinations")

pools = {}
for s in STRAINS:
    gate = set(PI_FULL.index[PI_FULL[s] >= PI_GATE])
    mix = set(wet.loc[(wet.strain == s) & (wet.step <= rec_k[s]), "key"].astype(str).str.strip())
    pools[s] = sorted(gate | mix, key=lambda k: -PI_FULL.loc[k, s])     # same pools as gated_combinations.py

print("--- Fixed sets ---", flush=True)
rng = np.random.default_rng(SEED)
cv = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=CV_REPEATS, random_state=SEED).split(np.zeros(n),
                                                                                                    y_all))
fixed = []
for s in STRAINS:
    yy = y_strain[s]
    boots = strat_boot(yy, rng)
    for k in (1, 2, 3):
        for ids in combinations(pools[s], k):
            sc = composite(ids)
            rec = {"strain": s, "order": {1: "single", 2: "pair", 3: "triad"}[k],
                   "combination": " + ".join(display(i) for i in ids), "keys": " + ".join(ids),
                   "AUC_composite": auc(sc, yy)}
            bs = np.array([auc(sc[b], yy[b]) for b in boots])
            rec["CI_low"], rec["CI_high"] = np.percentile(bs, [2.5, 97.5])
            if k > 1:
                member = [composite([i]) for i in ids]
                best = int(np.argmax([auc(m, yy) for m in member]))
                rec["best_member"] = display(ids[best])
                rec["AUC_best_member"] = auc(member[best], yy)
                d = np.array([auc(sc[b], yy[b]) - auc(member[best][b], yy[b]) for b in boots])
                rec["delta_vs_best_member"] = rec["AUC_composite"] - rec["AUC_best_member"]
                rec["delta_CI_low"], rec["delta_CI_high"] = np.percentile(d, [2.5, 97.5])
                for lab_, inter in (("additive", False), ("interaction", True)):
                    M = design(ids, inter)
                    O = np.full((CV_REPEATS, n), np.nan)
                    for f, (tr, te) in enumerate(cv):
                        O[f // N_SPLITS, te] = logit_oof(M, yy, tr, te)
                    rec[f"AUC_CV_logistic_{lab_}"] = np.mean([auc(O[r], yy) for r in range(CV_REPEATS)])
                rec["AUC_CV_interaction_minus_additive"] = (rec["AUC_CV_logistic_interaction"]
                                                            - rec["AUC_CV_logistic_additive"])
            fixed.append(rec)
F = pd.DataFrame(fixed)
# attach the Step 4 gains
members = lambda c: " | ".join(sorted(c.split(" + ")))          # tied pi can order members differently
F = F.assign(m=F.combination.map(members)).merge(
    gated.assign(m=gated.combination.map(members))[["strain", "m", "pi_comp", "pi_int"]],
    on=["strain", "m"], how="left").drop(columns="m")

# ==========================================
# 4) Nested: pick the best pair from the gated pool inside the training folds
# ==========================================
print("--- Nested selection ---", flush=True)
outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
             .split(np.zeros(n), y_all))


def fold_pi(fold_dir: Path):
    cnt, halves = {}, sorted(fold_dir.glob("half_*"))
    for h in halves:
        xls = pd.ExcelFile(h / "univariate_BRS.xlsx")
        for s in STRAINS:
            t = pd.read_excel(xls, f"Inhibition (pct)-{s}_Uni_All").sort_values("BRS", ascending=False)
            for m in t["Feature"].astype(str).str.strip().head(TOP_K):
                cnt.setdefault(m, dict.fromkeys(STRAINS, 0))[s] += 1
    return pd.DataFrame(cnt).T.reindex(columns=STRAINS).fillna(0) / len(halves)


fold_cache = [fold_pi(NCV_DIR / f"fold_{f:02d}") for f in range(len(outer))]
METHODS = ["best single", "first two by pi", "best pair", "best pair, logistic additive",
           "best pair, logistic interaction"]
nested, choices, oof_rows = [], [], []
for s in STRAINS:
    yy = y_strain[s]
    O = {m: np.full((N_REPEATS, n), np.nan) for m in METHODS}
    for f, (tr, te) in enumerate(outer):
        rep = f // N_SPLITS
        p = fold_cache[f][s]
        p = p[p.index.isin(Z.columns)].sort_values(ascending=False, kind="mergesort")
        pool = list(p.index[p >= PI_GATE])
        if len(pool) < POOL_MIN:
            pool = list(p.index[:POOL_MIN])
        single = max(pool, key=lambda i: auc(composite([i], tr), yy[tr]))
        pair = max(combinations(pool, 2), key=lambda ids: auc(composite(ids, tr), yy[tr]))
        O["best single"][rep, te] = composite([single], te)
        O["first two by pi"][rep, te] = composite(pool[:2], te)
        O["best pair"][rep, te] = composite(pair, te)
        for lab_, inter in (("additive", False), ("interaction", True)):
            O[f"best pair, logistic {lab_}"][rep, te] = logit_oof(design(pair, inter), yy, tr, te)
        choices.append({"strain": s, "fold": f, "pool_size": len(pool), "pool": "; ".join(display(i) for i in pool),
                        "chosen_single": display(single), "chosen_pair": " + ".join(display(i) for i in pair)})
    boots = strat_boot(yy, np.random.default_rng(SEED), 1000)
    for m in METHODS:
        a = np.mean([auc(O[m][r], yy) for r in range(N_REPEATS)])
        bs = [np.mean([auc(O[m][r, b], yy[b]) for r in range(N_REPEATS)]) for b in boots]
        rec = {"strain": s, "method": m, "AUC_nestedCV": a, "CI_low": np.percentile(bs, 2.5),
               "CI_high": np.percentile(bs, 97.5)}
        if m != "best single":
            d = [np.mean([auc(O[m][r, b], yy[b]) - auc(O["best single"][r, b], yy[b]) for r in range(N_REPEATS)])
                 for b in boots]
            rec["delta_vs_best_single"] = a - np.mean([auc(O["best single"][r], yy) for r in range(N_REPEATS)])
            rec["delta_CI_low"], rec["delta_CI_high"] = np.percentile(d, [2.5, 97.5])
        nested.append(rec)
        oof_rows.extend({"strain": s, "method": m, "repeat": r, "sample": i, "y": int(yy[i]), "oof": O[m][r, i]}
                        for r in range(N_REPEATS) for i in range(n))
N = pd.DataFrame(nested)
C = pd.DataFrame(choices)
freq = (C.groupby(["strain", "chosen_pair"]).size().rename("folds").reset_index()
        .sort_values(["strain", "folds"], ascending=[True, False]))

# ==========================================
# 5) Save
# ==========================================
readme = pd.DataFrame([
    ("labels", "strain-specific: high if the strain's AOI% reaches its 75th percentile"),
    ("composite", "mean of the autoscaled log1p intensities of the members"),
    ("fixed", "pairs and triads of the full-data gated pools (Table S16) as fixed sets; in-sample AUC with stratified "
              "bootstrap CI; logistic AUC under repeated stratified 5-fold CV (20 repeats), additive vs BRS design "
              "with product terms. Optimistic, because the pools were derived on all accessions"),
    ("nested", "outer folds and inner half-samples of nested_cv_signature.py; pool re-derived in each training set "
               f"(pi >= {PI_GATE}, at least {POOL_MIN} metabolites); best single and best pair chosen by training AUC; "
               "held-out composite (or logistic) scores; bootstrap CI over accessions"),
], columns=["field", "description"])
OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT_XLSX) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    F.round(3).to_excel(w, sheet_name="fixed_sets", index=False)
    N.round(3).to_excel(w, sheet_name="nested", index=False)
    freq.to_excel(w, sheet_name="nested_pair_choice", index=False)
    C.to_excel(w, sheet_name="nested_folds", index=False)
    pd.DataFrame(oof_rows).to_excel(w, sheet_name="nested_oof", index=False)
print(f"Saved {OUT_XLSX}")

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
show = F[F.order != "single"].sort_values(["strain", "AUC_composite"], ascending=[True, False])
print(show.groupby("strain").head(4)[["strain", "combination", "AUC_composite", "CI_low", "CI_high",
                                      "AUC_best_member", "delta_vs_best_member", "delta_CI_low", "delta_CI_high",
                                      "AUC_CV_logistic_additive", "AUC_CV_logistic_interaction", "pi_comp",
                                      "pi_int"]].round(2).to_string(index=False))
print(N.round(2).to_string(index=False))
print(freq.to_string(index=False))
