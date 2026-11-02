# -*- coding: utf-8 -*-
"""
Table 3: BRS-prioritized metabolite combinations as candidates for mixture assays.

Every candidate (single, pair, triad) is evaluated with ONE composite-score definition:
    raw intensity -> zeros replaced by half the column's minimum non-zero value
    (as in Data Preprocessing_Dimension Reduction.py) -> log1p -> per-metabolite
    z-score across the 44 accessions -> mean within the candidate set.

This replaces the three AUC routes used for Figs 6-8, which applied log1p to the
already z-scored matrix (filtered_metabolites.xlsx): Fig. 7 clipped every negative
z-score to zero, Fig. 6 dropped samples with z <= -1, and Fig. 8 used a
cross-validated logistic regression.

Outputs
    supplementary/ST_mixture_candidates.xlsx  (full table, threshold ranks, old-vs-new AUC,
                                               template for single-compound bioassay data)
    论文/overleaf_v6/tables/tableS8_combinations.tex (Supplementary Table S8, typeset excerpt)
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

# ==========================================
# 1) Configuration
# ==========================================
RAW_PATH = r"Data Preprocessing/filtered_notfound_removed.xlsx"   # raw intensities
LABEL_PATH = r"Data Preprocessing/filtered_metabolites.xlsx"      # BNI columns used by the published labels
RESULTS = Path(os.environ.get("BRS_OUT_DIR", "results"))           # BRS run to characterize
CANDIDATE_PATH = (RESULTS / "candidate_sets.xlsx" if (RESULTS / "candidate_sets.xlsx").exists()
                  else Path("Top5_single_pair_triple_BRS.xlsx"))   # written by build_candidate_sets.py
CANDIDATE_SHEET = "Sheet1"
NAME_MAP_PATH = r"name.xlsx"
OLD_AUC_PATH = r"supplementary/Supplementary_Data.xlsx"

OUT_XLSX = Path("supplementary/ST_mixture_candidates.xlsx")
OUT_TEX = Path("论文/overleaf_v6/tables/tableS8_combinations.tex")   # typeset in the Supplementary Information
TEX_WITH_BIO = False   # single-compound assays are reported only in their own table (Supplementary Table S19)

STRAINS = ["NF", "NV", "NM", "NU"]
MAIN_Q = 0.75
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
TOP_K_STABLE = 5          # "stable" = within the top 5 of the 30 candidates at a threshold
N_BOOT = 2000
N_PERM = 10000
CV_FOLDS, CV_REPEATS = 5, 20
SEED = 42
N_MAIN_TABLE = 8          # combinations shown in the main-text table

DISPLAY_FIX = {"L-TRYPTOPHAN": "L-Tryptophan", "Uric Acid": "Uric acid", "Ala-ala": "Ala-Ala",
               "Isoshaftoside": "Isoschaftoside"}

# Single-compound pure-culture results (written by Table_S7_bioassays.py); optional
BIOASSAY_PATH = Path("supplementary/ST_single_compound_bioassays.xlsx")
ABBR = {"Uric acid": "UA", "N-Acetyl-DL-glutamic acid": "NAG", "L-Phenylalanine": "Phe", "L-Tryptophan": "Trp",
        "6,8C-glycosylated flavone": "GF", "Isoschaftoside": "ISO", "Protocatechuic acid": "PCA",
        "2-Acetylpyrazine": "2-AP", "Disaccharide - C12H22O11": "DS", "2-Hydroxybenzenepropanoic acid": "HBP",
        "N,N-Dimethyl-L-arginine": "DMA", "Trimethyllysine": "TML", "L-Aspartic acid": "Asp", "Ala-Ala": "AA",
        "Catechol": "Cat", "Syringic acid": "SyA", "similar to Ferulic acid": "FA-like",
        "Phosphatidylethanolamine (PE) (11:0_21:1)": "PE"}

rng = np.random.default_rng(SEED)


# ==========================================
# 2) Data and labels
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
print(f"✅ {Z.shape[0]} accessions x {Z.shape[1]} metabolites; high-BNI = {y.sum()} / {len(y)} at q = {MAIN_Q}")


# ==========================================
# 3) Candidate sets and display names
# ==========================================
print("\n--- Step 2: Parsing BRS candidate sets ---")
id2name = pd.read_excel(NAME_MAP_PATH, sheet_name=0, dtype=str).set_index("ID")["Compound name"].str.strip().to_dict()
cand = pd.read_excel(CANDIDATE_PATH, sheet_name=CANDIDATE_SHEET)
brs_cols = {s: [c for c in cand.columns if s in str(c)][0] for s in STRAINS}

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
    brs = {s: float(row[brs_cols[s]]) for s in STRAINS}
    candidates.append({
        "ids": ids,
        "candidate": " + ".join(display(i) for i in ids),
        "order": {1: "single", 2: "pair", 3: "triad"}[len(ids)],
        "prioritized_under": ", ".join(f"{s} ({v:.2f})" for s, v in sorted(brs.items(), key=lambda kv: -kv[1]) if v > 0),
        "max_BRS": max(brs.values()),
        **{f"BRS_{s}": brs[s] for s in STRAINS},
    })
print(f"✅ {len(candidates)} candidate sets "
      f"({sum(c['order'] == 'single' for c in candidates)} singles, "
      f"{sum(c['order'] == 'pair' for c in candidates)} pairs, "
      f"{sum(c['order'] == 'triad' for c in candidates)} triads)")


# ==========================================
# 4) Composite scores, AUC, bootstrap CI, permutation p
# ==========================================
def rank_auc(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC for every column of `ranks`; `labels` is (n,) or (n_perm, n)."""
    labels = np.atleast_2d(labels)
    n1 = labels.sum(axis=1, keepdims=True)
    n0 = labels.shape[1] - n1
    return (labels @ ranks - n1 * (n1 + 1) / 2) / (n1 * n0)


print("\n--- Step 3: AUC, bootstrap CI and permutation tests ---")
S = np.column_stack([Z[c["ids"]].mean(axis=1).values for c in candidates])   # n x k composite scores
R = np.apply_along_axis(rankdata, 0, S)
auc_obs = rank_auc(R, y)[0]

# Stratified bootstrap; constituent metabolites are resampled jointly so that the
# combination-minus-constituent difference gets a paired interval
const_ids = sorted({i for c in candidates for i in c["ids"]})
S_all = np.hstack([S, Z[const_ids].values])
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
boot = np.empty((N_BOOT, S_all.shape[1]))
for b in range(N_BOOT):
    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    boot[b] = rank_auc(np.apply_along_axis(rankdata, 0, S_all[idx]), y[idx])[0]
ci_lo, ci_hi = np.percentile(boot[:, :S.shape[1]], [2.5, 97.5], axis=0)

# Label permutation; the max statistic over all candidates gives a family-wise adjusted p
perm_labels = np.array([rng.permutation(y) for _ in range(N_PERM)])
null = rank_auc(R, perm_labels)
p_raw = (1 + (null >= auc_obs).sum(axis=0)) / (N_PERM + 1)
p_fwer = (1 + (null.max(axis=1, keepdims=True) >= auc_obs).sum(axis=0)) / (N_PERM + 1)

# Global metabolome reference (mean of all z-scored metabolites), as in Fig. 8a
g_score = Z.mean(axis=1).values
g_rank = rankdata(g_score)[:, None]
g_auc = rank_auc(g_rank, y)[0, 0]
g_p = (1 + (rank_auc(g_rank, perm_labels)[:, 0] >= g_auc).sum()) / (N_PERM + 1)
print(f"✅ Global metabolome reference: AUC = {g_auc:.3f}, permutation p = {g_p:.4f}")


# ==========================================
# 5) Threshold stability and gain over constituents
# ==========================================
print("\n--- Step 4: Threshold stability and constituent comparison ---")
thr_auc = pd.DataFrame({f"q{int(q * 100)}": rank_auc(R, bni_labels(bni, q))[0] for q in THRESHOLDS})
thr_rank = thr_auc.rank(ascending=False, method="min").astype(int)
n_stable = (thr_rank <= TOP_K_STABLE).sum(axis=1)


def single_auc(met_id: str) -> float:
    return float(rank_auc(rankdata(Z[met_id].values)[:, None], y)[0, 0])


bio = pd.read_excel(BIOASSAY_PATH, sheet_name="all_results") if BIOASSAY_PATH.exists() else None


def bioassay_summary(ids, strains) -> str:
    """Pure-culture result of each constituent on the strains the set was prioritized under."""
    if bio is None:
        return ""
    active, inactive, untested = [], [], []
    for i in ids:
        nm = base_names[i]
        ab = ABBR.get(nm, nm)
        hits = bio[(bio.metabolite == nm) & bio.strain.isin(strains)]
        if hits.empty:
            untested.append(ab)
        elif (hits.status == "EC50").any():
            active += [f"{ab} {v:.0f} ({s})" for v, s in zip(hits.EC50_uM, hits.strain) if not np.isnan(v)]
        else:
            inactive.append(ab)
    parts = active + ([", ".join(inactive) + " n.i."] if inactive else []) + ([", ".join(untested) + " n.t."] if untested else [])
    return "; ".join(parts)


rows = []
for j, c in enumerate(candidates):
    const_auc = {display(i): single_auc(i) for i in c["ids"]}
    best_const = max(const_auc.values())
    best_id = max(c["ids"], key=single_auc)
    delta_boot = boot[:, j] - boot[:, S.shape[1] + const_ids.index(best_id)]
    d_lo, d_hi = np.percentile(delta_boot, [2.5, 97.5]) if c["order"] != "single" else (np.nan, np.nan)
    rows.append({
        "candidate": c["candidate"], "order": c["order"], "prioritized_under": c["prioritized_under"],
        "max_BRS": c["max_BRS"], **{f"BRS_{s}": c[f"BRS_{s}"] for s in STRAINS},
        "AUC": auc_obs[j], "AUC_CI_low": ci_lo[j], "AUC_CI_high": ci_hi[j],
        "best_constituent_AUC": best_const,
        "delta_AUC_vs_best_constituent": auc_obs[j] - best_const if c["order"] != "single" else np.nan,
        "delta_CI_low": d_lo, "delta_CI_high": d_hi,
        "constituent_AUCs": "; ".join(f"{k} {v:.2f}" for k, v in const_auc.items()),
        "p_perm": p_raw[j], "p_perm_FWER": p_fwer[j],
        "n_thresholds_top5": int(n_stable[j]),
        "rank_range_across_thresholds": f"{thr_rank.iloc[j].min()}-{thr_rank.iloc[j].max()}",
        "pure_culture_constituents": bioassay_summary(c["ids"], [s for s in STRAINS if c[f"BRS_{s}"] > 0]),
        "metabolite_ids": " | ".join(c["ids"]),
    })
res = pd.DataFrame(rows)
res_sorted = res.sort_values("AUC", ascending=False).reset_index(drop=True)
thr_table = pd.concat([res[["candidate", "order"]], thr_auc.round(3), thr_rank.add_prefix("rank_")], axis=1)

combos = res_sorted[res_sorted["order"] != "single"].reset_index(drop=True)
best_single = res_sorted[res_sorted["order"] == "single"].iloc[0]
print(f"✅ Best single: {best_single['candidate']} AUC = {best_single['AUC']:.3f}")
print(f"✅ Best combination: {combos.iloc[0]['candidate']} AUC = {combos.iloc[0]['AUC']:.3f} "
      f"[{combos.iloc[0]['AUC_CI_low']:.2f}, {combos.iloc[0]['AUC_CI_high']:.2f}]")


# ==========================================
# 5b) Cross-validated AUC with candidate choice inside the training folds
# ==========================================
print("\n--- Step 4b: Cross-validation with in-fold candidate selection ---")
is_combo = np.array([c["order"] != "single" for c in candidates])
cv_rows = []
for rep in range(CV_REPEATS):
    for tr, te in StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED + rep).split(S, y):
        a_tr = rank_auc(np.apply_along_axis(rankdata, 0, S[tr]), y[tr])[0]
        a_te = rank_auc(np.apply_along_axis(rankdata, 0, S[te]), y[te])[0]
        jc = np.flatnonzero(is_combo)[a_tr[is_combo].argmax()]
        js = np.flatnonzero(~is_combo)[a_tr[~is_combo].argmax()]
        cv_rows.append({"repeat": rep, "combo_chosen": candidates[jc]["candidate"], "combo_test_AUC": a_te[jc],
                        "single_chosen": candidates[js]["candidate"], "single_test_AUC": a_te[js]})
cv = pd.DataFrame(cv_rows)
cv["paired_diff"] = cv["combo_test_AUC"] - cv["single_test_AUC"]
cv_summary = pd.DataFrame([
    {"pool": "combinations", "in_sample_best_AUC": auc_obs[is_combo].max(),
     "cv_mean_AUC": cv["combo_test_AUC"].mean(), "cv_sd": cv["combo_test_AUC"].std(),
     "optimism": auc_obs[is_combo].max() - cv["combo_test_AUC"].mean(),
     "most_chosen": cv["combo_chosen"].mode()[0], "chosen_share": (cv["combo_chosen"] == cv["combo_chosen"].mode()[0]).mean()},
    {"pool": "single metabolites", "in_sample_best_AUC": auc_obs[~is_combo].max(),
     "cv_mean_AUC": cv["single_test_AUC"].mean(), "cv_sd": cv["single_test_AUC"].std(),
     "optimism": auc_obs[~is_combo].max() - cv["single_test_AUC"].mean(),
     "most_chosen": cv["single_chosen"].mode()[0], "chosen_share": (cv["single_chosen"] == cv["single_chosen"].mode()[0]).mean()},
    {"pool": "paired difference (combination - single)", "cv_mean_AUC": cv["paired_diff"].mean(),
     "cv_sd": cv["paired_diff"].std(),
     "most_chosen": f"95% interval [{cv['paired_diff'].quantile(0.025):.2f}, {cv['paired_diff'].quantile(0.975):.2f}]; "
                    f"combination higher in {(cv['paired_diff'] > 0).mean():.0%} of folds, equal in {(cv['paired_diff'] == 0).mean():.0%}"},
])
print(f"✅ CV mean AUC: combinations {cv['combo_test_AUC'].mean():.3f}, singles {cv['single_test_AUC'].mean():.3f}; "
      f"paired diff {cv['paired_diff'].mean():+.3f} "
      f"[{cv['paired_diff'].quantile(0.025):.2f}, {cv['paired_diff'].quantile(0.975):.2f}]")


# ==========================================
# 6) Published vs corrected AUC
# ==========================================
old7 = pd.read_excel(OLD_AUC_PATH, sheet_name="Fig7_auc")
old6 = pd.read_excel(OLD_AUC_PATH, sheet_name="Fig6_auc").set_index("metabolite_id")["AUC"]
key = {" | ".join(c["ids"]): c["candidate"] for c in candidates}
old_vs_new = pd.DataFrame({
    "candidate": [key.get(" | ".join(s.strip() for s in str(r).split("|")), r) for r in old7["combo_raw"]],
    "AUC_published_Fig7": old7["AUC_score"].values,
})
old_vs_new["AUC_published_Fig6"] = [
    old6.get(r.strip(), np.nan) if "|" not in str(r) else np.nan for r in old7["combo_raw"]]
old_vs_new = old_vs_new.merge(res[["candidate", "order", "AUC"]].rename(columns={"AUC": "AUC_corrected"}),
                              on="candidate", how="left")
old_vs_new["change_vs_Fig7"] = old_vs_new["AUC_corrected"] - old_vs_new["AUC_published_Fig7"]
old_vs_new = old_vs_new.sort_values("AUC_corrected", ascending=False)


# ==========================================
# 7) Bioassay template (constituents of all combinations)
# ==========================================
constituents = sorted({display(i) for c in candidates if c["order"] != "single" for i in c["ids"]})
bioassay = pd.DataFrame({
    "metabolite": constituents,
    "n_combinations_containing": [sum(m in c["candidate"].split(" + ") for c in candidates if c["order"] != "single")
                                  for m in constituents],
    **{f"max_inhibition_pct_{s}": "" for s in STRAINS},
    **{f"EC50_uM_{s}": "" for s in STRAINS},
    "concentration_range_tested_uM": "", "replicates": "", "source": "",
}).sort_values("n_combinations_containing", ascending=False)


# ==========================================
# 8) Export
# ==========================================
print("\n--- Step 5: Exporting ---")
readme = pd.DataFrame([
    ("purpose", "Candidate metabolite combinations for targeted mixture assays (main-text Table 3)"),
    ("addresses", "C172 (multiple testing), C247/C248/C253/C274 (singles vs combinations), C254 (validation), "
                  "C259-C262 (permutation test and global baseline), C329 (bioassay evidence must be shown)"),
    ("composite_score", "raw intensity -> zero replaced by half the minimum non-zero value -> log1p -> per-metabolite "
                        "z-score -> mean within the candidate set; identical for singles, pairs and triads"),
    ("labels", f"High-BNI if any of NF/NV/NM/NU reaches its own {int(MAIN_Q * 100)}th percentile (OR rule); "
               f"{int(y.sum())} high vs {int(len(y) - y.sum())} low"),
    ("AUC_CI", f"Stratified bootstrap, {N_BOOT} resamples, percentile interval"),
    ("delta_CI", "Paired bootstrap interval of AUC(combination) - AUC(best single constituent), same resamples"),
    ("cv_in_fold_selection", f"Repeated stratified {CV_FOLDS}-fold CV ({CV_REPEATS} repeats): the highest-AUC combination "
                             "and the highest-AUC single metabolite are chosen on the training folds and scored on the "
                             "held-out fold. Supersedes ST_cross_validated_auc.xlsx, which used the clipped score."),
    ("p_perm", f"Label permutation, {N_PERM} permutations, (1 + #null >= observed) / (N + 1)"),
    ("p_perm_FWER", "Max-statistic (Westfall-Young) adjustment over all 30 evaluated candidate sets"),
    ("n_thresholds_top5", f"Number of labelling thresholds ({', '.join(str(int(q * 100)) for q in THRESHOLDS)}th "
                          f"percentile) at which the set ranks in the top {TOP_K_STABLE} of the 30 candidates by AUC"),
    ("global_reference", f"Mean of all {Z.shape[1]} z-scored metabolites: AUC = {g_auc:.3f}, p = {g_p:.4f}"),
    ("why_corrected", "The published Fig. 6-8 AUCs applied log1p to an already z-scored matrix "
                      "(Fig. 7 clipped negative z-scores to 0; Fig. 6 dropped samples with z <= -1; Fig. 8 used "
                      "cross-validated logistic regression). See sheet old_vs_new."),
    ("caveat", "Candidate sets were selected by BRS on the same 44 accessions, so AUC and p are descriptive of "
               "this cohort; the FWER adjustment covers the 30 evaluated sets, not every enumerated combination."),
    ("bioassay_template", "Fill in the single-compound pure-culture results; constituents sorted by how many "
                          "candidate combinations contain them"),
], columns=["field", "description"])

OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT_XLSX) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    res_sorted.round(4).to_excel(w, sheet_name="all_candidates", index=False)
    combos.round(4).to_excel(w, sheet_name="combinations", index=False)
    thr_table.to_excel(w, sheet_name="threshold_ranks", index=False)
    cv_summary.round(4).to_excel(w, sheet_name="cv_summary", index=False)
    cv.round(4).to_excel(w, sheet_name="cv_folds", index=False)
    old_vs_new.round(4).to_excel(w, sheet_name="old_vs_new", index=False)
    bioassay.to_excel(w, sheet_name="bioassay_template", index=False)
print(f"✅ Saved: {OUT_XLSX}")


def tex(s: str) -> str:
    s = str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    return s.replace("Hydroxybenzenepropanoic", r"Hydroxy\-benzene\-propanoic")   # breakable in narrow columns


def fmt_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def num(x: float, sign: bool = False) -> str:
    return (f"{x:+.2f}" if sign else f"{x:.2f}").replace("-", "$-$")


main_rows = combos.head(N_MAIN_TABLE)
one_order = main_rows["order"].nunique() == 1
with_bio = bio is not None and TEX_WITH_BIO
cols = ["Candidate"] + ([] if one_order else ["Order"]) + ["Strain (BRS)", r"AUC [95\% CI]", r"$\Delta$AUC [95\% CI]",
                                                           r"$p_{\mathrm{adj}}$", "Stability"]
RAG = r">{\raggedright\arraybackslash}"   # needs \usepackage{array}
spec = RAG + "p{2.9cm}" + ("" if one_order else "l") + RAG + "p{1.3cm}cccc"
if with_bio:
    cols.append(r"Constituents in pure culture (EC$_{50}$, $\mu$M)")
    spec += RAG + "p{2.6cm}"
caption = (
    r"\caption{Leading combinations of the full-data BRS ranking. These rankings were not reproducible under half-sampling (main text, Fig.~2), so the list is exploratory"
    + (f" (all {len(main_rows)} are {main_rows['order'].iloc[0]}s)" if one_order else "") + ". "
    r"Strain: ammonia oxidizer(s) under which the set entered the BRS top-10 list (BRS in brackets). "
    r"AUC: composite score separating high- from low-BNI accessions (" + f"{int(y.sum())}/{int(len(y) - y.sum())}"
    r"), with stratified bootstrap 95\% CI. $\Delta$AUC: difference from the best single constituent, "
    r"with paired bootstrap 95\% CI. "
    r"$p_{\mathrm{adj}}$: label-permutation $p$, max-statistic adjusted over the 30 evaluated candidate sets. "
    r"Stability: number of labelling thresholds (60th--85th percentile) at which the set ranks in the top " + str(TOP_K_STABLE) + ". "
    + (r"Constituents in pure culture: single-compound results on the listed strain(s) (Supplementary Table S7); "
       r"n.i., no inhibition; n.t., not tested; "
       + "; ".join(f"{v}, {k}" for k, v in ABBR.items() if any(k in c for c in main_rows["candidate"])) + ". "
       if with_bio else "")
    + r"For reference, the best single metabolite (" + tex(best_single["candidate"]) + r") reached AUC = "
    + f"{best_single['AUC']:.2f}" + r".}\label{tab:mixture}")
lines = [
    r"% Generated by Table3_mixture_candidates.py -- do not edit by hand",
    r"\begin{table}[!htbp]",
    caption,
    r"\footnotesize\setlength{\tabcolsep}{2.5pt}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}" + spec + r"@{\extracolsep\fill}}",
    r"\toprule",
    " & ".join(cols) + r" \\",
    r"\midrule",
]
for _, r in main_rows.iterrows():
    cells = [tex(r["candidate"])] + ([] if one_order else [r["order"]]) + [
        tex(r["prioritized_under"]),
        f"{num(r['AUC'])} [{num(r['AUC_CI_low'])}, {num(r['AUC_CI_high'])}]",
        f"{num(r['delta_AUC_vs_best_constituent'], True)} [{num(r['delta_CI_low'])}, {num(r['delta_CI_high'])}]",
        fmt_p(r["p_perm_FWER"]),
        f"{r['n_thresholds_top5']}/{len(THRESHOLDS)}",
    ] + ([tex(r["pure_culture_constituents"])] if with_bio else [])
    lines.append(" & ".join(cells) + r" \\")
lines += [r"\botrule", r"\end{tabular*}", r"\end{table}", ""]
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
print(f"✅ Saved: {OUT_TEX}")
