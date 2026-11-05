# -*- coding: utf-8 -*-
"""
Table 1: benchmarking predictive performance, feature-selection reliability and interaction detection. The panels
differ in task, unit and denominator, so values are comparable within a panel but not across panels.

(a) wheat panel: metabolites selected and agreement of stability selection with each scorer (benchmark output),
    nested-CV AUC with 95% CI and membership Jaccard from the strict nested cross-validation (Step 3)
(b) synthetic data, feature selection: true and noise features selected per dataset (mean of 60 datasets), recall =
    100 x true / 5 and purity = 100 x true / (true + noise), both from the unrounded means
(c) synthetic data, interaction detection: planted pair called (n of 60), interaction calls involving a noise feature
    (total), combinations tested (mean per dataset)

Inputs: results/nested_cv.xlsx (sheet main), results/source_data/table1_source_data.xlsx (Supplementary Table S20)
Output: figures/output/Table1_benchmark.md, .csv
"""
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from common import RES, SOURCE, write_table

SRC = SOURCE / "table1_source_data.xlsx"
N_DATASETS, N_TRUE = 60, 5


def rnd(x, q):
    return str(Decimal(repr(float(x))).quantize(Decimal(q), rounding=ROUND_HALF_UP))


# ---- (a) wheat ----
full = pd.read_excel(SRC, sheet_name="wheat_stability_selection").set_index("scorer")
main = pd.read_excel(RES / "nested_cv.xlsx", sheet_name="main").set_index("method")
AGREE = "pi agreement between two disjoint sets of half-samples (mean Spearman over strains)"
SEL = "selected (pi >= 0.4), all strains"
rows_a = []
for label, full_key, ncv_key in (
        ("Pearson + stability selection", "Pearson + stability selection", "Pearson + stability selection (consensus list)"),
        ("Elastic Net + stability selection", "Elastic Net + stability selection", "Elastic Net + stability selection (consensus list)"),
        ("BRS without stability selection (top 14)", None, "BRS, single run (top 14)"),
        ("Full pipeline (BRS + stability selection)", "BRS + stability selection", "BRS + stability selection (consensus list)")):
    m = main.loc[ncv_key]
    auc_ci = f"{rnd(m['nested-CV AUC'], '0.01')} ({rnd(m['CI low'], '0.01')}–{rnd(m['CI high'], '0.01')})"
    jac = rnd(m["membership Jaccard"], "0.01")
    if full_key:
        f = full.loc[full_key]
        rows_a.append([label, str(int(f[SEL])), auc_ci, jac, rnd(f[AGREE], "0.01")])
    else:
        rows_a.append([label, "–", auc_ci, jac, "–"])

# ---- (b), (c) simulation: totals over the 60 datasets, rounded only for display ----
sc = pd.read_excel(SRC, sheet_name="simulation_scorers")
pv = pd.read_excel(SRC, sheet_name="simulation_pipeline_vs_direct")
assert len(sc) == len(pv) == N_DATASETS and (sc[["n", "seed"]].values == pv[["n", "seed"]].values).all()
tot = lambda d, col: float(d[col].astype(float).sum())
rows_b = []
for label, key in (("Pearson + stability selection", "Pearson + stability selection"),
                   ("Elastic Net + stability selection", "Elastic Net + stability selection"),
                   ("BRS without stability selection (top 10)", "BRS, single run (top 10)"),
                   ("Full pipeline (BRS + stability selection)", "BRS + stability selection")):
    T, Z = tot(sc, f"{key}: true selected"), tot(sc, f"{key}: noise selected")
    rows_b.append([label, rnd(T / N_DATASETS, "0.1"), rnd(100 * T / (N_TRUE * N_DATASETS), "0.1"),
                   rnd(Z / N_DATASETS, "0.1"), rnd(100 * T / (T + Z), "0.1")])


def det(k):
    k = int(k)
    return f"{k}/{N_DATASETS} ({rnd(100 * k / N_DATASETS, '0.1')}%)"


rows_c = [
    ["Pearson + stability selection", det(tot(sc, "Pearson pool: planted pair called")),
     str(int(tot(sc, "Pearson pool: interaction calls with a noise feature"))),
     rnd(tot(sc, "Pearson pool: combinations tested") / N_DATASETS, "0.1")],
    ["Elastic Net + stability selection", det(tot(sc, "Elastic Net pool: planted pair called")),
     str(int(tot(sc, "Elastic Net pool: interaction calls with a noise feature"))),
     rnd(tot(sc, "Elastic Net pool: combinations tested") / N_DATASETS, "0.1")],
    ["Full pipeline (BRS + stability selection)", det(tot(pv, "pipeline: planted pair called")),
     str(int(tot(pv, "pipeline: interaction calls with a noise feature"))),
     rnd(tot(pv, "pipeline: combinations tested") / N_DATASETS, "0.1")],
    ["Direct combination enumeration (BRS top 20/10)", det(tot(pv, "direct: planted pair called")),
     str(int(tot(pv, "direct: interaction calls with a noise feature"))), "310"],
]
assert set(pv["direct: combinations tested"]) == {310}      # 190 pairs of the top 20 + 120 triads of the top 10
tp = sc["Pearson: seconds per half-sample"].mean()
times = (f"Scoring time per half-sample: Pearson {'< 0.001' if tp < 0.001 else rnd(tp, '0.001')} s, Elastic Net "
         f"{rnd(sc['Elastic Net: seconds per half-sample'].mean(), '0.01')} s, BRS {round(sc['BRS: seconds per run'].mean())} s.")

write_table("Table1a_wheat", "Table 1(a). Wheat panel: predictive performance and selection reproducibility",
            ["Method", "Selected metabolites (n)", "Nested-CV AUC (95% CI)", "Jaccard (mean)", "Agreement (Spearman ρ)"],
            rows_a, "44 accessions; selected = π ≥ 0.4 for at least one strain; nested-CV AUC with stratified bootstrap 95% CI.")
write_table("Table1b_selection", "Table 1(b). Synthetic data: feature-selection accuracy",
            ["Method", "True features recovered (mean of 5)", "True-feature recall (%)", "Noise features selected (mean)",
             "Selected-set purity (%)"], rows_b,
            "60 datasets. Recall = 100 × true / 5; purity = 100 × true / (true + noise), from the unrounded means. " + times)
write_table("Table1c_interaction", "Table 1(c). Synthetic data: interaction detection and search efficiency",
            ["Method", "Planted pair detected (n/60, %)", "False interaction calls (total)",
             "Combinations tested (mean per dataset)"], rows_c,
            "60 datasets. False interaction calls: interaction calls involving a noise feature, summed over the datasets.")
