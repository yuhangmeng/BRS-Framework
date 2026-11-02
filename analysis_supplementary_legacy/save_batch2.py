# -*- coding: utf-8 -*-
"""Persist the completed batch-2 analyses as supplementary tables."""
import pandas as pd, numpy as np, shutil
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("LEGACY_SCRATCH", ROOT / "legacy_scratch"))   # intermediate files of the original runs
SUP = ROOT / "supplementary"

# ---------------------------------------------------------------- weight grid
g_new = pd.read_csv(SCRATCH / "grid36" / "sensitivity_table.csv")
g_old = pd.read_csv(ROOT / "results" / "benchmark_comprehensive" / "sensitivity_table.csv")

abg_new = g_new[g_new.weight_type.str.contains("a,b,g")]
abg_old = g_old[g_old.weight_type.str.contains("a,b,g")]
present_old = {(round(a, 2), round(b, 2), round(c, 2))
               for a, b, c in abg_old[["alpha", "beta", "gamma"]].values}
abg_new = abg_new.copy()
abg_new["in_published_run"] = [
    (round(a, 2), round(b, 2), round(c, 2)) in present_old
    for a, b, c in abg_new[["alpha", "beta", "gamma"]].values]

ref = abg_new[(abg_new.alpha == 0.1) & (abg_new.beta == 0.2) & (abg_new.gamma == 0.7)]
default = abg_new[(abg_new.alpha == 0.2) & (abg_new.beta == 0.2) & (abg_new.gamma == 0.6)]
ref_top5 = set(str(default.iloc[0]["top5"]).split("; ")) if len(default) else set()
abg_new["jaccard_top5_vs_default"] = [
    round(len(set(str(t).split("; ")) & ref_top5) / len(set(str(t).split("; ")) | ref_top5), 3)
    for t in abg_new["top5"]]

readme = pd.DataFrame({"field": [
    "purpose", "addresses", "what_was_wrong", "grid_definition",
    "result_recall", "result_ranking", "how_to_report"],
    "description": [
        "Complete (alpha, beta, gamma) weight-sensitivity grid for the univariate BRS",
        "Reviewer comment C78 ('What happened in 35?? A table or figure showing BRS output "
        "variance across weight configurations should appear, not just this claim')",
        "The published run used np.arange(0.1, 0.8, 0.1) in benchmark_comprehensive.py:983-984. "
        "numpy excludes the upper bound, so alpha = 0.8 and beta = 0.8 were never evaluated and "
        "the grid contained 34 of its 36 points. The two missing points, (0.1, 0.8, 0.1) and "
        "(0.8, 0.1, 0.1), are the two most extreme configurations. Re-running with "
        "np.arange(0.1, 0.9, 0.1) restores the full 36-point grid.",
        "All (alpha, beta, gamma) on a 0.1 step with each weight >= 0.1 and alpha+beta+gamma = 1",
        "Recall@10 for the planted signals is stable across the grid; this is the claim the "
        "manuscript can support",
        "The identity of the top-ranked features is NOT stable (mean pairwise Jaccard of the "
        "top-10 lists = 0.624); the manuscript should not claim that rankings are invariant",
        "Report stability of signal recovery, not stability of the exact ranking",
    ]})

out1 = SUP / "ST_weight_grid36.xlsx"
with pd.ExcelWriter(out1) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    abg_new.to_excel(w, sheet_name="abg_grid_36", index=False)
    g_new[g_new.weight_type.str.contains("Model")].to_excel(w, sheet_name="model_weights", index=False)
print(f"wrote {out1}")
print(f"   grid points: {len(abg_new)} (was {len(abg_old)});  newly added: {(~abg_new.in_published_run).sum()}")
print(abg_new[~abg_new.in_published_run][["alpha", "beta", "gamma", "recall_5", "recall_10"]].to_string(index=False))
print(f"   Recall@10 across the full grid: mean {abg_new.recall_10.mean():.3f}, "
      f"range {abg_new.recall_10.min():.2f}-{abg_new.recall_10.max():.2f}")

# ------------------------------------------------------------------- CV table
shutil.copy(SCRATCH / "ST_cross_validated_auc.xlsx", SUP / "ST_cross_validated_auc.xlsx")
cv = pd.read_excel(SUP / "ST_cross_validated_auc.xlsx", sheet_name="summary")
readme2 = pd.DataFrame({"field": [
    "purpose", "addresses", "design", "key_result", "caveat"],
    "description": [
        "Cross-validated AUC with candidate selection performed inside the fold",
        "Reviewer comments C99 ('Does this need cross validations?') and C69 (selection bias)",
        "Repeated stratified 5-fold (20 repeats, 100 folds). In each training split the "
        "highest-AUC candidate set is chosen, then scored on the held-out split. The composite "
        "score itself uses no labels, so the published per-sample scores were reused.",
        "The in-sample optimism is small (0.028 for combinations, 0.044 for single metabolites), "
        "so the reported AUC is not an overfitting artefact. However the paired difference "
        "between the best combination and the best single metabolite has a 95% interval of "
        "[-0.24, +0.46] and the combination wins in only 64% of folds, so the combination "
        "advantage is NOT statistically supported at n = 44.",
        "The 30 candidate sets were themselves derived from full-data BRS ranking, so one outer "
        "layer of selection remains outside the cross-validation loop. The numbers here are "
        "therefore still mildly optimistic.",
    ]})
pairs = pd.read_excel(SCRATCH / "ST_cross_validated_auc.xlsx", sheet_name="paired_differences")
with pd.ExcelWriter(SUP / "ST_cross_validated_auc.xlsx") as w:
    readme2.to_excel(w, sheet_name="README", index=False)
    cv.to_excel(w, sheet_name="summary", index=False)
    pairs.to_excel(w, sheet_name="paired_differences", index=False)
print(f"\nwrote {SUP / 'ST_cross_validated_auc.xlsx'}")
print(cv.drop(columns=[c for c in cv.columns if "most_selected" in c]).to_string(index=False))
