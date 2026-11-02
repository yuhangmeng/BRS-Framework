# -*- coding: utf-8 -*-
"""Persist the KOMP imbalance sweep and the reproducibility check."""
import pandas as pd
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("LEGACY_SCRATCH", ROOT / "legacy_scratch"))   # intermediate files of the original runs
SUP = ROOT / "supplementary"

xl = pd.ExcelFile(SCRATCH / "ST_komp_imbalance.xlsx")
sheets = {s: xl.parse(s) for s in xl.sheet_names}

readme = pd.DataFrame({"field": [
    "purpose", "addresses", "design", "ground_truth",
    "key_result_recall20", "key_result_recall50", "how_to_use_it", "caveat_1", "caveat_2"],
    "description": [
        "Sensitivity of feature-ranking recovery to class imbalance in the KOMP Pmm2 dataset",
        "Reviewer comment C86: 'KOMP design imbalance: 6 vs 40??? This is an opportunity to check "
        "different level of imbalance'",
        "The 6 Pmm2 knockouts were held fixed and the wildtype pool was subsampled to 6, 12, 24 "
        "and 40 animals, 25 random resamples per ratio (the 6:40 case is the full pool and so has "
        "a single deterministic draw). All seven ranking methods were rerun at every resample.",
        "Recomputed inside each resample (Mann-Whitney U, Benjamini-Hochberg FDR < 0.10, top-30 "
        "fallback), so no full-data information leaks into the subsampled comparisons",
        "BRS 0.54 / 0.58 / 0.60 / 0.50 across 6:6, 6:12, 6:24, 6:40; the linear methods (LASSO, "
        "Elastic Net) stay near 0.24 at every ratio",
        "BRS 0.833 / 0.835 / 0.845 / 0.80, the flattest profile of all seven methods; Random "
        "Forest ranges 0.60 to 0.84 and permutation importance 0.55 to 0.78 over the same range",
        "This converts the 6-vs-40 imbalance from a limitation into evidence for the robustness "
        "claim the paper makes: BRS is the method least affected by the imbalance ratio",
        "The ground-truth set is redefined within each resample, so recall is measured against "
        "that resample's own reference rather than a fixed list",
        "BRS is stochastic (see sheet reproducibility): single-run differences below about 0.10 "
        "Recall@20 are within its own run-to-run spread and should not be interpreted",
    ]})

repro = pd.DataFrame({
    "run": [0, 1, 2, 3, 4],
    "recall_5":  [0.25, 0.25, 0.25, 0.25, 0.25],
    "recall_10": [0.35, 0.35, 0.35, 0.30, 0.30],
    "recall_20": [0.55, 0.60, 0.50, 0.60, 0.50],
    "recall_50": [0.85, 0.85, 0.85, 0.85, 0.70],
})
repro_note = pd.DataFrame({"field": [
    "finding", "manuscript_values", "observed_range", "cause", "consequence", "fix"],
    "description": [
        "Five runs of the published BRS pipeline on identical KOMP data give different results",
        "Recall@20 = 0.60 and Recall@50 = 0.85 (both at the top of the observed range)",
        "Recall@10 0.30-0.35, Recall@20 0.50-0.60, Recall@50 0.70-0.85",
        "No random seed is fixed for the MLP initialisation, the Elastic Net cross-validation "
        "splits or the bootstrap resampling",
        "The reported margin over mutual information (0.60 vs 0.55 at Recall@20) is smaller than "
        "BRS's own run-to-run spread, so 'BRS achieved the highest Recall@20' is not supported "
        "by a single run. A reviewer rerunning the deposited code will most often obtain lower "
        "values than those printed in the manuscript.",
        "Set an explicit seed throughout and report the mean and standard deviation over repeated "
        "runs rather than a single draw",
    ]})

out = SUP / "ST_komp_imbalance.xlsx"
with pd.ExcelWriter(out) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    sheets["recall20_matrix"].to_excel(w, sheet_name="recall20_matrix", index=False)
    sheets["recall50_matrix"].to_excel(w, sheet_name="recall50_matrix", index=False)
    sheets["summary"].to_excel(w, sheet_name="summary", index=False)
    sheets["per_resample"].to_excel(w, sheet_name="per_resample", index=False)
    repro_note.to_excel(w, sheet_name="reproducibility_note", index=False)
    repro.to_excel(w, sheet_name="reproducibility", index=False)
print(f"wrote {out}")
print(f"sheets: {pd.ExcelFile(out).sheet_names}")
print(f"\nper-resample rows: {len(sheets['per_resample'])}")
