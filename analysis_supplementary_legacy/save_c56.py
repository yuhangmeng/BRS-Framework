# -*- coding: utf-8 -*-
"""Persist the C56 prescreen diagnostic as a supplementary table."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "supplementary" / "ST_prescreen_correlation_sign.xlsx"

summary = pd.DataFrame([
    dict(condition="Inhibition (%)-NF", n_features_tested=413, n_r_gt_0_2=93, n_r_lt_neg0_2=58,
         max_positive_r=0.590, most_negative_r=-0.405, pool_size_current=93,
         negatives_entering_abs_pool=23, pool_overlap=77, overlap_pct=77.0),
    dict(condition="Inhibition (%)-NV", n_features_tested=413, n_r_gt_0_2=44, n_r_lt_neg0_2=52,
         max_positive_r=0.502, most_negative_r=-0.384, pool_size_current=44,
         negatives_entering_abs_pool=53, pool_overlap=44, overlap_pct=44.0),
    dict(condition="Inhibition (%)-NU", n_features_tested=413, n_r_gt_0_2=69, n_r_lt_neg0_2=60,
         max_positive_r=0.566, most_negative_r=-0.389, pool_size_current=69,
         negatives_entering_abs_pool=43, pool_overlap=57, overlap_pct=57.0),
    dict(condition="Inhibition (%)-NM", n_features_tested=413, n_r_gt_0_2=51, n_r_lt_neg0_2=24,
         max_positive_r=0.446, most_negative_r=-0.345, pool_size_current=51,
         negatives_entering_abs_pool=37, pool_overlap=51, overlap_pct=51.0),
])

impact = pd.DataFrame([
    dict(level="prescreen pool (top 100 per condition)",
         description="metabolites entering BRS scoring",
         overlap="44-77% depending on condition"),
    dict(level="combination pool (union of per-condition top 10)",
         description="the pool that actually feeds the pair and triad analysis",
         overlap="32 of 35 metabolites identical (91%); 3 would enter, 3 would leave"),
])

readme = pd.DataFrame({
    "field": ["purpose", "addresses", "method", "key_result", "recommendation",
              "response_variable_note"],
    "description": [
        "Quantify the effect of prescreening on r > 0.2 versus |r| > 0.2",
        "Reviewer comment C56 (why are negatively correlated metabolites excluded?)",
        "Replicates the prescreening step of Main Method.py exactly (same input file, same "
        "numeric coercion, same cut-off); no downstream rerun was performed",
        "Negatively correlated metabolites exist (24-60 per condition at r < -0.2) but are "
        "consistently weaker than the positive tail (strongest negative -0.35 to -0.41 vs "
        "strongest positive 0.45 to 0.59). At the level that feeds the combination analysis "
        "(union of per-condition top 10), 32 of 35 metabolites are unchanged.",
        "Report this table and keep the positive-only prescreen; a full |r| rerun changes "
        "3 of 35 combination-pool metabolites and would require regenerating every downstream "
        "figure for a marginal change in the candidate set.",
        "AOI% is already expressed as PERCENT INHIBITION of ammonia oxidation. A metabolite "
        "that inhibits nitrification is therefore expected to correlate POSITIVELY with the "
        "response. A negative correlation identifies metabolites associated with reduced "
        "inhibition, which is a different biological question rather than a missed inhibitor.",
    ],
})

with pd.ExcelWriter(DEST) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    summary.to_excel(w, sheet_name="per_condition", index=False)
    impact.to_excel(w, sheet_name="impact", index=False)
print(f"wrote {DEST}")
