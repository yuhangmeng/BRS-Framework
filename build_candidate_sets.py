# -*- coding: utf-8 -*-
"""
Build the candidate-set list used by Table 3, Fig. 2, Fig. 4 and Table S7 directly from the
BRS workbooks, with the same rule as Fig. 2:
    per strain (NF, NV, NM, NU): top-10 sets by BRS (*_All sheets)
    -> union across strains -> mean BRS over the four strains (strain not in top-10 counts as 0)
    -> the N_PER_ORDER sets with the highest mean, for singles, pairs and triads.
Output mirrors Top5_single_pair_triple_BRS.xlsx [Sheet1]: first column = IDs joined by " | ",
one column per strain holding the BRS where the set is in that strain's top-10 (else 0).

    BRS_OUT_DIR=results_v2_deterministic python build_candidate_sets.py
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1) Configuration
# ==========================================
RESULTS = Path(os.environ.get("BRS_OUT_DIR", "results"))
BRS_FILES = {   # order -> (workbook, sheet kind, set column)
    "single": (RESULTS / "univariate_BRS.xlsx", "Uni", "Feature"),
    "pair": (RESULTS / "pairwise_BRS.xlsx", "Pairs", "Pair"),
    "triad": (RESULTS / "triple_BRS.xlsx", "Triplets", "Triplet"),
}
OUT_XLSX = RESULTS / "candidate_sets.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
TOPK_PER_STRAIN = 10
N_PER_ORDER = 10


def canon(s: str) -> str:
    return " | ".join(sorted(t.strip() for t in str(s).split("|")))


# ==========================================
# 2) Per-strain top-10, union, mean-BRS ranking
# ==========================================
print(f"--- Building candidate sets from {RESULTS} ---")
blocks = []
for order, (path, kind, col) in BRS_FILES.items():
    xls = pd.ExcelFile(path)
    tops = {}
    for s in STRAINS:
        d = pd.read_excel(xls, f"Inhibition (pct)-{s}_{kind}_All")
        d["key"] = d[col].map(canon)
        d = d.sort_values(["BRS", "key"], ascending=[False, True]).reset_index(drop=True)
        tops[s] = d.head(TOPK_PER_STRAIN).set_index("key")["BRS"]
    mat = pd.DataFrame(tops)[STRAINS]
    union_n = len(mat)
    mat["mean_BRS"] = mat[STRAINS].fillna(0.0).mean(axis=1)
    mat = mat.sort_values(["mean_BRS"], ascending=False).head(N_PER_ORDER)
    out = mat[STRAINS].fillna(0.0).rename(columns={s: f"Inhibition (%)-{s}" for s in STRAINS})
    out.insert(0, "Metabolite", out.index)
    out["order"] = order
    out["mean_BRS"] = mat["mean_BRS"]
    blocks.append(out.reset_index(drop=True))
    print(f"✅ {order}: union of per-strain top-{TOPK_PER_STRAIN} = {union_n} sets; kept {len(out)}")

cand = pd.concat(blocks, ignore_index=True)
with pd.ExcelWriter(OUT_XLSX) as w:
    cand.drop(columns=["order", "mean_BRS"]).to_excel(w, sheet_name="Sheet1", index=False)
    cand.to_excel(w, sheet_name="with_order", index=False)
print(f"✅ Saved: {OUT_XLSX} ({len(cand)} candidate sets)")
