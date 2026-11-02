# -*- coding: utf-8 -*-
"""
Sign of the prescreen correlation on the BRS input (default: log-scale input).
Per strain: metabolites with r > 0.2 and r < -0.2, the strongest positive and negative r, and how the prescreened
panel (at most 100 metabolites) changes when |r| > 0.2 is used instead of r > 0.2.

Output: supplementary/ST_prescreen_correlation_sign.xlsx (sheets per_condition, impact)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / os.environ.get("BRS_INPUT", "Data Preprocessing/filtered_metabolites_log.xlsx")
OUT = ROOT / "supplementary" / "ST_prescreen_correlation_sign.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
R_MIN, CAP = 0.2, 100

d = pd.read_excel(INPUT, sheet_name=0, decimal=",")
tcols = {s: [c for c in d.columns if "Inhibition" in str(c) and str(c).endswith(s)][0] for s in STRAINS}
X = d[[c for c in d.columns if c not in tcols.values()]].apply(pd.to_numeric, errors="coerce")
rows, imp = [], []
for s in STRAINS:
    r = X.corrwith(pd.to_numeric(d[tcols[s]], errors="coerce"))
    pos = r[r > R_MIN].sort_values(ascending=False).head(CAP)
    absr = r[r.abs() > R_MIN].reindex(r.abs().sort_values(ascending=False).index).dropna().head(CAP)
    rows.append({"strain": s, "n_r_gt_0.2": int((r > R_MIN).sum()), "n_r_lt_-0.2": int((r < -R_MIN).sum()),
                 "strongest_positive_r": r.max(), "strongest_negative_r": r.min()})
    imp.append({"strain": s, "panel_r_gt_0.2": len(pos), "panel_abs_r_gt_0.2": len(absr),
                "shared": len(set(pos.index) & set(absr.index)),
                "negatively_correlated_in_abs_panel": int((absr < 0).sum())})
P, I = pd.DataFrame(rows), pd.DataFrame(imp)
with pd.ExcelWriter(OUT) as w:
    pd.DataFrame([("input", str(INPUT.name)), ("panel", f"at most {CAP} metabolites with the largest r (or |r|)")],
                 columns=["field", "description"]).to_excel(w, sheet_name="README", index=False)
    P.round(3).to_excel(w, sheet_name="per_condition", index=False)
    I.to_excel(w, sheet_name="impact", index=False)
print(P.round(2).to_string(index=False))
print(I.to_string(index=False))
