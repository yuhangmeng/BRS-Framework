# -*- coding: utf-8 -*-
"""
Table 3: candidate mixtures for strain-specific assays. For each strain, metabolites enter the mixture in order of
decreasing selection probability for that strain, up to the size of the final mixture (chosen on all accessions with
the rule that the nested cross-validation applies inside each training set). Lev.: single-accession leverage
(dagger: >= 0.7). Double dagger: below the selection threshold (pi < 0.4; exploratory). AUC: nested-CV AUC of the
mixture up to and including that metabolite, at this fixed size (descriptive).

Inputs: results/stability_selection.xlsx (Step 2), results/nested_cv.xlsx (Step 3)
Output: figures/output/Table3_mixtures.md, .csv
"""
import pandas as pd

from common import GUILD, RES, STRAINS, pi_single, write_table

LEV_FLAG, PI_THR = 0.7, 0.4
K_COL = "k for the final mixture (inner rule on 25 full-data pairs)"
DISPLAY = {"6,8C-glycosylated": "6,8-C-Glycosylated", "gamma-": "γ-", "N-acetyltryptophan": "N-Acetyltryptophan",
           "Fatty acid - C18H30O3": "Fatty acid (C₁₈H₃₀O₃)", "Flavone - C17H14O6": "Flavone (C₁₇H₁₄O₆)",
           "similar to Coniferyl aldehyde": "Coniferyl aldehyde-like compound"}

P = pi_single()
final = pd.read_excel(RES / "nested_cv.xlsx", sheet_name="final_mixtures").set_index("target")
curve = pd.read_excel(RES / "nested_cv.xlsx", sheet_name="curves").set_index(["target", "k"])["nested-CV AUC (fixed k)"]


def display(s):
    for a, b in DISPLAY.items():
        s = s.replace(a, b)
    return s


rows = []
for s in STRAINS:
    members = [m.strip() for m in final.loc[s, "members"].split(";")]
    assert len(members) == int(final.loc[s, K_COL])
    for step, key in enumerate(members, 1):
        pi, lev = P.loc[key, f"pi_{s}"], P.loc[key, f"leverage_{s}"]
        rows.append([f"{s} ({GUILD[s]})" if step == 1 else "", step,
                     display(P.loc[key, "set"]) + ("‡" if pi < PI_THR else ""), f"{pi:.2f}",
                     "–" if pd.isna(lev) else f"{lev:.2f}" + ("†" if lev >= LEV_FLAG else ""),
                     f"{curve.loc[(s, step)]:.2f}"])
write_table("Table3_mixtures", "Table 3. Candidate mixtures for strain-specific assays",
            ["Strain", "Step", "Metabolite", "π", "Lev.", "AUC"], rows,
            "‡ below the selection threshold (π < 0.4; exploratory). † leverage ≥ 0.7.")
