# -*- coding: utf-8 -*-
"""
Table 2: reproducible single metabolites of high-BNI root exudates (consensus list): metabolites with selection
probability pi >= 0.4 for at least one strain, with pi per strain, single-accession leverage on the strain(s) where
pi >= 0.4 (dagger: >= 0.7) and the fraction of the ten outer cross-validation folds in which the metabolite was
re-selected from the training accessions alone. The caption's AUCs: composite score of the list, in-sample and under
the strict nested cross-validation.

Inputs: data/preprocessing/*.xlsx, data/metabolite_names.xlsx, results/stability_selection.xlsx (Step 2),
        results/nested_cv.xlsx (Step 3)
Output: figures/output/Table2_reproducible_metabolites.md, .csv
"""
import pandas as pd

from common import RES, STRAINS, auc, labels, names, pi_single, wheat, write_table

PI_SIG, LEV_FLAG = 0.4, 0.7
DISPLAY = {"6,8C-glycosylated": "6,8-C-Glycosylated", "gamma-": "γ-", "N-acetyltryptophan": "N-Acetyltryptophan"}
CLASS_SHORT = {"Amino acids and derivatives": "amino acid deriv.", "Amino acids": "amino acid",
               "Carbohydrates": "carbohydrate", "Flavonoids": "flavonoid", "Phenylpropanoids": "phenylpropanoid",
               "Lipids": "lipid", "Alkaloids": "alkaloid", "Nitrogenous bases": "nitrogenous base"}

P = pi_single()
_, classes = names()
sig = P[P["pi_max"] >= PI_SIG].copy()
sig["strains"] = [", ".join(s for s in STRAINS if r[f"pi_{s}"] >= PI_SIG) for _, r in sig.iterrows()]
sig["lev"] = [max(r[f"leverage_{s}"] for s in r["strains"].split(", ")) for _, r in sig.iterrows()]
resel = pd.read_excel(RES / "nested_cv.xlsx", sheet_name="reselection")
resel = resel.set_index(resel["metabolite key"].astype(str).str.strip())["re-selected in outer folds (of 10)"] / 10
sig["cv"] = [float(resel.get(k, 0.0)) for k in sig.index]
sig["first"] = sig["strains"].str.split(", ").str[0].map(STRAINS.index)
sig = sig.sort_values(["first", "pi_max"], ascending=[True, False])

Z, aoi = wheat()
auc_in = auc(Z[list(sig.index)].mean(axis=1).values, labels(aoi, "overall"))
auc_cv = pd.read_excel(RES / "nested_cv.xlsx", sheet_name="main").set_index("method").loc[
    "BRS + stability selection (consensus list)", "nested-CV AUC"]


def display(s):
    for a, b in DISPLAY.items():
        s = s.replace(a, b)
    return s


rows = []
for k, r in sig.iterrows():
    pis = [f"**{r[f'pi_{s}']:.2f}**" if r[f"pi_{s}"] >= PI_SIG else f"{r[f'pi_{s}']:.2f}" for s in STRAINS]
    rows.append([display(r["set"]), CLASS_SHORT.get(classes.get(k, ""), classes.get(k, "other") or "other"), r["strains"]]
                + pis + [f"{r['lev']:.2f}" + ("†" if r["lev"] >= LEV_FLAG else ""), f"{r['cv']:.1f}"])
write_table("Table2_reproducible_metabolites", "Table 2. Reproducible single metabolites (consensus list)",
            ["Metabolite", "Class", "Strain(s)", "π NF", "π NV", "π NM", "π NU", "Lev.", "CV"], rows,
            f"{len(sig)} metabolites with π ≥ {PI_SIG} for at least one strain (bold). Composite score of the list: "
            f"AUC {auc_in:.2f} in-sample and {auc_cv:.2f} under nested cross-validation.")
