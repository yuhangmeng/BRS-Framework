# -*- coding: utf-8 -*-
"""
Table 3: the stability-selected, strain-resolved consensus metabolite signature.

Members: single metabolites whose selection probability (fraction of 50 half-samples in a strain's
BRS top-10; results_stability_uni/stability_summary_uni.xlsx, log-scale input) reaches PI_SIG in at least
one strain. Leverage: largest difference in selection probability with and without a single accession
(stability_univariate_summary.py); >= LEV_FLAG marks a member that depends on one accession.
Columns: selection probability per strain, the strain(s) where pi >= PI_SIG, and the fraction of nested-CV
outer folds in which the metabolite was re-selected (results_nestedcv). Single-compound assay results are kept
out of this table (Supplementary Table S19 only).

Outputs: supplementary/ST_signature.xlsx, 论文/overleaf_v6/tables/table2_signature.tex
"""
import os

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

# ==========================================
# 1) Configuration
# ==========================================
STAB_XLSX = Path(os.environ.get("BRS_STAB_XLSX", "results_stability_uni/stability_summary_uni.xlsx"))
NESTED_XLSX = Path("results_nestedcv/nested_cv_summary.xlsx")
NAME_MAP_PATH = Path("name.xlsx")
RAW_PATH = Path("Data Preprocessing/filtered_notfound_removed.xlsx")
LABEL_PATH = Path("Data Preprocessing/filtered_metabolites.xlsx")
OUT_XLSX = Path("supplementary/ST_signature.xlsx")
OUT_TEX = Path("论文/overleaf_v6/tables/table2_signature.tex")
STRAINS = ["NF", "NV", "NM", "NU"]
PI_SIG = 0.4
LEV_FLAG = 0.7
NESTED_METHOD = f"stability pi>={PI_SIG}"
DISPLAY_FIX = {"L-TRYPTOPHAN": "L-Tryptophan", "Uric Acid": "Uric acid", "Ala-ala": "Ala-Ala",
               "Isoshaftoside": "Isoschaftoside"}
CLASS_SHORT = {"Amino acids and derivatives": "amino acid deriv.", "Amino acids": "amino acid",
               "Carbohydrates": "carbohydrate", "Flavonoids": "flavonoid", "Phenylpropanoids": "phenylpropanoid",
               "Lipids": "lipid", "Alkaloids": "alkaloid", "Nitrogenous bases": "nitrogenous base"}

# ==========================================
# 2) Signature members and annotations
# ==========================================
pi = pd.read_excel(STAB_XLSX, sheet_name="pi_single")
sig = pi[pi["pi_max"] >= PI_SIG].copy()
nm = pd.read_excel(NAME_MAP_PATH, sheet_name=0, dtype=str).set_index("ID")
names = nm["Compound name"].str.strip().to_dict()
classes = nm["CHEMICAL CLASS"].str.strip().to_dict()
sig["metabolite"] = sig["set"]
sig["base_name"] = [DISPLAY_FIX.get(names.get(k, k), names.get(k, k)) for k in sig["key"]]
sig["class"] = [CLASS_SHORT.get(classes.get(k, ""), classes.get(k, "other") or "other") for k in sig["key"]]
sig["signature_strains"] = [", ".join(s for s in STRAINS if r[f"pi_{s}"] >= PI_SIG) for _, r in sig.iterrows()]
sig["leverage"] = [max(r[f"leverage_{s}"] for s in r["signature_strains"].split(", ")) for _, r in sig.iterrows()]

# re-selection across the outer folds of the strict nested cross-validation (strict_nested_cv.py)
STRICT_XLSX = Path("supplementary/ST_strict_nested_cv.xlsx")
mem = pd.read_excel(STRICT_XLSX, sheet_name="reselection").set_index("metabolite key")["re-selected in outer folds (of 10)"] / 10
sig["nested_cv_fraction"] = [float(mem.get(str(k).strip(), 0.0)) for k in sig["key"]]
sig["first_strain"] = sig["signature_strains"].str.split(", ").str[0].map(STRAINS.index)   # same order as Fig. 4
sig = sig.sort_values(["first_strain", "pi_max"], ascending=[True, False])

# signature composite AUC (in-sample) for the caption
lab = pd.read_excel(LABEL_PATH, sheet_name=0)
bni = lab[[c for c in lab.columns if "Inhibition" in str(c)]]
y = (bni >= bni.quantile(0.75)).any(axis=1).astype(int).values
mets = [c for c in lab.columns if "Inhibition" not in str(c)]
X = pd.read_excel(RAW_PATH, sheet_name=0, decimal=",")[mets].apply(pd.to_numeric, errors="coerce")
X = X.apply(lambda s: s.where(s != 0, s[s != 0].min() / 2))
Z = np.log1p(X)
Z = (Z - Z.mean()) / Z.std()
Z.columns = [str(c).strip() for c in Z.columns]
score = Z[[str(k).strip() for k in sig["key"]]].mean(axis=1).values
r = rankdata(score)
n1, n0 = y.sum(), len(y) - y.sum()
auc_in = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
auc_cv = pd.read_excel(STRICT_XLSX, sheet_name="main").set_index("method").loc[
    "BRS + stability selection (consensus list)", "nested-CV AUC"]

cols = ["metabolite", "class", "signature_strains"] + [f"pi_{s}" for s in STRAINS] + \
       ["leverage", "nested_cv_fraction", "key"]
OUT_XLSX.parent.mkdir(exist_ok=True)
with pd.ExcelWriter(OUT_XLSX) as w:
    pd.DataFrame([("members", f"single metabolites with selection probability >= {PI_SIG} in at least one strain"),
                  ("selection probability", "fraction of 50 complementary half-samples in which the metabolite is in the strain's BRS top-10"),
                  ("leverage", f"max over accessions of pi(with) - pi(without) on the signature strain; >= {LEV_FLAG} = single-accession dependence"),
                  ("nested_cv_fraction", f"fraction of the 10 outer folds in which the metabolite was re-selected ({NESTED_METHOD})"),
                  ("signature AUC", f"in-sample {auc_in:.3f}; nested CV (pooled out-of-fold) {auc_cv:.3f}")],
                 columns=["field", "description"]).to_excel(w, sheet_name="README", index=False)
    sig[cols].round(3).to_excel(w, sheet_name="signature", index=False)
print(sig[cols[:-1]].round(2).to_string(index=False))
print(f"signature: {len(sig)} metabolites; in-sample AUC {auc_in:.3f}; nested-CV AUC {auc_cv:.3f}")


# ==========================================
# 3) LaTeX
# ==========================================
def tex(s):
    s = str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    s = s.replace("Disaccharide - C12H22O11", r"Disaccharide (C$_{12}$H$_{22}$O$_{11}$)")
    s = s.replace("6,8C-glycosylated", "6,8-C-Glycosylated").replace("gamma-", r"$\gamma$-")
    s = s.replace("N-acetyltryptophan", "N-Acetyltryptophan")
    return s.replace("Hydroxybenzenepropanoic", r"Hydroxy\-benzene\-propanoic")


RAG = r">{\raggedright\arraybackslash}"
lines = [
    r"% Generated by Table3_signature.py -- do not edit by hand",
    r"\begin{table*}[!htbp]",
    r"\caption{Reproducible single metabolites of high-BNI root exudates (consensus list). Members are the " + str(len(sig)) +
    r" metabolites whose selection probability $\pi$ (fraction of 50 half-samples in which the metabolite was in a "
    r"strain's BRS top ten) reached " + f"{PI_SIG}" + r" for at least one strain; values $\geq$ " + f"{PI_SIG}" +
    r" are in bold. Lev.: single-accession leverage, the largest difference between the selection probabilities "
    r"computed over the half-samples that contain and that lack one accession ($\geq$ " + f"{LEV_FLAG}" +
    r" indicates dependence on a single accession, marked $^\dagger$). CV: fraction of the ten outer cross-validation folds in which the metabolite was re-selected from "
    r"the training accessions alone. The composite score of the list separated high- from low-BNI "
    r"accessions with AUC " + f"{auc_in:.2f}" + r" in-sample and " + f"{auc_cv:.2f}" +
    r" under nested cross-validation.}\label{tab:signature}",
    r"\footnotesize\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}" + RAG + r"p{4.6cm}" + RAG + r"p{2.7cm}ccccccc"
    r"@{\extracolsep\fill}}",
    r"\toprule",
    r"Metabolite & Class & Strain(s) & $\pi_{\mathrm{NF}}$ & $\pi_{\mathrm{NV}}$ & $\pi_{\mathrm{NM}}$ & "
    r"$\pi_{\mathrm{NU}}$ & Lev. & CV \\",
    r"\midrule",
]
for _, r_ in sig.iterrows():
    pis = [(r"\textbf{%.2f}" % r_[f"pi_{s}"]) if r_[f"pi_{s}"] >= PI_SIG else "%.2f" % r_[f"pi_{s}"] for s in STRAINS]
    lines.append(" & ".join([tex(r_["metabolite"]), tex(r_["class"]), r_["signature_strains"]] + pis +
                            [("%.2f" % r_["leverage"]) + (r"$^\dagger$" if r_["leverage"] >= LEV_FLAG else ""),
                             "%.1f" % r_["nested_cv_fraction"]]) + r" \\")
lines += [r"\botrule", r"\end{tabular*}", r"\end{table*}", ""]
OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {OUT_XLSX}, {OUT_TEX}")
