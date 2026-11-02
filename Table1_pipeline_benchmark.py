# -*- coding: utf-8 -*-
"""
Main-text Table 1: pipeline-level benchmark under the same data splits and evaluation rules.

    (a) wheat panel: stability selection with Pearson correlation, the Elastic Net or BRS as the scorer, a single
        BRS run without stability selection, and directly enumerated combinations. Number selected and agreement:
        full-data stability selection (benchmark_pipeline_wheat.py); AUC, CI and Jaccard: strict nested
        cross-validation (strict_nested_cv.py)
    (b) simulation (60 datasets, known truth): selected true and noise features, planted pair called, interaction
        calls involving a noise feature and combinations tested (benchmark_scorers_sim.py, benchmark_pipeline_sim.py)

Outputs: 论文/overleaf_v6/tables/table1_pipeline.tex, table1_pipeline.json (cells for the Word copy)
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WHEAT = ROOT / "supplementary" / "ST_pipeline_benchmark.xlsx"
STRICT = ROOT / "supplementary" / "ST_strict_nested_cv.xlsx"
SIM_SC = ROOT / "results_sample_size" / "scorers_sim.xlsx"
SIM_PD = ROOT / "results_sample_size" / "pipeline_vs_direct.xlsx"
OUT_TEX = ROOT / "论文" / "overleaf_v6" / "tables" / "table1_pipeline.tex"
OUT_JSON = OUT_TEX.with_suffix(".json")

CAPTION = (
    "Pipeline-level benchmark under the same data splits and evaluation rules. (a) Wheat panel. Selected, "
    "metabolites with {pi} for at least one strain on all accessions; agreement, Spearman correlation of the selection "
    "probabilities between two disjoint sets of half-samples (mean over strains); AUC, strict nested cross-validation "
    "with all preprocessing, labels and selection inside the training folds (95% bootstrap interval); Jaccard, mean "
    "pairwise overlap of the lists of the ten outer folds. Directly enumerated combinations reached at most {pi_c} "
    "and are not scored, because their candidate list was built on all accessions (Section 3.2). (b) Simulation, 60 "
    "datasets with 30-200 samples and known truth. True and noise, features selected per dataset (mean; five true "
    "features); planted pair, datasets in which its interaction was called reproducible; false calls, interaction "
    "calls involving a noise feature (total); tested, combinations tested per dataset (mean). Scoring time per "
    "half-sample: Pearson {t_p} s, Elastic Net {t_e} s, BRS {t_b} s. Details in Supplementary Table S20.")


def r2(x):
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def r1(x):
    return str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


# ---- (a) wheat ----
full = pd.read_excel(WHEAT, sheet_name="wheat_full_data").set_index("scorer")
main = pd.read_excel(STRICT, sheet_name="main").set_index("method")
AGREE = "pi agreement between two disjoint sets of half-samples (mean Spearman over strains)"


def ncv(name):
    m = main.loc[name]
    return [r2(m["nested-CV AUC"]), f"{r2(m['CI low'])}–{r2(m['CI high'])}", r2(m["membership Jaccard"])]


rows_a = []
for sc, label in (("Pearson", "Pearson + stability selection"), ("Elastic Net", "Elastic Net + stability selection")):
    f = full.loc[f"{sc} + stability selection"]
    rows_a.append([label, str(int(f["selected (pi >= 0.4), all strains"])), r2(f[AGREE])]
                  + ncv(f"{sc} + stability selection (consensus list)"))
rows_a.append(["BRS without stability selection (single run, top 14)", "–", "–"] + ncv("BRS, single run (top 14)"))
f = full.loc["BRS + stability selection"]
rows_a.append(["Full pipeline (BRS + stability selection)", str(int(f["selected (pi >= 0.4), all strains"])),
               r2(f[AGREE])] + ncv("BRS + stability selection (consensus list)"))
rows_a.append(["Direct combination enumeration (BRS)", "0", "–", "n.s.", "", "–"])

# ---- (b) simulation ----
sc_all = pd.read_excel(SIM_SC, sheet_name="overall").iloc[0]
pd_by = pd.read_excel(SIM_PD, sheet_name="by_n")
pipe_tested = pd_by["pipeline: combinations tested"].mean()
rows_b = [
    ["Pearson + stability selection", r1(sc_all["Pearson + stability selection: true selected"]),
     r1(sc_all["Pearson + stability selection: noise selected"]), str(int(sc_all["Pearson pool: planted pair called"])),
     str(int(sc_all["Pearson pool: interaction calls with a noise feature"])), r1(sc_all["Pearson pool: combinations tested"])],
    ["Elastic Net + stability selection", r1(sc_all["Elastic Net + stability selection: true selected"]),
     r1(sc_all["Elastic Net + stability selection: noise selected"]), str(int(sc_all["Elastic Net pool: planted pair called"])),
     str(int(sc_all["Elastic Net pool: interaction calls with a noise feature"])), r1(sc_all["Elastic Net pool: combinations tested"])],
    ["BRS without stability selection (single run, top 10)", r1(sc_all["BRS, single run (top 10): true selected"]),
     r1(sc_all["BRS, single run (top 10): noise selected"]), "–", "–", "–"],
    ["Full pipeline (BRS + stability selection)", r1(sc_all["BRS + stability selection: true selected"]),
     r1(sc_all["BRS + stability selection: noise selected"]), str(int(round(10 * pd_by["pipeline: planted pair called"]).sum())),
     str(int(pd_by["pipeline: interaction calls with a noise feature"].sum())), r1(pipe_tested)],
    ["Direct combination enumeration (BRS top 20/10)", "–", "–", str(int(round(10 * pd_by["direct: planted pair called"]).sum())),
     str(int(pd_by["direct: interaction calls with a noise feature"].sum())), str(int(pd_by["direct: combinations tested"].mean()))],
]

tp = sc_all["Pearson: seconds per half-sample"]
cap_vals = dict(t_p="< 0.001" if tp < 0.001 else f"{tp:.3f}", t_e=r2(sc_all["Elastic Net: seconds per half-sample"]),
                t_b=str(int(round(sc_all["BRS: seconds per run"]))))
head_a = ["Method", "Selected", "Agreement", "Nested-CV AUC", "95% CI", "Jaccard"]
head_b = ["Method", "True", "Noise", "Planted pair (of 60)", "False calls", "Tested"]
OUT_JSON.write_text(json.dumps({"caption": CAPTION.format(pi="π̂ ≥ 0.4", pi_c="π̂ = 0.10 (pairs) and 0.04 (triads)", **cap_vals),
                                "panel_a": "(a) Wheat panel (44 accessions)", "head_a": head_a, "rows_a": rows_a,
                                "panel_b": "(b) Simulation (60 datasets, known truth)", "head_b": head_b, "rows_b": rows_b},
                               ensure_ascii=False, indent=1), encoding="utf-8")

tex = lambda s: s.replace("–", "--").replace("%", r"\%")
cap_tex = (CAPTION.format(pi=r"$\hat\pi \geq 0.4$", pi_c=r"$\hat\pi = 0.10$ (pairs) and 0.04 (triads)",
                          **{k: v.replace("<", "$<$") for k, v in cap_vals.items()})
           .replace("30-200", "30--200").replace("%", r"\%").replace("(Section 3.2)", r"(Section~\ref{sec:direct})"))
lines = ["% Generated by Table1_pipeline_benchmark.py -- do not edit by hand", r"\begin{table}[!htbp]",
         r"\caption{" + cap_tex + r"}\label{tab:benchmark}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}lccccc@{\extracolsep\fill}}", r"\toprule",
         r"\multicolumn{6}{l}{\textit{(a) Wheat panel (44 accessions)}} \\", " & ".join(tex(c) for c in head_a) + r" \\",
         r"\midrule"]
lines += [" & ".join(tex(c) for c in r) + r" \\" for r in rows_a]
lines += [r"\midrule", r"\multicolumn{6}{l}{\textit{(b) Simulation (60 datasets, known truth)}} \\",
          " & ".join(head_b) + r" \\", r"\midrule"]
lines += [" & ".join(tex(c) for c in r) + r" \\" for r in rows_b]
lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
