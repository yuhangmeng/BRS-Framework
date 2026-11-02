# -*- coding: utf-8 -*-
"""
Supplementary Table S16: candidate cards for experimental follow-up of the gated pairs (Step 4).

One row per pair of each strain's gated pool, collecting the evidence of every pipeline step:
    members        selection probability pi for the strain, single-accession leverage
    Step 4         pi_comp / pi_int with their label-permutation fractions, full-data R2 gains
    discrimination in-sample composite AUC (descriptive: the pool was selected on the same accessions), the better
                   member's AUC, cross-validated logistic AUC without and with the product term
    evidence       read against the synthetic positive control (results/gated_synthetic.xlsx):
                   complementarity  pi_comp >= mean of true, non-interacting feature pairs  -> "as true complementary pairs"
                                    pi_comp >= MIN_PI                                       -> "recurrent"
                   interaction      pi_int >= mean of the planted interaction               -> "at planted-interaction level"
                                    pi_int >= mean of true, non-interacting feature pairs   -> "above non-interacting pairs"
                                    otherwise                                               -> "not distinguishable"
    suggested test factorial design against Bliss independence, pair versus members, or members only
The procedure-level estimate (best pair chosen inside training folds) is reported per strain.

Inputs: supplementary/ST_gated_combinations.xlsx, ST_gated_combination_auc.xlsx, ST_incremental_signature.xlsx,
        results/gated_synthetic.xlsx
Output: supplementary/ST_candidate_cards.xlsx
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SUPP = ROOT / "supplementary"
OUT = SUPP / "ST_candidate_cards.xlsx"
GUILD = {"NF": "AOA", "NV": "AOA", "NM": "AOB", "NU": "AOB"}
MIN_PI = 0.15          # about four times the largest label-permutation fraction (0.04); edges of Fig. S7

gated = pd.read_excel(SUPP / "ST_gated_combinations.xlsx", sheet_name="combinations")
auc = pd.read_excel(SUPP / "ST_gated_combination_auc.xlsx", sheet_name="fixed_sets")
nested = pd.read_excel(SUPP / "ST_gated_combination_auc.xlsx", sheet_name="nested")
wet = pd.read_excel(SUPP / "ST_incremental_signature.xlsx", sheet_name="wetlab_lists")
syn = pd.read_excel(ROOT / "results" / "gated_synthetic.xlsx", sheet_name="by_kind")

ref = syn[syn.order == "pair"].set_index("kind")
COMP_TRUE = float(ref.loc["true features, no interaction", "pi_comp_mean"])
INT_TRUE = float(ref.loc["true features, no interaction", "pi_int_mean"])
INT_PLANTED = float(ref.loc["planted pair", "pi_int_mean"])
print(f"reference: pi_comp true pairs {COMP_TRUE:.2f}; pi_int true non-interacting {INT_TRUE:.2f}, "
      f"planted {INT_PLANTED:.2f}")

members = lambda c: " | ".join(sorted(c.split(" + ")))
auc = auc[auc.order == "pair"].assign(m=lambda d: d.combination.map(members))
pairs = gated[gated.order == "pair"].assign(m=lambda d: d.combination.map(members))
pairs = pairs.merge(auc[["strain", "m", "AUC_composite", "CI_low", "CI_high", "best_member", "AUC_best_member",
                         "delta_vs_best_member", "delta_CI_low", "delta_CI_high", "AUC_CV_logistic_additive",
                         "AUC_CV_logistic_interaction"]], on=["strain", "m"], how="left")


def member_info(strain, name):
    r = wet[(wet.strain == strain) & (wet.metabolite_added == name)].iloc[0]
    lev = f"{r.leverage:.2f}" + (" (single-accession)" if r.leverage >= 0.7 else "")
    return f"{name}: pi {r.pi_strain:.2f}, leverage {lev}"


def comp_evidence(p):
    return "as true complementary pairs" if p >= COMP_TRUE else ("recurrent" if p >= MIN_PI else "not recurrent")


def int_evidence(p):
    if p >= INT_PLANTED:
        return "at planted-interaction level"
    return "above non-interacting pairs" if p >= INT_TRUE else "not distinguishable"


def test(r):
    if r.pi_int >= INT_TRUE:
        return "factorial (checkerboard) design against Bliss independence"
    if r.pi_comp >= MIN_PI:
        return "pair versus members at equal total concentration (additivity)"
    return "members only, within the cumulative mixture"


rows = []
for _, r in pairs.iterrows():
    a, b = r.combination.split(" + ")
    rows.append({
        "strain": r.strain, "guild": GUILD[r.strain], "pair": r.combination,
        "member_1": member_info(r.strain, a), "member_2": member_info(r.strain, b),
        "pi_comp": r.pi_comp, "pi_comp_permutation": r.pi_comp_null, "complementarity_evidence": comp_evidence(r.pi_comp),
        "pi_int": r.pi_int, "pi_int_permutation": r.pi_int_null, "interaction_evidence": int_evidence(r.pi_int),
        "R2_gain_complementarity_full": r.comp_gain_full, "R2_gain_interaction_full": r.int_gain_full,
        "AUC_in_sample_descriptive": r.AUC_composite, "AUC_CI_low": r.CI_low, "AUC_CI_high": r.CI_high,
        "better_member": r.best_member, "AUC_better_member": r.AUC_best_member,
        "delta_AUC": r.delta_vs_best_member, "delta_CI_low": r.delta_CI_low, "delta_CI_high": r.delta_CI_high,
        "AUC_CV_logistic_additive": r.AUC_CV_logistic_additive,
        "AUC_CV_logistic_with_product": r.AUC_CV_logistic_interaction,
        "suggested_test": test(r)})
cards = pd.DataFrame(rows)
cards["_rank"] = cards[["pi_comp", "pi_int"]].max(axis=1)
cards = (cards.sort_values(["strain", "_rank"], ascending=[True, False], key=lambda s: s.map(
    {"NF": 0, "NV": 1, "NM": 2, "NU": 3}) if s.name == "strain" else s).drop(columns="_rank"))

proc = nested[nested.method.isin(["best single", "best pair", "best pair, logistic interaction"])]
readme = pd.DataFrame([
    ("purpose", "evidence per gated pair for planning mixture and synergy assays; a computational inference, not a "
                "demonstration of joint action"),
    ("pi_comp / pi_int", "fraction of 50 half-samples in which the additive (complementarity) or product-term "
                         "(interaction) gain in out-of-fold R2 exceeded 0.02; permutation: same fraction under 20 "
                         "label permutations per half-sample"),
    ("evidence", f"read against the synthetic positive control: true complementary (non-interacting) feature pairs "
                 f"reached mean pi_comp {COMP_TRUE:.3f} and pi_int {INT_TRUE:.3f}; the planted interaction reached "
                 f"mean pi_int {INT_PLANTED:.3f}; recurrent: pi >= {MIN_PI}"),
    ("AUC_in_sample_descriptive", "composite (mean of autoscaled log intensities) against the strain's own label, on "
                                  "the accessions from which the pool was selected: describes the candidate, does "
                                  "not estimate performance; see nested_summary"),
    ("nested_summary", "best single metabolite and best pair chosen inside the training folds of the nested CV; "
                       "held-out AUC with bootstrap CI"),
    ("suggested_test", f"factorial design if pi_int >= {INT_TRUE:.3f}; pair versus members if pi_comp >= {MIN_PI}; "
                       "otherwise members within the cumulative mixture of Table 3"),
], columns=["field", "description"])
with pd.ExcelWriter(OUT) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    cards.round(3).to_excel(w, sheet_name="cards", index=False)
    proc.round(3).to_excel(w, sheet_name="nested_summary", index=False)
    syn.round(3).to_excel(w, sheet_name="synthetic_reference", index=False)
print(f"Saved {OUT}")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
show = cards[cards.suggested_test != "members only, within the cumulative mixture"]
print(show[["strain", "pair", "pi_comp", "complementarity_evidence", "pi_int", "interaction_evidence",
            "AUC_in_sample_descriptive", "suggested_test"]].to_string(index=False))
