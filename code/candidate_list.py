# -*- coding: utf-8 -*-
"""
Candidate list for experiments (Supplementary Table S16 of the manuscript): computational evidence only.

Part A  single-metabolite candidates: reproducible metabolites (tier 1, pi >= 0.4 for the strain) and exploratory
        candidates below the threshold (tier 3): members of the final mixtures and of exploratory combination pools
Part B  every pair and triad of the combination test: tier 2 = pair of tier 1 metabolites whose complementarity
        recurred as often as that of truly complementary features, or whose interaction gain exceeded that of
        non-interacting feature pairs, in the synthetic positive control of the manuscript (reference values below);
        tier 3 = combinations with exploratory members; other combinations of tier 1 metabolites: not prioritized
Tiers are descriptive categories of computational evidence, not a ranking of inhibitory potency or the probability
that an assay succeeds. Within a strain, metabolites follow the selection-probability order of the mixtures.

    python code/candidate_list.py      (needs the outputs of stability_selection.py, nested_cv.py, combination_test.py)
Output: results/candidate_list.xlsx (README, partA_single_metabolites, partB_combinations)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "candidate_list.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
PI_THR, LEV_FLAG, MIN_PI, CV_HALF = 0.4, 0.7, 0.15, 5
NAME = {"NF": "NF (Ca. Nitrosocosmicus franklandianus, AOA)", "NV": "NV (Nitrososphaera viennensis, AOA)",
        "NM": "NM (Nitrosospira multiformis, AOB)", "NU": "NU (Nitrosomonas ureae, AOB)"}
NE = "Not evaluated"
MARKER = "statistical association in 44 accessions only: may be a BNI-associated marker rather than an active inhibitor"


# class labels in the short form of main-text Table 2 (the source list spells "Carboxilic")
CLASS = {"Flavonoids": "flavonoid", "Phenylpropanoids": "phenylpropanoid", "Amino acids and derivatives": "amino acid deriv.",
         "Lipids": "lipid", "Carboxilic acids": "carboxylic acid", "Carboxylic acids": "carboxylic acid", "Amines": "amine",
         "Nitrogenous bases": "nitrogenous base"}


def pretty(s):
    s = str(s).replace("6,8C-glycosylated", "6,8-C-Glycosylated").replace("gamma-", "γ-")
    return s.replace("Fatty acid - C18H30O3", "Fatty acid (C18H30O3)").replace("N-acetyltryptophan", "N-Acetyltryptophan")


# mean recurrence in the synthetic positive control of the manuscript (Supplementary Table S15): truly complementary
# feature pairs without interaction (complementarity and interaction) and the planted product interaction
COMP_TRUE, INT_TRUE, INT_PLANTED = 0.490, 0.184, 0.328
stab = pd.read_excel(RES / "stability_selection.xlsx", sheet_name="pi_single")
stab = stab.set_index(stab["key"].astype(str).str.strip())
lev = pd.read_excel(RES / "stability_selection.xlsx", sheet_name="leverage")
lev["key"] = lev["key"].astype(str).str.strip()
lev = lev.set_index(["key", "strain"])
final = pd.read_excel(RES / "nested_cv.xlsx", sheet_name="final_mixtures").set_index("target")
K = "k for the final mixture (inner rule on 25 full-data pairs)"
resel = pd.read_excel(RES / "nested_cv.xlsx", sheet_name="reselection")
pools = pd.read_excel(RES / "combination_test.xlsx", sheet_name="pools").set_index("strain")
combos = pd.read_excel(RES / "combination_test.xlsx", sheet_name="combinations")
# per strain: metabolites in selection-probability order (ties by the full-data BRS), as for the mixtures
wet = []
for s_ in STRAINS:
    o = stab[stab[f"pi_{s_}"] > 0].sort_values([f"pi_{s_}", f"BRS_full_{s_}"], ascending=False, kind="mergesort")
    for step, (key, r) in enumerate(o.iterrows(), 1):
        wet.append({"strain": s_, "step": step, "metabolite_added": r["set"], "class": r["chemical_class"],
                    "pi_strain": r[f"pi_{s_}"], "leverage": r[f"leverage_{s_}"], "key": key})
wet = pd.DataFrame(wet)
pool_members = {s: [p.strip() for p in str(pools.loc[s, "pool"]).split(";")] for s in pools.index}
k_final = {s: int(final.loc[s, K]) for s in STRAINS}


# ---------------------------------------------------------------- Part A: single metabolites
def other_pi(key, s):
    return "; ".join(f"{o} {stab.loc[key, f'pi_{o}']:.2f}" for o in STRAINS if o != s) if key in stab.index else NE


def top_acc(key, s):
    if (key, s) not in lev.index:
        return NE
    r = lev.loc[(key, s)]
    return f"{r.top_accession} (pi {r.pi_with_top:.2f} with / {r.pi_without_top:.2f} without)"


rows_a = []
cv = dict(zip(resel["metabolite key"].astype(str).str.strip(), resel["re-selected in outer folds (of 10)"].astype(int)))
for s in STRAINS:
    w = wet[wet.strain == s].sort_values("step")
    for _, r in w.iterrows():
        tier1 = r.pi_strain >= PI_THR
        in_mix = r.step <= k_final[s]
        in_pool = r.metabolite_added in pool_members.get(s, [])
        if not (tier1 or in_mix or in_pool):
            continue
        key, pi, lv = r.key, r.pi_strain, r.leverage
        c = cv.get(key) if tier1 else None
        why = [f"pi = {pi:.2f} for {s} ({'at or above' if tier1 else 'below'} the threshold 0.40)"]
        if tier1:
            why.append(f"re-selected in {c} of 10 outer training sets of the strict nested cross-validation")
        why.append(f"member {int(r.step)} of the final {s} mixture (Table 3)" if in_mix else "not in the final mixture")
        if in_pool:
            why.append("in the " + ("stability-selected" if tier1 else "exploratory") + " pool of the combination test")
        lim = []
        if lv >= LEV_FLAG:
            lim.append(f"single-accession leverage {lv:.2f} (>= 0.7): the selection may rest on one accession")
        if tier1 and c < CV_HALF:
            lim.append(f"re-selected in fewer than half of the outer folds ({c} of 10)")
        if tier1 and abs(pi - PI_THR) < 1e-9:
            lim.append("selection probability exactly at the threshold")
        if not tier1:
            lim.append("below the stability threshold (exploratory)")
            if pools.loc[s, "pool type"] == "exploratory":
                lim.append(f"no {s} metabolite reached the threshold; the pool of {int(pools.loc[s, 'size'])} is a cut-off "
                           "inherited from an earlier analysis, not a pre-specified rule")
            elif not in_pool:
                lim.append("not tested in the combination test (Step 4)")
        lim.append(MARKER)
        exp = (f"pure-compound inhibition assay against {s} over a dose-response series; include the other three "
               "strains to assess target-strain specificity")
        if lv >= LEV_FLAG:
            exp += "; interpret with caution, as the statistical support depends on one accession"
        if not tier1:
            exp = "after the tier 1 candidates: " + exp
        rows_a.append({
            "target strain": NAME[s], "position in pi order (Table 3 step)": int(r.step),
            "metabolite": pretty(r.metabolite_added), "feature ID": key, "chemical class": CLASS.get(r["class"], r["class"]),
            "selection probability (target strain)": round(pi, 2), "selection probability (other strains)": other_pi(key, s),
            "CV re-selection (of 10 outer folds)": c if tier1 else NE, "leverage (target strain)": round(lv, 2),
            "leverage >= 0.7": "yes" if lv >= LEV_FLAG else "no", "most influential accession": top_acc(key, s),
            "evidence tier": "Tier 1" if tier1 else "Tier 3",
            "in final mixture (Table 3)": f"yes (step {int(r.step)})" if in_mix else "no",
            "in combination test (Step 4)": ("yes, stability-selected pool" if tier1 and in_pool else
                                              "yes, exploratory pool" if in_pool else "no"),
            "computational rationale": "; ".join(why), "suggested first experiment": exp, "key limitation": "; ".join(lim),
            "_o": (0 if tier1 else 1, STRAINS.index(s), int(r.step))})
part_a = pd.DataFrame(rows_a).sort_values("_o").drop(columns="_o")


# ---------------------------------------------------------------- Part B: combinations (all Step 4 combinations)
def member(s, name):
    m = wet[(wet.strain == s) & (wet.metabolite_added == name)].iloc[0]
    return m.pi_strain, m.leverage


rows_b = []
for _, r in combos.iterrows():
    s, order = r.strain, r.order
    names = r.combination.split(" + ")
    info = [member(s, n) for n in names]
    stable = all(p >= PI_THR for p, _ in info)
    if order == "pair":
        comp_sig = ("as frequent as truly complementary features (>= %.2f)" % COMP_TRUE if r.pi_comp >= COMP_TRUE else
                    "recurrent (>= %.2f), below truly complementary features (%.2f)" % (MIN_PI, COMP_TRUE)
                    if r.pi_comp >= MIN_PI else "not recurrent")
        int_sig = ("at planted-interaction level (>= %.2f)" % INT_PLANTED if r.pi_int >= INT_PLANTED else
                   "above non-interacting feature pairs (>= %.2f)" % INT_TRUE if r.pi_int >= INT_TRUE else
                   "recurrent (>= %.2f), not above non-interacting feature pairs (%.2f)" % (MIN_PI, INT_TRUE)
                   if r.pi_int >= MIN_PI else "not recurrent")
        signal_c, signal_i = r.pi_comp >= COMP_TRUE, r.pi_int >= INT_TRUE
    else:   # positive-control references are defined for pairs; triads are read only against the recurrence level
        comp_sig = "recurrent (>= %.2f)" % MIN_PI if r.pi_comp >= MIN_PI else "not recurrent"
        int_sig = "recurrent (>= %.2f)" % MIN_PI if r.pi_int >= MIN_PI else "not recurrent"
        signal_c = signal_i = False
    repro = bool(r.reproducible_complementarity or r.reproducible_interaction)
    recurrent = max(r.pi_comp, r.pi_int) >= MIN_PI
    if stable:
        tier = ("Tier 2" if (signal_c or signal_i) else
                "not prioritized (stable members; recurrent signal below the positive-control reference)" if recurrent
                else "not prioritized (stable members; no recurrent signal)")
    else:
        tier = "Tier 3"
    why = ["all members reproducible (pi >= 0.40)" if stable else
           "exploratory pool: members below the threshold (pi " + ", ".join(f"{p:.2f}" for p, _ in info) + ")"]
    if signal_c:
        why.append(f"complementarity gain in {r.pi_comp:.0%} of half-samples, as often as truly complementary features "
                   f"in the synthetic positive control ({COMP_TRUE:.0%})")
    elif r.pi_comp >= MIN_PI:
        why.append(f"complementarity gain recurrent in {r.pi_comp:.0%} of half-samples")
    if signal_i:
        why.append(f"interaction gain in {r.pi_int:.0%} of half-samples, above non-interacting feature pairs of the "
                   f"positive control ({INT_TRUE:.0%}; planted interaction {INT_PLANTED:.0%})")
    elif r.pi_int >= MIN_PI:
        why.append(f"interaction gain recurrent in {r.pi_int:.0%} of half-samples")
    if len(why) == 1:
        why.append("no recurrent complementarity or interaction gain")
    # rule of candidate_cards.py: factorial design if pi_int >= INT_TRUE; pair versus members if the complementarity
    # is recurrent (and, for pairs with both signals, if it reaches the level of truly complementary features)
    tests = []
    if signal_c or (r.pi_comp >= MIN_PI and not signal_i):
        tests.append("single compounds versus the combination at equal total concentration, and a dose-dependent "
                     "mixture series at a fixed ratio")
    if signal_i:
        tests.append("factorial (checkerboard) assay against an explicitly defined non-interaction reference model "
                     "(e.g. Bliss independence)")
    val = "; ".join(tests) if tests else "not prioritized; if at all, only within the cumulative mixture of Table 3"
    if tests and tier != "Tier 2":
        val = "lower priority, after tier 1 and 2 candidates: " + val
    lim = ["did not meet the reproducibility criterion (pi >= 0.7 and at least 0.3 above permutation); no combination did"]
    if signal_c or signal_i or r.pi_comp >= MIN_PI or r.pi_int >= MIN_PI:
        lim.append("a statistical gain does not show joint biochemical action: a single active member co-exuded with "
                   "an inactive partner gives the same pattern")
    if not stable:
        lim.append("members below the stability threshold")
    if any(lv >= LEV_FLAG for _, lv in info):
        lim.append("a member has single-accession leverage >= 0.7")
    if repro:
        lim = ["meets the reproducibility criterion"]
    rows_b.append({
        "target strain": NAME[s], "metabolite combination": pretty(r.combination), "type": order,
        "pool type": "stability-selected" if pools.loc[s, "pool type"] == "stability-selected" else f"exploratory ({s})",
        "members' selection probability (target strain)": " / ".join(f"{p:.2f}" for p, _ in info),
        "members' leverage": " / ".join(f"{lv:.2f}" for _, lv in info),
        "complementarity recurrence (pi_comp)": round(r.pi_comp, 2),
        "interaction recurrence (pi_int)": round(r.pi_int, 2),
        "permutation comparison (pi_comp / pi_int under label permutation)": f"{r.pi_comp_null:.3f} / {r.pi_int_null:.3f}",
        "complementarity signal": comp_sig, "interaction signal": int_sig,
        "reproducibility criterion met": "yes" if repro else "no",
        "evidence tier": tier, "computational rationale": "; ".join(why), "suggested validation": val,
        "key limitation": "; ".join(lim),
        # order: tier 2; stable members with a recurrent signal; tier 3 with a recurrent signal; the rest
        "_o": (0 if tier == "Tier 2" else (1 if stable else 2) if recurrent else (3 if stable else 4),
               STRAINS.index(s), 0 if order == "pair" else 1, -max(r.pi_comp, r.pi_int))})
part_b = pd.DataFrame(rows_b).sort_values("_o").drop(columns="_o")

readme = pd.DataFrame([
    ("purpose", "computational candidate list of the wheat application for planning BNI assays: which metabolite to "
                "test first for each strain, the evidence behind it, which pairs warrant follow-up and why, and which "
                "candidates are exploratory or depend on single accessions"),
    ("status", "no candidate has been tested for inhibition in this study; every suggested experiment is future work. "
               "A candidate may be a BNI-associated marker rather than an active inhibitor; the pure-compound assays "
               "distinguish the two"),
    ("evidence tiers", "descriptive categories of computational evidence, not a ranking of inhibitory potency and not "
                       "the probability that an assay succeeds. Tier 1: single metabolite with pi >= 0.4 for the strain. "
                       "Tier 2: pair of tier 1 metabolites with a complementarity or interaction signal. Tier 3: "
                       "metabolite below the threshold or combination containing one (exploratory). Combinations of "
                       "tier 1 metabolites without a signal are listed as not prioritized, separately for those whose "
                       "gain recurred (>= 0.15) but stayed below the positive-control reference and those without a "
                       "recurrent gain"),
    ("which first", "within each strain, single metabolites are listed in the selection-probability order of main-text "
                    "Table 3 (position); the first tier 1 metabolite of a strain has the strongest computational support "
                    "for that strain. NU has no tier 1 candidate"),
    ("selection probability", "fraction of 50 complementary half-samples in which the metabolite was in the strain's "
                              "BRS top ten; it measures reproducible association in this cohort, not the probability of "
                              "inhibitory activity"),
    ("CV re-selection", "number of the ten outer training sets of the strict nested cross-validation in which the "
                        "metabolite was re-selected (Table 2 of the manuscript); computed for the reproducible metabolites only "
                        "(Not evaluated otherwise); fewer than 5 of 10 is flagged"),
    ("leverage", "largest difference in selection probability between the half-samples with and without one accession; "
                 ">= 0.7 indicates that the selection may rest on a single accession; the most influential accession "
                 "and the selection probabilities with and without it are given"),
    ("complementarity / interaction recurrence", "fraction of the 50 half-samples in which the out-of-fold R2 gain of "
                                                 "the additive model over the best member (complementarity) or of the "
                                                 "product terms over the additive model (interaction) exceeded 0.02 "
                                                 "(Supplementary Table S15)"),
    ("permutation comparison", "the same fraction under label permutation (20 permutations per half-sample)"),
    ("signals", f"pairs, read against the synthetic positive control of the manuscript (Supplementary Table S15): "
                f"complementarity signal if pi_comp >= "
                f"{COMP_TRUE:.2f} (truly complementary features); interaction signal if pi_int >= {INT_TRUE:.2f} "
                f"(non-interacting true feature pairs; planted interaction {INT_PLANTED:.2f}); recurrent if >= "
                f"{MIN_PI} (about four times the largest permutation fraction). Triads are read against the recurrence "
                "level only"),
    ("reproducibility criterion", "pi >= 0.7 and at least 0.3 above the permutation fraction; no combination met it, "
                                  "so tier 2 combinations are hypotheses worth testing, not confirmed interactions and "
                                  "not evidence of biochemical synergy"),
    ("scope", "the global nested cross-validation AUC (0.75) refers to the consensus list as a whole, not to any single "
              "candidate; the stability of single metabolites does not make their combinations reproducible"),
], columns=["field", "description"])
with pd.ExcelWriter(OUT) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    part_a.to_excel(w, sheet_name="partA_single_metabolites", index=False)
    part_b.to_excel(w, sheet_name="partB_combinations", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(part_a[["evidence tier", "target strain", "position in pi order (Table 3 step)", "metabolite",
              "selection probability (target strain)", "CV re-selection (of 10 outer folds)", "leverage (target strain)",
              "in final mixture (Table 3)", "in combination test (Step 4)"]].to_string(index=False))
print(part_b.groupby(["evidence tier", "pool type", "type"]).size().to_string())
print(part_b[part_b["evidence tier"] == "Tier 2"][["metabolite combination", "complementarity recurrence (pi_comp)",
                                                   "interaction recurrence (pi_int)", "suggested validation"]].to_string(index=False))
print(f"Saved {OUT}: part A {len(part_a)} rows, part B {len(part_b)} rows")
