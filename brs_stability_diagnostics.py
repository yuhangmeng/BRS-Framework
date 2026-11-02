# -*- coding: utf-8 -*-
"""
Why are combination rankings unstable under half-sampling? Diagnostics on results_stability/.

(a) Pool effect: for each set, how often it was enumerated at all (its members in that half-sample's
    pool) and how often it reached the top-K when it was enumerated (conditional selection probability).
(b) Bagged consensus: mean BRS of each set over the half-samples in which it was enumerated, computed
    separately on two disjoint groups of 25 half-samples (even / odd complementary pairs). Agreement of
    the two consensus rankings measures the stability of an aggregated (bagged) BRS ranking.
(c) Full-data top-10 versus the consensus ranking.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STAB_DIR = Path("results_stability")
FULL_RUN = Path(os.environ.get("BRS_OUT_DIR", "results_v2_deterministic"))   # full-data BRS run
STRAINS = ["NF", "NV", "NM", "NU"]
ORDERS = {"single": ("univariate_BRS.xlsx", "Uni", "Feature"),
          "pair": ("pairwise_BRS.xlsx", "Pairs", "Pair"),
          "triad": ("triple_BRS.xlsx", "Triplets", "Triplet")}
TOP_K = 10
MIN_ENUM = 5           # consensus only for sets enumerated in >= MIN_ENUM half-samples of a group


def canon(s):
    return " | ".join(sorted(t.strip() for t in str(s).split("|")))


subs = sorted(d for d in STAB_DIR.glob("sub_*") if (d / "triple_BRS.xlsx").stat().st_size > 0)
pair_of = {d.name: int(d.name.split("_")[1]) // 2 for d in subs}

rows_a, rows_b, rows_c = [], [], []
for order, (fname, kind, col) in ORDERS.items():
    # long table: sub, strain, key, BRS, in_top
    recs = []
    for d in subs:
        xls = pd.ExcelFile(d / fname)
        for s in STRAINS:
            t = pd.read_excel(xls, f"Inhibition (pct)-{s}_{kind}_All")
            t["key"] = t[col].map(canon)
            t = t.sort_values(["BRS", "key"], ascending=[False, True]).reset_index(drop=True)
            t["in_top"] = t.index < TOP_K
            recs.append(pd.DataFrame({"sub": d.name, "group": pair_of[d.name] % 2, "strain": s,
                                      "key": t["key"], "BRS": t["BRS"], "in_top": t["in_top"]}))
    L = pd.concat(recs, ignore_index=True)
    B = len(subs)

    # full-data top-10 per strain
    xls = pd.ExcelFile(FULL_RUN / fname)
    full_top = {}
    for s in STRAINS:
        t = pd.read_excel(xls, f"Inhibition (pct)-{s}_{kind}_All")
        t["key"] = t[col].map(canon)
        full_top[s] = list(t.sort_values("BRS", ascending=False)["key"].head(TOP_K))

    for s in STRAINS:
        Ls = L[L.strain == s]
        g = Ls.groupby("key").agg(n_enum=("sub", "nunique"), n_top=("in_top", "sum"))
        g["pi"] = g["n_top"] / B
        g["pi_given_enum"] = g["n_top"] / g["n_enum"]
        ft = g.reindex(full_top[s])
        rows_a.append({"order": order, "strain": s, "sets_enumerated": len(g),
                       "fulltop10_mean_enum_rate": ft["n_enum"].fillna(0).mean() / B,
                       "fulltop10_mean_pi": ft["pi"].fillna(0).mean(),
                       "fulltop10_mean_pi_given_enum": ft["pi_given_enum"].mean(),
                       "max_pi": g["pi"].max(), "max_pi_given_enum(n_enum>=10)": g.loc[g.n_enum >= 10, "pi_given_enum"].max()})
        # bagged consensus in two disjoint groups of half-samples
        cons = {}
        for grp in (0, 1):
            Lg = Ls[Ls.group == grp]
            c = Lg.groupby("key").agg(mean_brs=("BRS", "mean"), n=("sub", "nunique"))
            cons[grp] = c[c.n >= MIN_ENUM]["mean_brs"]
        common = cons[0].index.intersection(cons[1].index)
        rho = spearmanr(cons[0][common], cons[1][common]).correlation if len(common) > 3 else np.nan
        top0 = set(cons[0].sort_values(ascending=False).index[:TOP_K])
        top1 = set(cons[1].sort_values(ascending=False).index[:TOP_K])
        call = Ls.groupby("key").agg(mean_brs=("BRS", "mean"), n=("sub", "nunique"))
        call = call[call.n >= 2 * MIN_ENUM]["mean_brs"].sort_values(ascending=False)
        rows_b.append({"order": order, "strain": s, "sets_in_both_groups": len(common),
                       "spearman_consensus_groupA_vs_B": rho,
                       "top10_overlap_consensus_A_vs_B": len(top0 & top1),
                       "top10_overlap_consensus_vs_fulldata": len(set(call.index[:TOP_K]) & set(full_top[s]))})
        for k in call.index[:TOP_K]:
            rows_c.append({"order": order, "strain": s, "key": k, "consensus_mean_BRS": call[k],
                           "n_enum": int(g.loc[k, "n_enum"]), "pi": g.loc[k, "pi"],
                           "pi_given_enum": g.loc[k, "pi_given_enum"], "in_full_top10": k in full_top[s]})

A, Bc, C = pd.DataFrame(rows_a), pd.DataFrame(rows_b), pd.DataFrame(rows_c)
with pd.ExcelWriter(STAB_DIR / "stability_diagnostics.xlsx") as w:
    A.to_excel(w, sheet_name="pool_effect", index=False)
    Bc.to_excel(w, sheet_name="bagged_consensus", index=False)
    C.to_excel(w, sheet_name="consensus_top10", index=False)
pd.set_option("display.width", 220)
print(A.round(2).to_string(index=False))
print()
print(Bc.round(2).to_string(index=False))
