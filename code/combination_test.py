# -*- coding: utf-8 -*-
"""
Step 4: combination test among reproducible metabolites.

Pairs and triads are enumerated only within a small, fixed pool per strain: the metabolites with pi >= 0.4 for that
strain (stability-selected pool). A strain without reproducible metabolites gets an exploratory pool of its
N_EXPLORATORY metabolites with the highest selection probability (in the manuscript: NU, eight metabolites, a cut-off
inherited from an earlier analysis, not a pre-specified rule). On the same 50 half-samples as stability selection,
the Validity term is decomposed into two out-of-fold (OOF) R^2 gains of linear models:

    complementarity gain  pair : R2[x1, x2]                       - max(R2[x1], R2[x2])
                          triad: R2[x1, x2, x3]                   - max over its pairs R2[xi, xj]
    interaction gain      pair : R2[x1, x2, x1*x2]                - R2[x1, x2]
                          triad: R2[x1, x2, x3, g(c c), g(c c c)] - max(R2[x1, x2, x3],
                                 max over its pairs R2[x1, x2, x3, g(ci cj)])   (g = tanh(z / 3))
A gain is present in a half-sample when it exceeds 0.02. Reported per combination: the fraction of half-samples in
which it is present (pi_comp, pi_int) and the same fraction under label permutation (20 per half-sample). A
combination is reproducible if this fraction is >= 0.7 and at least 0.3 above its permutation value.

    python code/combination_test.py          (needs the outputs of stability_selection.py)
Output: results/combination_test.xlsx (pools, combinations)
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "preprocessing" / "filtered_metabolites_log.xlsx"
SUBSAMPLES = ROOT / "results" / "stability_runs" / "subsamples.csv"
STAB_XLSX = ROOT / "results" / "stability_selection.xlsx"
OUT_XLSX = ROOT / "results" / "combination_test.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
FOLDS, SEED = 5, 42
DELTA = 0.02
N_PERM = 20
PI_GATE = 0.4
N_EXPLORATORY = 8
TANH_SCALE = 3.0


# ==========================================
# 1) Gains
# ==========================================
def oof_r2(X, y):
    oof = np.zeros_like(y, dtype=float)
    for tr, va in KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X):
        oof[va] = LinearRegression().fit(X[tr], y[tr]).predict(X[va])
    return max(0.0, float(r2_score(y, oof)))


def std(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-8)


def g(z):
    return np.tanh(z / TANH_SCALE)


def gains(cols, y):
    """cols: list of 2 or 3 column vectors. Returns (complementarity gain, interaction gain)."""
    k = len(cols)
    single = [oof_r2(std(c.reshape(-1, 1)), y) for c in cols]
    add = oof_r2(std(np.column_stack(cols)), y)
    if k == 2:
        inter = oof_r2(std(np.column_stack(cols + [cols[0] * cols[1]])), y)
        return add - max(single), inter - add
    c = [v - v.mean() for v in cols]
    pairs = list(combinations(range(3), 2))
    sub_add = [oof_r2(std(np.column_stack([cols[i], cols[j]])), y) for i, j in pairs]
    sub_int = [oof_r2(std(np.column_stack(cols + [g(c[i] * c[j])])), y) for i, j in pairs]
    full = np.column_stack(cols + [g(c[0] * c[1]), g(c[0] * c[2]), g(c[1] * c[2]), g(c[0] * c[1] * c[2])])
    return add - max(sub_add), oof_r2(std(full), y) - max([add] + sub_int)


def score_combos(X, y, pool, halves, rng):
    """Gains for all pairs and triads of pool on every half-sample, observed and under label permutation."""
    rows = []
    combos = [c for k in (2, 3) for c in combinations(pool, k)]
    for combo in combos:
        cols_full = [X[:, j] for j in combo]
        comp_full, int_full = gains(cols_full, y)
        comp, inter, comp0, inter0 = [], [], [], []
        for idx in halves:
            cols = [X[idx, j] for j in combo]
            a, b = gains(cols, y[idx])
            comp.append(a)
            inter.append(b)
            for _ in range(N_PERM):
                a0, b0 = gains(cols, rng.permutation(y[idx]))
                comp0.append(a0)
                inter0.append(b0)
        comp, inter, comp0, inter0 = map(np.array, (comp, inter, comp0, inter0))
        rows.append({"combo": combo, "order": "pair" if len(combo) == 2 else "triad",
                     "comp_gain_full": comp_full, "int_gain_full": int_full,
                     "comp_gain_median": np.median(comp), "int_gain_median": np.median(inter),
                     "pi_comp": np.mean(comp > DELTA), "pi_int": np.mean(inter > DELTA),
                     "pi_comp_null": np.mean(comp0 > DELTA), "pi_int_null": np.mean(inter0 > DELTA)})
    return rows


# ==========================================
# 2) Pools and test
# ==========================================
def main():
    inp = pd.read_excel(INPUT, sheet_name=0, decimal=",")
    tcol = {s: [c for c in inp.columns if "Inhibition" in str(c) and str(c).endswith(s)][0] for s in STRAINS}
    mets = [c for c in inp.columns if c not in tcol.values()]
    keys = [str(c).strip() for c in mets]
    X = inp[mets].apply(pd.to_numeric, errors="coerce").values
    halves = [np.array(list(map(int, v.split()))) for v in pd.read_csv(SUBSAMPLES)["samples"]]
    pi = pd.read_excel(STAB_XLSX, sheet_name="pi_single")
    pi["key"] = pi["key"].astype(str).str.strip()
    pi = pi.set_index("key")
    name = pi["set"].to_dict()
    rng = np.random.default_rng(SEED)
    out, pools = [], []
    for s in STRAINS:
        y = pd.to_numeric(inp[tcol[s]], errors="coerce").values
        # selection-probability order, ties by the full-data BRS (as for the mixtures)
        order = pi.sort_values([f"pi_{s}", f"BRS_full_{s}"], ascending=False, kind="mergesort")
        order = order[order[f"pi_{s}"] > 0]
        stable = list(order.index[order[f"pi_{s}"] >= PI_GATE])
        pool_keys, kind = (stable, "stability-selected") if stable else (list(order.index[:N_EXPLORATORY]), "exploratory")
        pools.append({"strain": s, "pool type": kind, "pool": "; ".join(name.get(k, k) for k in pool_keys),
                      "size": len(pool_keys)})
        idx = [keys.index(k) for k in pool_keys]
        print(f"{s}: {kind} pool of {len(idx)} -> {len(idx) * (len(idx) - 1) // 2} pairs, "
              f"{len(list(combinations(idx, 3)))} triads", flush=True)
        for r in score_combos(X, y, idx, halves, rng):
            r["strain"] = s
            r["combination"] = " + ".join(name.get(keys[j], keys[j]) for j in r.pop("combo"))
            out.append(r)
    R = pd.DataFrame(out)
    R["reproducible_complementarity"] = (R.pi_comp >= 0.7) & (R.pi_comp > R.pi_comp_null + 0.3)
    R["reproducible_interaction"] = (R.pi_int >= 0.7) & (R.pi_int > R.pi_int_null + 0.3)
    cols = ["strain", "order", "combination", "comp_gain_full", "comp_gain_median", "pi_comp", "pi_comp_null",
            "int_gain_full", "int_gain_median", "pi_int", "pi_int_null", "reproducible_complementarity",
            "reproducible_interaction"]
    with pd.ExcelWriter(OUT_XLSX) as w:
        pd.DataFrame([("design", "pairs and triads within each strain's pool (pi >= 0.4; exploratory pool of the "
                                 f"{N_EXPLORATORY} highest selection probabilities if no metabolite reaches 0.4); "
                                 "log-scale input; the 50 half-samples of stability selection"),
                      ("complementarity gain", "OOF R2 of the additive model minus that of the best embedded lower order"),
                      ("interaction gain", "pairs: OOF R2 gain from adding the product term to the additive model; "
                                           "triads: OOF R2 of the full product design minus that of the best additive "
                                           "model with at most one pairwise product"),
                      ("pi", f"fraction of half-samples with gain > {DELTA}; null: {N_PERM} label permutations per "
                             "half-sample"),
                      ("reproducible", "pi >= 0.7 and at least 0.3 above the null fraction")],
                     columns=["field", "description"]).to_excel(w, sheet_name="README", index=False)
        pd.DataFrame(pools).to_excel(w, sheet_name="pools", index=False)
        R[cols].round(3).to_excel(w, sheet_name="combinations", index=False)
    print(pd.DataFrame(pools).to_string(index=False))
    print(f"reproducible combinations: {int((R.reproducible_complementarity | R.reproducible_interaction).sum())}; "
          f"saved {OUT_XLSX}")


if __name__ == "__main__":
    main()
