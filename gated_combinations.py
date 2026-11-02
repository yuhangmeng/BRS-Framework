# -*- coding: utf-8 -*-
"""
Stability-gated combination test (Step 4 of the pipeline).

Combinations are enumerated only within a small, fixed pool of reproducible single metabolites (the "gate"), so
the pool no longer changes between half-samples. For every pair and triad in the pool we decompose the BRS
Validity term into two out-of-fold (OOF) R^2 gains of linear models on the same 50 complementary half-samples:

    complementarity gain  pair : R2[x1, x2]                     - max(R2[x1], R2[x2])
                          triad: R2[x1, x2, x3]                 - max over its pairs R2[xi, xj]
    interaction gain      pair : R2[x1, x2, x1*x2]              - R2[x1, x2]
                          triad: R2[x1, x2, x3, g(c c), g(c c c)] - max(R2[x1, x2, x3],
                                 max over its pairs R2[x1, x2, x3, g(ci cj)])   (BRS triad design)
                          i.e. beyond the best model with a single pairwise product, so that a triad does not
                          inherit the interaction of an embedded pair

A gain is "present" in a half-sample when it exceeds DELTA (share of variance). For each combination we report the
fraction of half-samples in which it is present (pi_comp, pi_int) and the same fraction under label permutation
(null; N_PERM permutations per half-sample).

Modes
    --data wheat      pools = per-strain consensus metabolites (pi >= 0.4) and recommended mixture members;
                      log-scale input; the 50 half-samples of results_stability_uni
    --data synthetic  positive control on the ten benchmark datasets: the gate is the single-feature selection
                      probability from univariate BRS over 50 half-samples (pi >= 0.4); the planted pair is a true
                      product interaction, the linear features only add

Outputs: supplementary/ST_gated_combinations.xlsx (wheat), results/gated_synthetic.xlsx (synthetic)
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent
STRAINS = ["NF", "NV", "NM", "NU"]
FOLDS, SEED = 5, 42
DELTA = 0.02
N_PERM = 20
PI_GATE = 0.4
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
    full_idx = np.arange(len(y))
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


def halves_from_csv(path):
    s = pd.read_csv(path)
    return [np.array(list(map(int, v.split()))) for v in s["samples"]]


def make_halves(n, n_pairs=25, seed=2026):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_pairs):
        p = rng.permutation(n)
        out += [np.sort(p[:n // 2]), np.sort(p[n // 2:2 * (n // 2)])]
    return out


# ==========================================
# 2) Wheat
# ==========================================
def run_wheat():
    inp = pd.read_excel(ROOT / "Data Preprocessing" / "filtered_metabolites_log.xlsx", sheet_name=0, decimal=",")
    tcol = {s: [c for c in inp.columns if "Inhibition" in str(c) and str(c).endswith(s)][0] for s in STRAINS}
    mets = [c for c in inp.columns if c not in tcol.values()]
    keys = [str(c).strip() for c in mets]
    X = inp[mets].apply(pd.to_numeric, errors="coerce").values
    halves = halves_from_csv(ROOT / "results_stability_uni" / "subsamples.csv")
    pi = pd.read_excel(ROOT / "results_stability_uni" / "stability_summary_uni.xlsx", sheet_name="pi_single")
    pi["key"] = pi["key"].astype(str).str.strip()
    wet = pd.read_excel(ROOT / "supplementary" / "ST_incremental_signature.xlsx", sheet_name="wetlab_lists")
    rec = pd.read_excel(ROOT / "supplementary" / "ST_incremental_signature.xlsx", sheet_name="recommended_size")
    rec = rec.set_index("target")["recommended_k"]
    name = dict(zip(pi["key"], pi["set"]))
    rng = np.random.default_rng(SEED)
    out, pools = [], []
    for s in STRAINS:
        y = pd.to_numeric(inp[tcol[s]], errors="coerce").values
        gate = set(pi.loc[pi[f"pi_{s}"] >= PI_GATE, "key"])
        mix = set(wet.loc[(wet.strain == s) & (wet.step <= rec[s]), "key"].astype(str).str.strip())
        pool_keys = sorted(gate | mix, key=lambda k: -float(pi.set_index("key").loc[k, f"pi_{s}"]))
        pools.append({"strain": s, "pool": "; ".join(name.get(k, k) for k in pool_keys), "size": len(pool_keys)})
        idx = [keys.index(k) for k in pool_keys]
        print(f"{s}: pool of {len(idx)} -> {len(idx) * (len(idx) - 1) // 2} pairs, "
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
    out_x = ROOT / "supplementary" / "ST_gated_combinations.xlsx"
    with pd.ExcelWriter(out_x) as w:
        pd.DataFrame([("design", "combinations enumerated only within each strain's gated pool (consensus pi >= 0.4 "
                                 "and recommended mixture members); log-scale input; 50 half-samples"),
                      ("complementarity gain", "OOF R2 of the additive model minus that of the best embedded lower order"),
                      ("interaction gain", "pairs: OOF R2 gain from adding the product term to the additive model; "
                                           "triads: OOF R2 of the full product design (BRS) minus that of the best "
                                           "additive model with at most one pairwise product"),
                      ("pi", f"fraction of half-samples with gain > {DELTA}; null: {N_PERM} label permutations per half-sample"),
                      ("reproducible", "pi >= 0.7 and at least 0.3 above the null fraction")],
                     columns=["field", "description"]).to_excel(w, sheet_name="README", index=False)
        pd.DataFrame(pools).to_excel(w, sheet_name="pools", index=False)
        R[cols].round(3).to_excel(w, sheet_name="combinations", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(R[cols].round(2).to_string(index=False))
    print(f"Saved {out_x}")


# ==========================================
# 3) Synthetic positive control
# ==========================================
def synthetic_seed(seed):
    import torch
    torch.set_num_threads(1)
    import benchmark_comprehensive as bc
    X, y, fn, true_u, true_p, true_t = bc.generate_synthetic(seed=seed)
    halves = make_halves(len(y))
    cnt = np.zeros(X.shape[1])
    for idx in halves:
        torch.manual_seed(42)
        np.random.seed(42)
        ranked, _ = bc.brs_univariate(X[idx], y[idx], fn)
        for f in ranked[:10]:
            cnt[fn.index(f)] += 1
    pi = cnt / len(halves)
    pool = list(np.where(pi >= PI_GATE)[0])
    rows = score_combos(X, y, pool, halves, np.random.default_rng(SEED))
    tp = tuple(sorted(fn.index(f) for f in true_p))
    tt = tuple(sorted(fn.index(f) for f in true_t))
    truth = set(fn.index(f) for f in true_u)
    for r in rows:
        c = tuple(sorted(r["combo"]))
        r["kind"] = ("planted pair" if c == tp else "planted triad" if c == tt else
                     "true features, no interaction" if set(c) <= truth else "contains noise")
        r["combination"] = " + ".join(fn[j] for j in r.pop("combo"))
        r["seed"] = seed
    summary = {"seed": seed, "gate_size": len(pool), "gate": "; ".join(fn[j] for j in pool),
               "true_in_gate": len(truth & set(pool)), "pair_in_gate": set(tp) <= set(pool),
               "triad_in_gate": set(tt) <= set(pool)}
    return rows, summary


def run_synthetic(workers):
    seeds = [42] + list(range(1, 10))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(synthetic_seed, seeds))
    R = pd.DataFrame([r for rows, _ in res for r in rows])
    S = pd.DataFrame([s for _, s in res])
    K = R.groupby(["kind", "order"]).agg(n=("pi_int", "size"), pi_comp_mean=("pi_comp", "mean"),
                                         pi_int_mean=("pi_int", "mean"), pi_int_null_mean=("pi_int_null", "mean"),
                                         int_gain_full_median=("int_gain_full", "median")).reset_index()
    out_x = ROOT / "results" / "gated_synthetic.xlsx"
    with pd.ExcelWriter(out_x) as w:
        S.to_excel(w, sheet_name="gates", index=False)
        K.round(3).to_excel(w, sheet_name="by_kind", index=False)
        R.round(3).to_excel(w, sheet_name="combinations", index=False)
    pd.set_option("display.width", 250)
    print(S.to_string(index=False))
    print(K.round(2).to_string(index=False))
    print(R[R.kind.str.startswith("planted")][["seed", "kind", "pi_comp", "pi_int", "pi_int_null",
                                               "int_gain_full"]].round(2).to_string(index=False))
    print(f"Saved {out_x}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["wheat", "synthetic"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run_wheat() if a.data == "wheat" else run_synthetic(a.workers)
