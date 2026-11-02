# -*- coding: utf-8 -*-
"""
Pipeline-level benchmark on the synthetic data (Supplementary Table S21, sheet 'simulation'): what does restricting
the combination test to the stability-selected pool (Steps 2 and 4) change compared with testing directly
enumerated combinations?

Both arms use the combination test of Step 4 with the same criterion (pi >= 0.7 and >= 0.3 above the permutation
value; gains of out-of-fold R^2 > 0.02), the same 50 half-samples and the same 20 label permutations per half-sample:
    pipeline   all pairs and triads of the stability-selected pool (pi >= 0.4; pool from results_sample_size/)
    direct     pairs of the top 20 and triads of the top 10 metabolites of a single full-data univariate BRS run,
               as for the directly enumerated combinations of the paper (Section 2.1.1)
Datasets: the synthetic generator of the benchmark, n in {30, 50, 75, 100, 150, 200}, seeds 42 and 1-9.

The gains are computed with a vectorized least-squares version of gated_combinations.gains (identical values;
checked at start-up) so that the ~300 directly enumerated combinations per dataset can be tested.

Output: results_sample_size/pipeline_vs_direct.xlsx
Run:    python benchmark_pipeline_sim.py --workers 8
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent
SIM_DIR = ROOT / "results_sample_size"
OUT_XLSX = SIM_DIR / "pipeline_vs_direct.xlsx"
N_LIST = [30, 50, 75, 100, 150, 200]
SEEDS = [42] + list(range(1, 10))
PI_REPRO, MARGIN, DELTA, N_PERM, FOLDS, SEED = 0.7, 0.3, 0.02, 20, 5, 42
TOP_PAIR, TOP_TRIAD = 20, 10
_SPLITS = {}


def splits(m):
    if m not in _SPLITS:
        _SPLITS[m] = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(np.zeros(m)))
    return _SPLITS[m]


def oof_r2(D, Y):
    """Out-of-fold R^2 of OLS with intercept for every column of Y (as gated_combinations.oof_r2)."""
    oof = np.empty_like(Y)
    A = np.column_stack([np.ones(len(D)), D])
    for tr, va in splits(len(D)):
        coef = np.linalg.lstsq(A[tr], Y[tr], rcond=None)[0]
        oof[va] = A[va] @ coef
    ss_res = ((Y - oof) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return np.maximum(0.0, 1.0 - ss_res / ss_tot)


def g(z):
    return np.tanh(z / 3.0)


def gains(cols, Y):
    """(complementarity gain, interaction gain) for every column of Y, as gated_combinations.gains."""
    if len(cols) == 2:
        single = np.max([oof_r2(c[:, None], Y) for c in cols], axis=0)
        add = oof_r2(np.column_stack(cols), Y)
        inter = oof_r2(np.column_stack(cols + [cols[0] * cols[1]]), Y)
        return add - single, inter - add
    c = [v - v.mean() for v in cols]
    pr = list(combinations(range(3), 2))
    add = oof_r2(np.column_stack(cols), Y)
    sub_add = np.max([oof_r2(np.column_stack([cols[i], cols[j]]), Y) for i, j in pr], axis=0)
    sub_int = np.max([oof_r2(np.column_stack(cols + [g(c[i] * c[j])]), Y) for i, j in pr], axis=0)
    full = np.column_stack(cols + [g(c[0] * c[1]), g(c[0] * c[2]), g(c[1] * c[2]), g(c[0] * c[1] * c[2])])
    return add - sub_add, oof_r2(full, Y) - np.maximum(add, sub_int)


def test_combos(X, y, combos, halves, perms):
    rows = []
    for combo in combos:
        comp, inter, comp0, inter0 = [], [], [], []
        for idx, P in zip(halves, perms):
            Y = np.column_stack([y[idx]] + [y[idx][p] for p in P])
            a, b = gains([X[idx, j] for j in combo], Y)
            comp.append(a[0]); inter.append(b[0]); comp0 += list(a[1:]); inter0 += list(b[1:])
        pc, pi_ = np.mean(np.array(comp) > DELTA), np.mean(np.array(inter) > DELTA)
        pc0, pi0 = np.mean(np.array(comp0) > DELTA), np.mean(np.array(inter0) > DELTA)
        rows.append({"combo": tuple(sorted(combo)), "pi_comp": pc, "pi_comp_null": pc0, "pi_int": pi_, "pi_int_null": pi0,
                     "repro_int": bool(pi_ >= PI_REPRO and pi_ >= pi0 + MARGIN),
                     "repro_comp": bool(pc >= PI_REPRO and pc >= pc0 + MARGIN)})
    return rows


def run_one(n, seed):
    import torch
    torch.set_num_threads(1)
    import benchmark_comprehensive as bc
    import gated_combinations as gc

    X, y, fn, true_u, true_p, true_t = bc.generate_synthetic(n_samples=n, seed=seed)
    halves = gc.make_halves(n)
    rng = np.random.default_rng(seed)
    perms = [[rng.permutation(len(idx)) for _ in range(N_PERM)] for idx in halves]
    truth = {fn.index(f) for f in true_u}
    tp = tuple(sorted(fn.index(f) for f in true_p))
    tt = tuple(sorted(fn.index(f) for f in true_t))

    gate = json.loads((SIM_DIR / f"n{n:03d}_s{seed:02d}.json").read_text(encoding="utf-8"))["gate"]
    pool = [fn.index(f) for f in gate]
    torch.manual_seed(42)
    np.random.seed(42)
    ranked, _ = bc.brs_univariate(X, y, fn)
    top = [fn.index(f) for f in ranked]
    arms = {"pipeline": [c for k in (2, 3) for c in combinations(pool, k)],
            "direct": list(combinations(top[:TOP_PAIR], 2)) + list(combinations(top[:TOP_TRIAD], 3))}
    out = {"n": n, "seed": seed}
    for arm, combos in arms.items():
        res = test_combos(X, y, combos, halves, perms)
        def kind(c):
            return ("planted pair" if c == tp else "planted triad" if c == tt else
                    "contains planted pair" if set(tp) <= set(c) else
                    "true features only" if set(c) <= truth else "contains noise")
        k = [kind(r["combo"]) for r in res]
        out.update({
            f"{arm}: combinations tested": len(res),
            f"{arm}: planted pair tested": "planted pair" in k,
            f"{arm}: planted pair called": any(r["repro_int"] for r, kk in zip(res, k) if kk == "planted pair"),
            f"{arm}: planted triad called": any(r["repro_int"] for r, kk in zip(res, k) if kk == "planted triad"),
            f"{arm}: triads containing the planted pair called": sum(r["repro_int"] for r, kk in zip(res, k) if kk == "contains planted pair"),
            f"{arm}: interaction calls with a noise feature": sum(r["repro_int"] for r, kk in zip(res, k) if kk == "contains noise"),
            f"{arm}: interaction calls among true features without interaction": sum(r["repro_int"] for r, kk in zip(res, k) if kk == "true features only"),
            f"{arm}: complementarity calls with a noise feature": sum(r["repro_comp"] for r, kk in zip(res, k) if kk == "contains noise"),
        })
    return out


def check_against_reference():
    """The vectorized gains must reproduce gated_combinations.gains on observed labels."""
    import benchmark_comprehensive as bc
    import gated_combinations as gc
    X, y, fn, *_ = bc.generate_synthetic(n_samples=50, seed=42)
    idx = gc.make_halves(50)[0]
    for combo in [(0, 1), (2, 5), (0, 1, 2), (3, 7, 11)]:
        ref = gc.gains([X[idx, j] for j in combo], y[idx])
        new = gains([X[idx, j] for j in combo], y[idx][:, None])
        assert np.allclose(ref, [new[0][0], new[1][0]], atol=1e-8), (combo, ref, new)
    print("vectorized gains match gated_combinations.gains")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    check_against_reference()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, n, s): (n, s) for n in N_LIST for s in SEEDS}
        for i, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            print(f"{i}/{len(futs)} done (n={futs[fu][0]}, seed={futs[fu][1]})", flush=True)
    D = pd.DataFrame(rows).sort_values(["n", "seed"])
    num = [c for c in D.columns if c not in ("n", "seed")]
    S = D.groupby("n")[num].agg(lambda v: v.mean() if v.dtype == bool else v.sum()).reset_index()
    for arm in ("pipeline", "direct"):
        S[f"{arm}: combinations tested"] = D.groupby("n")[f"{arm}: combinations tested"].mean().values
    readme = pd.DataFrame([
        ("by_n", "fractions of the 10 datasets for 'tested'/'called' columns; totals over the 10 datasets for counts "
                 "of calls; mean number of combinations tested per dataset."),
        ("criterion", "interaction (or complementarity) gain > 0.02 in >= 70% of 50 half-samples and >= 30 points "
                      "above the same fraction under label permutation (20 permutations per half-sample)."),
    ], columns=["sheet", "description"])
    with pd.ExcelWriter(OUT_XLSX) as w:
        readme.to_excel(w, sheet_name="README", index=False)
        S.round(3).to_excel(w, sheet_name="by_n", index=False)
        D.to_excel(w, sheet_name="per_dataset", index=False)
    pd.set_option("display.width", 250)
    print(S.round(2).T.to_string())
    print("Saved", OUT_XLSX)
