# -*- coding: utf-8 -*-
"""
Scorer comparison on the synthetic data, where the truth is known (Supplementary Table S20, sheet 'simulation_scorers').

For the 60 datasets of the sample-size simulation (n = 30-200, ten seeds each) and the same 50 half-samples:
    selection     stability selection (top 10 per half-sample, pi >= 0.4) with Pearson correlation (|r|), the
                  Elastic Net (non-zero |coefficients|) or BRS (pool stored in results_sample_size/), and single
                  full-data runs of each scorer (top 10): true and noise features selected
    combinations  the combination test of Step 4 (benchmark_pipeline_sim.gains, same permutations) on the Pearson
                  and Elastic Net pools; the BRS pool and the directly enumerated combinations are taken from
                  results_sample_size/pipeline_vs_direct.xlsx
    cost          wall-clock time of the scoring per half-sample and of the combination tests
Prescreen as in the benchmarks: |r| > 0.2 (direction unknown), at most 100 features.

Output: results_sample_size/scorers_sim.xlsx
Run:    python benchmark_scorers_sim.py --workers 4
"""
import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SIM_DIR = ROOT / "results_sample_size"
OUT_XLSX = SIM_DIR / "scorers_sim.xlsx"
N_LIST = [30, 50, 75, 100, 150, 200]
SEEDS = [42] + list(range(1, 10))
TOP_K, PI_GATE, R_MIN, CAP, SEED = 10, 0.4, 0.2, 100, 42


def rank_pearson(X, y):
    Xc, yc = X - X.mean(0), y - y.mean()
    r = (Xc * yc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    a = np.abs(np.nan_to_num(r))
    keep = [j for j in np.argsort(-a) if a[j] > R_MIN][:CAP]
    return keep, a


def rank_enet(X, y):
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    keep, a = rank_pearson(X, y)
    if not keep:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coef = np.abs(ElasticNetCV(l1_ratio=[0.7, 0.85, 1.0], cv=3, random_state=SEED, max_iter=2000)
                      .fit(StandardScaler().fit_transform(X[:, keep]), y).coef_)
    order = sorted(range(len(keep)), key=lambda i: (-coef[i], -a[keep[i]]))
    return [keep[i] for i in order if coef[i] > 0]


def run_one(n, seed):
    import torch
    torch.set_num_threads(1)
    import benchmark_comprehensive as bc
    import benchmark_pipeline_sim as bps
    import gated_combinations as gc

    X, y, fn, true_u, true_p, true_t = bc.generate_synthetic(n_samples=n, seed=seed)
    halves = gc.make_halves(n)
    rng = np.random.default_rng(seed)                     # same permutations as benchmark_pipeline_sim.py
    perms = [[rng.permutation(len(idx)) for _ in range(bps.N_PERM)] for idx in halves]
    truth = {fn.index(f) for f in true_u}
    tp = tuple(sorted(fn.index(f) for f in true_p))
    tt = tuple(sorted(fn.index(f) for f in true_t))
    out = {"n": n, "seed": seed}

    def sel(name, chosen):
        out[f"{name}: true selected"] = len(set(chosen) & truth)
        out[f"{name}: noise selected"] = len(set(chosen) - truth)
        out[f"{name}: planted pair members both selected"] = set(tp) <= set(chosen)

    gate = json.loads((SIM_DIR / f"n{n:03d}_s{seed:02d}.json").read_text(encoding="utf-8"))["gate"]
    sel("BRS + stability selection", [fn.index(f) for f in gate])

    pools = {}
    for name, ranker in (("Pearson", lambda X_, y_: rank_pearson(X_, y_)[0]), ("Elastic Net", rank_enet)):
        cnt, t0 = np.zeros(X.shape[1]), time.perf_counter()
        for idx in halves:
            for j in ranker(X[idx], y[idx])[:TOP_K]:
                cnt[j] += 1
        out[f"{name}: seconds per half-sample"] = (time.perf_counter() - t0) / len(halves)
        pools[name] = [int(j) for j in np.where(cnt / len(halves) >= PI_GATE)[0]]
        sel(f"{name} + stability selection", pools[name])
        sel(f"{name}, single run (top 10)", ranker(X, y)[:TOP_K])
    torch.manual_seed(42)
    np.random.seed(42)
    t0 = time.perf_counter()
    ranked, _ = bc.brs_univariate(X, y, fn)
    out["BRS: seconds per run"] = time.perf_counter() - t0
    sel("BRS, single run (top 10)", [fn.index(f) for f in ranked[:TOP_K]])

    for name, pool in pools.items():
        combos = [c for k in (2, 3) for c in combinations(pool, k)]
        t0 = time.perf_counter()
        res = bps.test_combos(X, y, combos, halves, perms)
        out[f"{name} pool: seconds for the combination test"] = time.perf_counter() - t0
        def kind(c):
            return ("planted pair" if c == tp else "planted triad" if c == tt else
                    "contains planted pair" if set(tp) <= set(c) else
                    "true features only" if set(c) <= truth else "contains noise")
        k = [kind(r["combo"]) for r in res]
        out[f"{name} pool: combinations tested"] = len(res)
        out[f"{name} pool: planted pair called"] = any(r["repro_int"] for r, kk in zip(res, k) if kk == "planted pair")
        out[f"{name} pool: interaction calls with a noise feature"] = sum(r["repro_int"] for r, kk in zip(res, k) if kk == "contains noise")
        out[f"{name} pool: complementarity calls with a noise feature"] = sum(r["repro_comp"] for r, kk in zip(res, k) if kk == "contains noise")
    # time per combination of the Step 4 test (for the cost of direct enumeration)
    if combos := [c for c in combinations(range(6), 2)]:
        t0 = time.perf_counter()
        bps.test_combos(X, y, combos, halves, perms)
        out["seconds per pair test"] = (time.perf_counter() - t0) / len(combos)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, n, s): (n, s) for n in N_LIST for s in SEEDS}
        for i, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            print(f"{i}/{len(futs)} done (n={futs[fu][0]}, seed={futs[fu][1]})", flush=True)
    D = pd.DataFrame(rows).sort_values(["n", "seed"])
    num = [c for c in D.columns if c not in ("n", "seed")]
    S = D.groupby("n")[num].agg(lambda v: v.sum() if (v.dtype == bool or "called" in v.name or "calls" in v.name) else v.mean()).reset_index()
    T = D[num].agg(lambda v: v.sum() if (v.dtype == bool or "called" in v.name or "calls" in v.name) else v.mean()).to_frame("all 60 datasets").T
    readme = pd.DataFrame([
        ("by_n / overall", "selected features: mean per dataset; 'both selected', 'called' and 'calls': number of datasets "
                           "or total calls over the datasets; seconds: mean"),
        ("truth", "five true single features (three linear, two non-linear), planted pair X_lin_1 x X_nonlin_1, planted "
                  "triad X_lin_2 x X_lin_3 x X_nonlin_2, 195 noise features"),
    ], columns=["sheet", "description"])
    with pd.ExcelWriter(OUT_XLSX) as w:
        readme.to_excel(w, sheet_name="README", index=False)
        T.round(3).to_excel(w, sheet_name="overall", index=False)
        S.round(3).to_excel(w, sheet_name="by_n", index=False)
        D.to_excel(w, sheet_name="per_dataset", index=False)
    pd.set_option("display.width", 250)
    print(T.round(2).T.to_string())
    print("Saved", OUT_XLSX)
