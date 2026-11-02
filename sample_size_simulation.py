# -*- coding: utf-8 -*-
"""
How many samples are needed to find a synergistic pair reproducibly? Sample-size simulation with the synthetic
generator of the benchmark (five single signals, one planted product pair, one planted three-way product, 195 noise
features), n in N_LIST, ten datasets per n (seeds 42 and 1-9), 50 complementary half-samples each.

Per dataset:
    gate         univariate BRS on every half-sample; pi = fraction in the top 10; gated pool = pi >= 0.4
    Step 4       gated combination test (gated_combinations.gains) on the gated pool: a pair or triad is called
                 reproducible if pi >= 0.7 and >= 0.3 above its permutation value (20 permutations per half-sample)
    oracle       the same gains for fixed reference sets, whether gated or not: the planted pair, a pair of true
                 features without interaction (X_lin_1 + X_lin_3), a noise pair and the planted triad
    TreeSHAP     direct enumeration without a pool: random forest (500 trees) on all features in each half-sample;
                 fraction of half-samples in which the planted pair is among the top 10 of all pairs (mean |phi_ij|)

Triad interaction gains are measured beyond the best model with a single pairwise product (RULE), so that a triad
containing the planted pair does not inherit its interaction.

Output: results_sample_size/ (one json per dataset, summary workbook sample_size_summary.xlsx)
Run: python sample_size_simulation.py --workers 10      (resumable: datasets finished under RULE are skipped;
     --reuse-treeshap keeps the TreeSHAP results of an earlier run, which do not depend on RULE)
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_sample_size"
N_LIST = [30, 50, 75, 100, 150, 200]
SEEDS = [42] + list(range(1, 10))
PI_GATE, TOP_K = 0.4, 10
PI_REPRO, MARGIN = 0.7, 0.3
REFS = {"planted pair": ("X_lin_1", "X_nonlin_1"), "true pair, no interaction": ("X_lin_1", "X_lin_3"),
        "noise pair": ("X_noise_1", "X_noise_2"), "planted triad": ("X_lin_2", "X_lin_3", "X_nonlin_2")}
RULE = "triad gain beyond best single pairwise product"
TS_KEYS = ("treeshap_pi_planted_pair_top10", "treeshap_median_rank_planted_pair")


def pis(gains_list, null_list, delta):
    return float(np.mean(np.array(gains_list) > delta)), float(np.mean(np.array(null_list) > delta))


def run_one(n, seed, reuse_treeshap=False):
    import torch
    from sklearn.ensemble import RandomForestRegressor
    torch.set_num_threads(1)
    import benchmark_comprehensive as bc
    import gated_combinations as gc

    X, y, fn, true_u, true_p, true_t = bc.generate_synthetic(n_samples=n, seed=seed)
    halves = gc.make_halves(n)
    rng = np.random.default_rng(seed)
    p = X.shape[1]

    # --- gate: univariate BRS on every half-sample ---
    cnt = np.zeros(p)
    for idx in halves:
        torch.manual_seed(42)
        np.random.seed(42)
        ranked, _ = bc.brs_univariate(X[idx], y[idx], fn)
        for f in ranked[:TOP_K]:
            cnt[fn.index(f)] += 1
    pi = cnt / len(halves)
    pool = [int(j) for j in np.where(pi >= PI_GATE)[0]]
    truth = {fn.index(f) for f in true_u}
    tp = tuple(sorted(fn.index(f) for f in true_p))
    tt = tuple(sorted(fn.index(f) for f in true_t))

    # --- Step 4 on the gated pool ---
    calls = []
    for r in gc.score_combos(X, y, pool, halves, rng) if len(pool) >= 2 else []:
        c = tuple(sorted(r["combo"]))
        kind = ("planted pair" if c == tp else "planted triad" if c == tt else
                "contains planted pair" if set(tp) <= set(c) else
                "true features, no interaction" if set(c) <= truth else "contains noise")
        calls.append({"members": [fn[j] for j in c], "kind": kind, "order": r["order"],
                      "pi_comp": r["pi_comp"], "pi_int": r["pi_int"],
                      "pi_int_null": r["pi_int_null"],
                      "repro_int": bool(r["pi_int"] >= PI_REPRO and r["pi_int"] >= r["pi_int_null"] + MARGIN),
                      "repro_comp": bool(r["pi_comp"] >= PI_REPRO and r["pi_comp"] >= r["pi_comp_null"] + MARGIN)})

    # --- oracle: fixed reference sets ---
    oracle = {}
    for name, members in REFS.items():
        cols_idx = [fn.index(m) for m in members]
        comp, inter, comp0, inter0 = [], [], [], []
        for idx in halves:
            cols = [X[idx, j] for j in cols_idx]
            a, b = gc.gains(cols, y[idx])
            comp.append(a)
            inter.append(b)
            for _ in range(gc.N_PERM):
                a0, b0 = gc.gains(cols, rng.permutation(y[idx]))
                comp0.append(a0)
                inter0.append(b0)
        pc, pc0 = pis(comp, comp0, gc.DELTA)
        pint, pint0 = pis(inter, inter0, gc.DELTA)
        oracle[name] = {"pi_comp": pc, "pi_comp_null": pc0, "pi_int": pint, "pi_int_null": pint0,
                        "repro_int": bool(pint >= PI_REPRO and pint >= pint0 + MARGIN)}

    # --- TreeSHAP direct enumeration (no pool) ---
    path = OUT / f"n{n:03d}_s{seed:02d}.json"
    if reuse_treeshap and path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        ts = {k: prev[k] for k in TS_KEYS}
    else:
        import shap
        iu = np.triu_indices(p, 1)
        flat_tp = int(np.where((iu[0] == tp[0]) & (iu[1] == tp[1]))[0][0])
        in_top, ranks = [], []
        for idx in halves:
            m = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=1).fit(X[idx], y[idx])
            phi = np.abs(np.asarray(shap.TreeExplainer(m).shap_interaction_values(X[idx]))).mean(0)[iu]
            rank = int((phi > phi[flat_tp]).sum()) + 1
            ranks.append(rank)
            in_top.append(rank <= TOP_K)
        ts = dict(zip(TS_KEYS, (float(np.mean(in_top)), float(np.median(ranks)))))

    res = {"n": n, "seed": seed, "rule": RULE, "gate_size": len(pool), "gate": [fn[j] for j in pool],
           "gate_true": len(truth & set(pool)), "gate_noise": len(set(pool) - truth),
           "pair_in_gate": set(tp) <= set(pool), "triad_in_gate": set(tt) <= set(pool),
           "pi_members_pair": [float(pi[j]) for j in tp], "calls": calls, "oracle": oracle, **ts}
    path.write_text(json.dumps(res), encoding="utf-8")
    return n, seed


def summarize():
    rows, call_rows = [], []
    for f in sorted(OUT.glob("n*_s*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rec = {k: r[k] for k in ("n", "seed", "gate_size", "gate_true", "gate_noise", "pair_in_gate", "triad_in_gate",
                                 "treeshap_pi_planted_pair_top10", "treeshap_median_rank_planted_pair")}
        for name, o in r["oracle"].items():
            for k, v in o.items():
                rec[f"{name}: {k}"] = v
        c = pd.DataFrame(r["calls"])
        pp = c[c.kind == "planted pair"] if len(c) else c
        rec["pipeline_finds_planted_pair"] = bool(len(pp) and pp.repro_int.any())
        planted = c.kind.isin(["planted pair", "planted triad", "contains planted pair"]) if len(c) else c
        rec["false_interaction_calls"] = int(c[~planted].repro_int.sum()) if len(c) else 0
        rec["nested_triad_calls"] = int(c[c.kind == "contains planted pair"].repro_int.sum()) if len(c) else 0
        rec["combos_tested"] = len(c)
        rows.append(rec)
        for x in r["calls"]:
            call_rows.append({"n": r["n"], "seed": r["seed"], **x})
    D = pd.DataFrame(rows).sort_values(["n", "seed"])
    agg = {"gate_size": "mean", "gate_true": "mean", "gate_noise": "mean", "pair_in_gate": "mean",
           "triad_in_gate": "mean", "treeshap_pi_planted_pair_top10": "mean",
           "pipeline_finds_planted_pair": "mean", "false_interaction_calls": "sum", "nested_triad_calls": "sum",
           "combos_tested": "sum"}
    for name in REFS:
        agg[f"{name}: pi_int"] = "mean"
        agg[f"{name}: pi_comp"] = "mean"
        agg[f"{name}: pi_int_null"] = "mean"
        agg[f"{name}: repro_int"] = "mean"
    S = D.groupby("n").agg(agg).reset_index()
    S.insert(1, "datasets", D.groupby("n").size().values)
    with pd.ExcelWriter(OUT / "sample_size_summary.xlsx") as w:
        pd.DataFrame([(l.strip(),) for l in __doc__.strip().splitlines() if l.strip()],
                     columns=["notes"]).to_excel(w, sheet_name="README", index=False)
        S.round(3).to_excel(w, sheet_name="by_n", index=False)
        D.round(3).to_excel(w, sheet_name="per_dataset", index=False)
        pd.DataFrame(call_rows).round(3).to_excel(w, sheet_name="step4_calls", index=False)
    pd.set_option("display.width", 250)
    print(S.round(2).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--reuse-treeshap", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    def done(n, s):
        f = OUT / f"n{n:03d}_s{s:02d}.json"
        return f.exists() and json.loads(f.read_text(encoding="utf-8")).get("rule") == RULE

    if not a.summary_only:
        todo = [(n, s) for n in N_LIST for s in SEEDS if not done(n, s)]
        todo.sort(key=lambda t: -t[0])                       # largest first, for load balancing
        print(f"{len(todo)} datasets to run", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_one, n, s, a.reuse_treeshap) for n, s in todo]
            for f in as_completed(futs):
                n, s = f.result()
                print(f"done n={n} seed={s}", flush=True)
    summarize()
