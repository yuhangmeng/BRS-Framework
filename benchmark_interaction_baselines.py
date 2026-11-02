# -*- coding: utf-8 -*-
"""
Interaction-Aware Baselines and Multi-Seed Benchmark for BRS
=============================================================
Extends benchmark_comprehensive.py (imported, NOT modified) with:

  (1) External interaction baselines: TreeSHAP interaction values
      (Lundberg et al. 2020, Nat Mach Intell 2:56-67) from one LightGBM and
      one RandomForest model fitted on ALL features.
        - univariate score : mean_n |SHAP_i|  (= |row sum of phi_i.|)
        - pair score       : mean_n |phi_ij|
        - triplet score    : |phi_ij| + |phi_ik| + |phi_jk| (pairwise-additive
                             PROXY; TreeSHAP has no native 3-way term)
  (2) Validity-only ablation: BRS with (alpha, beta, gamma) = (0, 0, 1).
  (3) Multi-seed evaluation of all methods on generate_synthetic(seed=s).
  (4) Reproducibility check of seed 42 against the published
      results/benchmark_comprehensive/full_benchmark_table.csv.

Pair / triplet protocol (same as every baseline in benchmark_comprehensive):
  candidate pool = the method's own top-20 univariate features; all C(20,2)
  pairs are ranked; triplets are ranked within 500 sampled combinations of
  C(20,3) that always include the true triplet. If a true member is not in
  the pool the rank is n/a. For SHAP-interaction the global rank of the true
  pair among all C(p,2) pairs is reported as a secondary, pool-free metric.

Outputs:
  results/benchmark_interaction/
    per_seed.csv                -- one row per (seed, method)
    summary.csv                 -- per-method medians / IQR / counts
    reproducibility_seed42.csv  -- seed-42 values vs. published table

Usage (from the repository root):
  python benchmark_interaction_baselines.py                  # all seeds
  python benchmark_interaction_baselines.py --seeds 42       # one seed
  python benchmark_interaction_baselines.py --workers 3      # seeds in parallel
  Seeds already complete in per_seed.csv are skipped (use --fresh to redo);
  per_seed.csv is merged on every write, so separate runs can add seeds.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# benchmark_comprehensive writes to CWD-relative paths on import -> run from repo root
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import benchmark_comprehensive as bc  # noqa: E402
import lightgbm as lgb  # noqa: E402
import shap  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

# ====================== Config ======================
OUT_DIR = Path("results/benchmark_interaction")
PUBLISHED_CSV = Path("results/benchmark_comprehensive/full_benchmark_table.csv")

SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TOP_K_POOL = bc.TOP_UNI_FOR_COMBO      # 20 -- candidate pool for pair/triplet ranks
MAX_COMBOS = 500                       # sampling cap hard-coded in bc (triplets only here)
PAIR_TOP = 10                          # "true pair in top-10" threshold

# Model settings copied verbatim from bc.lgbm_shap_rank / bc.rf_rank
SHAPINT_LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_data_in_leaf=3, random_state=42, verbose=-1)
SHAPINT_RF_PARAMS = dict(n_estimators=500, random_state=42, n_jobs=-1)

VALIDITY_ONLY_WEIGHTS = (0.0, 0.0, 1.0)   # (alpha, beta, gamma)

# Published methods first, in run_full_benchmark order (keeps bc.RNG stream identical)
PUBLISHED_METHODS = ["BRS", "LASSO", "ElasticNet", "RandomForest",
                     "LightGBM+SHAP", "PermutationImp", "MutualInfo"]
NEW_METHODS = ["BRS (validity only)", "SHAP-interaction (LightGBM)",
               "SHAP-interaction (RF)"]
METHODS = PUBLISHED_METHODS + NEW_METHODS

PAIR_DEF = {
    "BRS": "BRS pair ensemble on [xi,xj,xi*xj]",
    "BRS (validity only)": "BRS pair score, weights (0,0,1)",
    "LASSO": "|coef| of xi*xj in LassoCV refit",
    "ElasticNet": "|coef| of xi*xj in ElasticNetCV refit",
    "RandomForest": "impurity importance of xi*xj in RF refit",
    "LightGBM+SHAP": "mean|SHAP| of xi*xj in LightGBM refit",
    "SHAP-interaction (LightGBM)": "mean|phi_ij| (TreeSHAP interaction)",
    "SHAP-interaction (RF)": "mean|phi_ij| (TreeSHAP interaction)",
}
TRIPLET_DEF = {
    "BRS": "BRS triplet ensemble on [xi,xj,xk,tanh products]",
    "BRS (validity only)": "BRS triplet score, weights (0,0,1)",
    "RandomForest": "sum impurity importance of product terms in RF refit",
    "LightGBM+SHAP": "sum mean|SHAP| of product terms in LightGBM refit",
    "SHAP-interaction (LightGBM)": "PROXY: sum of 3 pairwise mean|phi|",
    "SHAP-interaction (RF)": "PROXY: sum of 3 pairwise mean|phi|",
}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ====================== RNG Alignment ======================

def _reset_rng(seed):
    """Reset every RNG stream bc uses, as a fresh import with SEED=seed would."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    bc.RNG = np.random.default_rng(seed)


def _burn_bootstrap_draws(n):
    """Consume the bc.RNG draws bc.bootstrap_stability would make (skipped here),
    so later triplet sampling sees the same stream as run_full_benchmark."""
    for _ in range(bc.STABILITY_B):
        bc.RNG.choice(n, size=max(2, int(0.8 * n)), replace=False)


@contextmanager
def _brs_weights(alpha, beta, gamma):
    """Temporarily override bc.BRS_ALPHA/BETA/GAMMA (read at call time by
    bc._brs_combo_rank); always restored."""
    old = (bc.BRS_ALPHA, bc.BRS_BETA, bc.BRS_GAMMA)
    bc.BRS_ALPHA, bc.BRS_BETA, bc.BRS_GAMMA = alpha, beta, gamma
    try:
        yield
    finally:
        bc.BRS_ALPHA, bc.BRS_BETA, bc.BRS_GAMMA = old


# ====================== SHAP-Interaction Baselines ======================

def _shap_interaction_scores(model, X):
    """Return (univariate mean|SHAP|, pair matrix mean|phi_ij|, additivity error)."""
    expl = shap.TreeExplainer(model)
    phi = expl.shap_interaction_values(X)
    if isinstance(phi, list):
        phi = phi[0]
    phi = np.asarray(phi, dtype=float)                 # (n, p, p)
    pair_mat = np.abs(phi).mean(0)                     # mean |phi_ij|
    uni = np.abs(phi.sum(2)).mean(0)                   # mean |SHAP_i|
    base = float(np.ravel(expl.expected_value)[0])
    add_err = float(np.abs(phi.sum((1, 2)) + base - model.predict(X)).max())
    return uni, pair_mat, add_err


def _matrix_scorers(X, pair_mat):
    """Scorers with bc's scorer_fn(cols, y) signature that look up pair_mat.
    Columns are mapped back to indices via their bytes (all columns distinct)."""
    key = {X[:, j].tobytes(): j for j in range(X.shape[1])}
    assert len(key) == X.shape[1], "duplicate columns: byte lookup ambiguous"

    def _idx(cols):
        return [key[np.ascontiguousarray(c).tobytes()] for c in cols]

    def pair_score(cols, y):
        i, j = _idx(cols)
        return float(pair_mat[i, j])

    def triplet_score(cols, y):
        i, j, k = _idx(cols)
        return float(pair_mat[i, j] + pair_mat[i, k] + pair_mat[j, k])

    return pair_score, triplet_score


def _global_pair_rank(pair_mat, feat_names, true_pair):
    """Rank of the true pair among all C(p,2) pairs (1 + #strictly greater),
    plus the number of other pairs tied with it."""
    iu = np.triu_indices(pair_mat.shape[0], 1)
    s = pair_mat[iu]
    t = pair_mat[feat_names.index(true_pair[0]), feat_names.index(true_pair[1])]
    return int((s > t).sum()) + 1, int(s.size), int((s == t).sum()) - 1


# ====================== Per-Method Evaluation ======================

def _combo_fields(prefix, rank, ranked, truth, n_cand, applicable):
    """Status / rank / candidate-set size columns for one combo metric."""
    in_pool = sum(m in ranked[:TOP_K_POOL] for m in truth)
    if not applicable:
        status = "not_applicable"
    elif rank is None:
        status = "not_in_pool" if in_pool < len(truth) else "missing"
    else:
        status = "ranked"
    return {
        f"{prefix}_rank": float(rank) if rank else np.nan,
        f"{prefix}_status": status,
        f"{prefix}_n_candidates": n_cand if rank else np.nan,
        f"{prefix}_members_in_pool": in_pool if applicable else np.nan,
    }


def run_method(name, X, y, fn, true_u, true_p, true_t):
    """Univariate recall, pair rank and triplet rank for one method."""
    k = min(TOP_K_POOL, X.shape[1])
    n_pair_cand = min(comb(k, 2), MAX_COMBOS)
    n_trip_cand = min(comb(k, 3), MAX_COMBOS)
    extra = {}

    # --- univariate ranking ---
    t0 = time.time()
    if name == "BRS (validity only)":
        # brs_univariate binds alpha/beta/gamma at def time -> pass explicitly
        ranked, _ = bc.brs_univariate(X, y, fn, alpha=0.0, beta=0.0, gamma=1.0)
    elif name.startswith("SHAP-interaction"):
        model = (lgb.LGBMRegressor(**SHAPINT_LGBM_PARAMS) if "LightGBM" in name
                 else RandomForestRegressor(**SHAPINT_RF_PARAMS))
        model.fit(X, y)
        uni, pair_mat, add_err = _shap_interaction_scores(model, X)
        ranked = [fn[j] for j in np.argsort(uni)[::-1]]
        pair_fn, trip_fn = _matrix_scorers(X, pair_mat)
        g_rank, g_n, g_ties = _global_pair_rank(pair_mat, fn, true_p)
        extra = dict(pair_rank_global=g_rank, pair_n_candidates_global=g_n,
                     pair_ties_global=g_ties, shap_additivity_err=add_err)
    else:
        ranked, _ = bc.ALL_METHODS[name](X, y, fn)
    t_uni = time.time() - t0

    # --- combo rankers (None = not applicable) ---
    if name == "BRS" or name == "BRS (validity only)":
        def pair_rk():
            return bc._brs_combo_rank(X, y, fn, ranked, true_p, combo_size=2)

        def trip_rk():
            return bc._brs_combo_rank(X, y, fn, ranked, true_t, combo_size=3)
    elif name.startswith("SHAP-interaction"):
        def pair_rk():
            return bc._baseline_combo_rank(X, y, fn, ranked, true_p, 2, pair_fn)

        def trip_rk():
            return bc._baseline_combo_rank(X, y, fn, ranked, true_t, 3, trip_fn)
    else:
        pair_rk = trip_rk = None
        if name in bc.BASELINE_PAIR_SCORERS:
            def pair_rk():
                return bc._baseline_combo_rank(X, y, fn, ranked, true_p, 2,
                                               bc.BASELINE_PAIR_SCORERS[name])
        if name in bc.BASELINE_TRIPLET_SCORERS:
            def trip_rk():
                return bc._baseline_combo_rank(X, y, fn, ranked, true_t, 3,
                                               bc.BASELINE_TRIPLET_SCORERS[name])

    weights = (VALIDITY_ONLY_WEIGHTS if name == "BRS (validity only)"
               else (bc.BRS_ALPHA, bc.BRS_BETA, bc.BRS_GAMMA))
    with _brs_weights(*weights):
        t0 = time.time()
        pr = pair_rk() if pair_rk else None
        t_pair = time.time() - t0
        t0 = time.time()
        tr = trip_rk() if trip_rk else None
        t_trip = time.time() - t0

    row = dict(
        method=name,
        recall_5=bc.recall_at_k(ranked, 5, true_u),
        recall_10=bc.recall_at_k(ranked, 10, true_u),
        recall_20=bc.recall_at_k(ranked, 20, true_u),
    )
    row.update(_combo_fields("pair", pr, ranked, true_p, n_pair_cand,
                             pair_rk is not None))
    row.update(_combo_fields("triplet", tr, ranked, true_t, n_trip_cand,
                             trip_rk is not None))
    row.update(pair_definition=PAIR_DEF.get(name, ""),
               triplet_definition=TRIPLET_DEF.get(name, ""))
    row.update(extra)
    row.update(runtime_uni_s=t_uni, runtime_pair_s=t_pair,
               runtime_triplet_s=t_trip, runtime_s=t_uni + t_pair + t_trip,
               top5="; ".join(ranked[:5]))
    return row


def run_seed(seed, n_workers=1):
    """All methods on generate_synthetic(seed); order mirrors run_full_benchmark."""
    _reset_rng(seed)
    X, y, fn, true_u, true_p, true_t = bc.generate_synthetic(seed=seed)
    rows = []
    for name in METHODS:
        _log(f"seed {seed:>2} | {name} ...")
        row = run_method(name, X, y, fn, true_u, true_p, true_t)
        if name in PUBLISHED_METHODS:
            _burn_bootstrap_draws(len(y))
        row.update(seed=seed, n_workers=n_workers)
        rows.append(row)
        _log(f"seed {seed:>2} | {name}: R@5/10/20={row['recall_5']:.1f}/"
             f"{row['recall_10']:.1f}/{row['recall_20']:.1f}  "
             f"pair={row['pair_rank']} ({row['pair_status']})  "
             f"triplet={row['triplet_rank']} ({row['triplet_status']})  "
             f"global_pair={row.get('pair_rank_global', '-')}  "
             f"[{row['runtime_s']:.1f}s]")
    return rows


# ====================== Summary & Checks ======================

def _ordered(df):
    df = df.copy()
    df["_s"] = df["seed"].map({s: i for i, s in enumerate(SEEDS)}).fillna(999)
    df["_m"] = df["method"].map({m: i for i, m in enumerate(METHODS)})
    df = df.sort_values(["_s", "seed", "_m"]).drop(columns=["_s", "_m"])
    lead = ["seed", "method"]
    return df[lead + [c for c in df.columns if c not in lead]]


def _rank_stats(d, prefix, top=None):
    """Median / IQR over ranked seeds, n/a counts, median with n/a as worst."""
    st = d[f"{prefix}_status"]
    if (st == "not_applicable").all():
        return {}
    ranks = d.loc[st == "ranked", f"{prefix}_rank"].astype(float)
    n_cand = d.loc[st == "ranked", f"{prefix}_n_candidates"].max()
    worst = (n_cand if np.isfinite(n_cand) else
             (comb(TOP_K_POOL, 2) if prefix == "pair"
              else min(comb(TOP_K_POOL, 3), MAX_COMBOS))) + 1
    filled = d[f"{prefix}_rank"].where(st == "ranked", worst).astype(float)
    q = (lambda p: float(np.percentile(ranks, p)) if len(ranks) else np.nan)
    out = {
        f"{prefix}_rank_median": q(50),
        f"{prefix}_rank_q1": q(25),
        f"{prefix}_rank_q3": q(75),
        f"{prefix}_n_ranked": int(len(ranks)),
        f"{prefix}_n_na": int((st != "ranked").sum()),
        f"{prefix}_rank_median_na_as_worst": float(np.median(filled)),
        f"{prefix}_n_candidates": n_cand,
    }
    if top is not None:
        out[f"{prefix}_n_top{top}"] = int((ranks <= top).sum())
    return out


def summarize(df):
    """Per-method aggregate over seeds."""
    rows = []
    for m in METHODS:
        d = df[df["method"] == m]
        if d.empty:
            continue
        r = dict(method=m, n_seeds=int(d["seed"].nunique()),
                 recall_5_mean=d["recall_5"].mean(),
                 recall_10_mean=d["recall_10"].mean(),
                 recall_20_mean=d["recall_20"].mean())
        r.update(_rank_stats(d, "pair", top=PAIR_TOP))
        if "pair_rank_global" in d and d["pair_rank_global"].notna().any():
            g = d["pair_rank_global"].dropna().astype(float)
            r.update(pair_global_rank_median=float(g.median()),
                     pair_global_rank_q1=float(np.percentile(g, 25)),
                     pair_global_rank_q3=float(np.percentile(g, 75)),
                     pair_global_n_top10=int((g <= PAIR_TOP).sum()),
                     pair_global_n_candidates=int(d["pair_n_candidates_global"].max()))
        r.update(_rank_stats(d, "triplet"))
        r.update(runtime_median_s=d["runtime_s"].median(),
                 runtime_total_s=d["runtime_s"].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def check_reproducibility(df):
    """Compare seed-42 results with the published full_benchmark_table.csv."""
    if not PUBLISHED_CSV.exists() or not (df["seed"] == 42).any():
        return None
    pub = pd.read_csv(PUBLISHED_CSV)
    d42 = df[df["seed"] == 42].set_index("method")
    rows = []
    for _, p in pub.iterrows():
        if p["method"] not in d42.index:
            continue
        r = d42.loc[p["method"]]
        for col in ["recall_5", "recall_10", "recall_20", "pair_rank",
                    "triplet_rank", "top5"]:
            pv, rv = p[col], r[col]
            if col == "top5":
                ok = str(pv).strip() == str(rv).strip()
            else:
                pv, rv = float(pv), float(rv)
                ok = (np.isnan(pv) and np.isnan(rv)) or abs(pv - rv) < 1e-9
            rows.append(dict(method=p["method"], metric=col, published=pv,
                             reproduced=rv, match=bool(ok)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reproducibility_seed42.csv", index=False)
    bad = out[~out["match"]]
    _log(f"Reproducibility (seed 42): {len(out) - len(bad)}/{len(out)} match")
    if len(bad):
        _log("MISMATCHES:\n" + bad.to_string(index=False))
    return out


def _write(df):
    """Merge with seeds already on disk (written by other runs), then save."""
    csv = OUT_DIR / "per_seed.csv"
    if csv.exists():
        disk = pd.read_csv(csv)
        df = pd.concat([disk[~disk["seed"].isin(df["seed"])], df],
                       ignore_index=True)
    df = _ordered(df)
    df.to_csv(csv, index=False)
    summarize(df).to_csv(OUT_DIR / "summary.csv", index=False)
    return df


# ====================== Main ======================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fresh", action="store_true",
                    help="rerun requested seeds even if already in per_seed.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "per_seed.csv"
    df = (pd.read_csv(csv) if csv.exists() and not args.fresh
          else pd.DataFrame())
    if len(df):
        complete = df.groupby("seed")["method"].nunique()
        done = set(complete[complete == len(METHODS)].index)
        df = df[df["seed"].isin(done)]
    else:
        done = set()
    todo = [s for s in seeds if s not in done]
    _log(f"shap {shap.__version__} | lightgbm {lgb.__version__} | "
         f"seeds to run: {todo} (already done: {sorted(done)}) | "
         f"workers={args.workers}")

    t_all = time.time()
    if args.workers <= 1:
        for s in todo:
            t0 = time.time()
            df = _write(pd.concat([df, pd.DataFrame(run_seed(s, 1))],
                                  ignore_index=True))
            _log(f"seed {s} done in {time.time() - t0:.0f}s -> {csv}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_seed, s, args.workers): s for s in todo}
            for f in as_completed(futs):
                df = _write(pd.concat([df, pd.DataFrame(f.result())],
                                      ignore_index=True))
                _log(f"seed {futs[f]} done -> {csv}")

    if len(df):
        df = _write(df)
        check_reproducibility(df)
        summ = summarize(df)
        cols = ["method", "n_seeds", "recall_10_mean", "recall_20_mean",
                "pair_rank_median", "pair_rank_q1", "pair_rank_q3",
                "pair_n_top10", "pair_n_na", "triplet_rank_median",
                "triplet_n_na", "runtime_median_s"]
        _log("Summary:\n" + summ[[c for c in cols if c in summ]].to_string(index=False))
    _log(f"Total elapsed: {time.time() - t_all:.0f}s | outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
