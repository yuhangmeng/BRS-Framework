# -*- coding: utf-8 -*-
"""
Nested cross-validation of the stability-selected consensus metabolite signature.

Outer loop: repeated stratified 5-fold CV (2 repeats) on the high-/low-BNI label.
Inner loop (training accessions only):
    - stability selection: N_INNER_PAIRS complementary pairs of half-samples, univariate BRS
      (Main Method.py, unchanged) on each; pi = fraction of half-samples in a strain's top-10;
      signature = metabolites with max_strain pi >= threshold (fallback: the FALLBACK_N highest pi);
    - naive signature: the FALLBACK_N metabolites with the highest single-run BRS on the training set;
    - best single metabolite: highest training AUC among all metabolites.
Each signature is scored on the held-out accessions by its composite score (mean of log-transformed,
autoscaled intensities; label-free). Out-of-fold scores are pooled within each repeat for the AUC.

    python nested_cv_signature.py --workers 4
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedStratifiedKFold

# ==========================================
# 1) Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "results_nestedcv"
RAW_PATH = ROOT / "Data Preprocessing" / "filtered_notfound_removed.xlsx"
LABEL_PATH = ROOT / "Data Preprocessing" / "filtered_metabolites.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
N_SPLITS, N_REPEATS = 5, 2
N_INNER_PAIRS = 10          # 20 half-samples per outer training set
TOP_K = 10
THRESHOLDS = [0.4, 0.3]
FALLBACK_N = 14
MAIN_Q = 0.75
SEED = 42
N_BOOT = 2000
N_RANDOM = 500
LABEL_QS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
THREADS_PER_WORKER = 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==========================================
# 2) Data, labels, composite score
# ==========================================
lab = pd.read_excel(LABEL_PATH, sheet_name=0)
bni = lab[[c for c in lab.columns if "Inhibition" in str(c)]]
y = (bni >= bni.quantile(MAIN_Q)).any(axis=1).astype(int).values
met_cols = [c for c in lab.columns if "Inhibition" not in str(c)]
raw = pd.read_excel(RAW_PATH, sheet_name=0, decimal=",")
X = raw[met_cols].apply(pd.to_numeric, errors="coerce")
X = X.apply(lambda s: s.where(s != 0, s[s != 0].min() / 2))
Z = np.log1p(X)
Z = (Z - Z.mean()) / Z.std()
Z.columns = [str(c).strip() for c in Z.columns]
# matrix used by the BRS prescreen (Main Method.py: Pearson r > 0.2 on the BRS input, BRS_INPUT)
X_pre = pd.read_excel(ROOT / os.environ.get("BRS_INPUT", "Data Preprocessing/filtered_metabolites.xlsx"),
                      sheet_name=0, decimal=",")[met_cols].apply(pd.to_numeric, errors="coerce")
X_pre.columns = [str(c).strip() for c in X_pre.columns]
BNI_VALS = bni.apply(pd.to_numeric, errors="coerce").values


def max_strain_r(rows):
    """Largest Pearson r with any strain's AOI% on the given rows (the BRS prescreen statistic)."""
    xs = X_pre.values[rows]
    out = np.full(xs.shape[1], -np.inf)
    for j in range(BNI_VALS.shape[1]):
        yy = BNI_VALS[rows, j]
        ok = np.isfinite(yy)
        xc, yc = xs[ok] - np.nanmean(xs[ok], axis=0), yy[ok] - yy[ok].mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nansum(xc * yc[:, None], axis=0) / np.sqrt(np.nansum(xc ** 2, axis=0) * (yc ** 2).sum())
        out = np.fmax(out, np.nan_to_num(r, nan=-np.inf))
    return pd.Series(out, index=X_pre.columns)


def auc(score, yy):
    r = rankdata(score)
    n1 = yy.sum()
    n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ==========================================
# 3) Univariate BRS jobs
# ==========================================
def uni_done(out: Path) -> bool:
    f = out / "univariate_BRS.xlsx"
    if not f.exists() or f.stat().st_size == 0:
        return False
    try:
        return "heatmap_values_wide" in pd.ExcelFile(f).sheet_names
    except Exception:
        return False


def run_uni(out: Path, idx: np.ndarray) -> str:
    out.mkdir(parents=True, exist_ok=True)
    if uni_done(out):
        return f"{out.relative_to(OUT_ROOT)} cached"
    np.savetxt(out / "samples.txt", np.sort(idx), fmt="%d")
    env = dict(os.environ, BRS_OUT_DIR=str(out), BRS_SAMPLE_FILE=str(out / "samples.txt"), MPLBACKEND="Agg",
               PYTHONIOENCODING="utf-8", LOKY_MAX_CPU_COUNT="1",
               OMP_NUM_THREADS=str(THREADS_PER_WORKER), MKL_NUM_THREADS=str(THREADS_PER_WORKER))
    with open(out / "run.log", "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-u", "Main Method.py"], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return f"{out.relative_to(OUT_ROOT)} {'FAILED' if r.returncode else 'done'}"


def top_lists(out: Path):
    xls = pd.ExcelFile(out / "univariate_BRS.xlsx")
    res = {}
    for s in STRAINS:
        t = pd.read_excel(xls, f"Inhibition (pct)-{s}_Uni_All").sort_values("BRS", ascending=False)
        res[s] = t[["Feature", "BRS"]].assign(Feature=lambda d: d["Feature"].astype(str).str.strip())
    return res


# ==========================================
# 4) Main
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(Z, y))

    jobs = []
    for f, (tr, te) in enumerate(outer):
        fold_dir = OUT_ROOT / f"fold_{f:02d}"
        jobs.append((fold_dir / "full", tr))
        for p in range(N_INNER_PAIRS):
            perm = rng.permutation(tr)
            k = len(tr) // 2
            jobs += [(fold_dir / f"half_{2 * p:02d}", perm[:k]), (fold_dir / f"half_{2 * p + 1:02d}", perm[k:2 * k])]
    log(f"{len(outer)} outer folds, {len(jobs)} univariate BRS runs, workers={args.workers}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_uni, d, idx) for d, idx in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            msg = fu.result()
            if "FAILED" in msg or i % 20 == 0 or i == len(futs):
                log(f"{i}/{len(futs)} {msg}")

    # ---- evaluate ----
    methods = [f"stability pi>={t}" for t in THRESHOLDS] + [f"naive top-{FALLBACK_N}", f"correlation top-{FALLBACK_N}",
                                                            "best single"]
    oof = {m: np.full((N_REPEATS, len(y)), np.nan) for m in methods}
    oof_rand = np.full((N_RANDOM, N_REPEATS, len(y)), np.nan)   # random signatures drawn from the prescreened pool
    rrng = np.random.default_rng(SEED + 1)
    fold_rows, membership = [], []
    for f, (tr, te) in enumerate(outer):
        rep = f // N_SPLITS
        fold_dir = OUT_ROOT / f"fold_{f:02d}"
        halves = sorted(fold_dir.glob("half_*"))
        cnt = {}
        for h in halves:
            for s, t in top_lists(h).items():
                for m in t["Feature"].head(TOP_K):
                    cnt.setdefault(m, dict.fromkeys(STRAINS, 0))[s] += 1
        pi = pd.DataFrame(cnt).T.reindex(columns=STRAINS).fillna(0) / len(halves)
        pi_max = pi.max(axis=1).sort_values(ascending=False)
        sigs = {}
        for t in THRESHOLDS:
            s_ = list(pi_max[pi_max >= t].index)
            sigs[f"stability pi>={t}"] = s_ if len(s_) >= 3 else list(pi_max.index[:FALLBACK_N])
        full = top_lists(fold_dir / "full")
        best = pd.concat([full[s].set_index("Feature")["BRS"] for s in STRAINS], axis=1).max(axis=1)
        sigs[f"naive top-{FALLBACK_N}"] = list(best.sort_values(ascending=False).index[:FALLBACK_N])
        r_tr = max_strain_r(tr)
        sigs[f"correlation top-{FALLBACK_N}"] = list(r_tr.sort_values(ascending=False).index[:FALLBACK_N])
        pool = [c for c in r_tr.index[r_tr > 0.2] if c in Z.columns]
        for b in range(N_RANDOM):
            oof_rand[b, rep, te] = Z[list(rrng.choice(pool, FALLBACK_N, replace=False))].mean(axis=1).values[te]
        tr_auc = {c: max(auc(Z[c].values[tr], y[tr]), 0) for c in Z.columns}
        sigs["best single"] = [max(tr_auc, key=tr_auc.get)]
        for m, ids in sigs.items():
            ids = [i for i in ids if i in Z.columns]
            score = Z[ids].mean(axis=1).values
            oof[m][rep, te] = score[te]
            fold_rows.append({"fold": f, "repeat": rep, "method": m, "n_metabolites": len(ids),
                              "train_AUC": auc(score[tr], y[tr]), "test_AUC": auc(score[te], y[te])})
            membership += [{"fold": f, "method": m, "metabolite": i} for i in ids]

    folds = pd.DataFrame(fold_rows)
    mem = pd.DataFrame(membership)

    # stratified bootstrap over accessions of the repeat-averaged pooled OOF AUC (paired across methods)
    def oof_auc(m, idx):
        return np.mean([auc(oof[m][r, idx], y[idx]) for r in range(N_REPEATS)])

    brng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boot = {m: [] for m in methods}
    for _ in range(N_BOOT):
        idx = np.concatenate([brng.choice(pos, len(pos)), brng.choice(neg, len(neg))])
        for m in methods:
            boot[m].append(oof_auc(m, idx))
    boot = {m: np.array(v) for m, v in boot.items()}

    def jaccard_mean(m):
        sets = [set(g["metabolite"]) for _, g in mem[mem.method == m].groupby("fold")]
        return np.mean([len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]])

    summ = []
    for m in methods:
        pooled = [auc(oof[m][r], y) for r in range(N_REPEATS)]
        d = folds[folds.method == m]
        summ.append({"method": m, "pooled_OOF_AUC_mean": np.mean(pooled), "pooled_OOF_AUC_by_repeat": ", ".join(f"{v:.3f}" for v in pooled),
                     "OOF_AUC_CI_low": np.percentile(boot[m], 2.5), "OOF_AUC_CI_high": np.percentile(boot[m], 97.5),
                     "fold_test_AUC_mean": d["test_AUC"].mean(), "fold_test_AUC_sd": d["test_AUC"].std(),
                     "train_AUC_mean": d["train_AUC"].mean(), "n_metabolites_mean": d["n_metabolites"].mean(),
                     "membership_jaccard_across_folds": jaccard_mean(m)})
    rand_auc = np.array([np.mean([auc(oof_rand[b, r], y) for r in range(N_REPEATS)]) for b in range(N_RANDOM)])
    summ.append({"method": f"random {FALLBACK_N} from prescreen (n={N_RANDOM})", "pooled_OOF_AUC_mean": rand_auc.mean(),
                 "OOF_AUC_CI_low": np.percentile(rand_auc, 2.5), "OOF_AUC_CI_high": np.percentile(rand_auc, 97.5),
                 "n_metabolites_mean": FALLBACK_N})
    summ = pd.DataFrame(summ)
    for m in methods[:3]:   # how often a random draw from the prescreened pool does at least as well
        summ.loc[summ.method == m, "frac_random_draws_>="] = np.mean(rand_auc >= summ.loc[summ.method == m, "pooled_OOF_AUC_mean"].item())
    # labelling-threshold sensitivity: signatures are selected from continuous AOI%, so the same out-of-fold
    # scores can be evaluated against high-/low-BNI labels defined at other percentiles
    thr_rows = []
    for q in LABEL_QS:
        yq = (bni >= bni.quantile(q)).any(axis=1).astype(int).values
        rec = {"label_percentile": int(round(q * 100)), "n_high": int(yq.sum()), "n_low": int(len(yq) - yq.sum())}
        for m in methods[:-1]:   # 'best single' is chosen with the 75th-percentile labels, so it is left out
            rec[m] = np.mean([auc(oof[m][r], yq) for r in range(N_REPEATS)])
        thr_rows.append(rec)
    thr = pd.DataFrame(thr_rows)
    ref = methods[0]
    diffs = pd.DataFrame([{"method": ref, "versus": m,
                           "delta_OOF_AUC": np.mean([auc(oof[ref][r], y) - auc(oof[m][r], y) for r in range(N_REPEATS)]),
                           "CI_low": np.percentile(boot[ref] - boot[m], 2.5), "CI_high": np.percentile(boot[ref] - boot[m], 97.5),
                           "P_boot(delta<=0)": np.mean(boot[ref] - boot[m] <= 0)} for m in methods[1:]])
    freq = (mem.groupby(["method", "metabolite"]).size() / len(outer)).rename("fraction_of_folds").reset_index()
    oof_long = pd.DataFrame([{"method": m, "repeat": r, "sample": i, "y": int(y[i]), "oof_score": oof[m][r, i]}
                             for m in methods for r in range(N_REPEATS) for i in range(len(y))])
    with pd.ExcelWriter(OUT_ROOT / "nested_cv_summary.xlsx") as w:
        summ.to_excel(w, sheet_name="summary", index=False)
        diffs.to_excel(w, sheet_name="paired_differences", index=False)
        thr.to_excel(w, sheet_name="label_thresholds", index=False)
        oof_long.to_excel(w, sheet_name="oof_scores", index=False)
        folds.to_excel(w, sheet_name="folds", index=False)
        freq.sort_values(["method", "fraction_of_folds"], ascending=[True, False]).to_excel(w, sheet_name="signature_membership", index=False)
        mem.to_excel(w, sheet_name="membership_by_fold", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(diffs.round(3).to_string(index=False))
    print(thr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
