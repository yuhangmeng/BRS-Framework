# -*- coding: utf-8 -*-
"""
Step 3 and the evaluation of the pipeline: strict nested cross-validation (main analysis of the manuscript).

Everything that is estimated from data is estimated inside each outer training set (5 folds x 2 repeats, seed 42):
    preprocessing   half-minimum imputation of zeros, log1p and autoscaling with constants from the training
                    accessions (fold-specific BRS input, also used for the composite score)
    labels          high/low inhibition: 75th percentile of each strain's AOI% in the training accessions; the
                    held-out accessions are labelled with these training thresholds
    selection       prescreen, BRS and stability selection on 20 training half-samples (10 complementary pairs),
                    consensus list = max-strain pi >= 0.4
    mixture size    k chosen inside the training set for the order it is applied to: the top-k composite of the
                    training set's selection-probability order is scored on each half of the ten pairs of training
                    half-samples (transform and label thresholds from the other half); smallest k within 0.02 of
                    the inner maximum
The outer test accessions are used only to evaluate. Comparators in the same folds: single BRS run (top 14),
Pearson correlation and Elastic Net as the stability-selection scorer, and their single runs (top 14). The final
mixtures use the same size rule with the full-data half-samples of stability_selection.py.
The AOI% response in data/preprocessing/filtered_metabolites.xlsx is autoscaled on all accessions; this leaves the
labels unchanged (Supplementary Methods S4 of the manuscript).

Run:  python code/nested_cv.py --run --workers 6     (210 BRS runs, about 40 min; resumable)
      python code/nested_cv.py --eval                (needs the outputs of stability_selection.py)
Output: results/nested_cv_runs/ (fold inputs and BRS runs), results/nested_cv.xlsx
"""
import argparse
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "nested_cv_runs"
STAB_UNI = ROOT / "results" / "stability_runs"                       # full-data half-samples (final model)
STAB_XLSX = ROOT / "results" / "stability_selection.xlsx"
LAYOUT = ROOT / "data" / "preprocessing" / "filtered_metabolites.xlsx"
RAW_PATH = ROOT / "data" / "preprocessing" / "filtered_notfound_removed.xlsx"
ACCESSIONS = ROOT / "data" / "preprocessing" / "accessions.csv"
OUT_XLSX = ROOT / "results" / "nested_cv.xlsx"
N_INNER_PAIRS = 10
STRAINS = ["NF", "NV", "NM", "NU"]
N_SPLITS, N_REPEATS, SEED, TOP_K, Q = 5, 2, 42, 10, 0.75
K_MAX = {"overall": 30, "NF": 15, "NV": 15, "NM": 15, "NU": 15}
PARSIMONY, PI_THR, FALLBACK_N, R_MIN, CAP = 0.02, 0.4, 14, 0.2, 100
N_BOOT, THREADS_PER_WORKER, N_RANDOM = 2000, 2, 500
THRESHOLDS = [0.3, 0.4, 0.5, 0.6]
LABEL_QS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==========================================
# 1) Data
# ==========================================
lab = pd.read_excel(LAYOUT, sheet_name="Sheet1", decimal=",")
TCOLS = [c for c in lab.columns if "Inhibition" in str(c)]
METS = [c for c in lab.columns if c not in TCOLS]
RAW = pd.read_excel(RAW_PATH, sheet_name=0, decimal=",")[METS].apply(pd.to_numeric, errors="coerce")
KEYS = [str(c).strip() for c in METS]
AOI = lab[[[c for c in TCOLS if c.endswith(s)][0] for s in STRAINS]].apply(pd.to_numeric, errors="coerce")
AOI.columns = STRAINS
n = len(RAW)
COLS = {"overall": STRAINS, **{s: [s] for s in STRAINS}}


def transform(train_idx):
    """Half-minimum imputation, log1p and autoscaling with constants from train_idx (all rows transformed)."""
    Xt = RAW.iloc[train_idx]
    X = RAW.where(RAW != 0, Xt.where(Xt != 0).min() / 2, axis=1)
    L = np.log1p(X)
    Z = (L - L.iloc[train_idx].mean()) / L.iloc[train_idx].std()
    Z.columns = KEYS
    return Z


def labels(train_idx, target):
    """Labels of all rows with the training 75th percentile(s)."""
    q = AOI.iloc[train_idx].quantile(Q)
    hi = AOI >= q
    return (hi.any(axis=1) if target == "overall" else hi[target]).astype(int).values


Y_GLOBAL = {t: labels(np.arange(n), t) for t in COLS}
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

OUTER = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
             .split(np.zeros(n), Y_GLOBAL["overall"]))


def inner_halves():
    """10 complementary pairs of half-samples per outer training set; one generator (seed 42) over all folds."""
    rng = np.random.default_rng(SEED)
    out = []
    for tr, _ in OUTER:
        h = []
        for _ in range(N_INNER_PAIRS):
            perm = rng.permutation(tr)
            k = len(tr) // 2
            h += [np.sort(perm[:k]), np.sort(perm[k:2 * k])]
        out.append(h)
    return out


# ==========================================
# 2) BRS runs on fold-specific inputs
# ==========================================
def uni_done(out: Path) -> bool:
    return (out / "DONE").exists()


def run_uni(out: Path, idx, inp: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    if uni_done(out):
        return f"{out.relative_to(OUT_ROOT)} cached"
    np.savetxt(out / "samples.txt", np.sort(idx), fmt="%d")
    env = dict(os.environ, BRS_INPUT=str(inp), BRS_OUT_DIR=str(out), BRS_SAMPLE_FILE=str(out / "samples.txt"),
               MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", LOKY_MAX_CPU_COUNT="1",
               OMP_NUM_THREADS=str(THREADS_PER_WORKER), MKL_NUM_THREADS=str(THREADS_PER_WORKER))
    with open(out / "run.log", "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-u", str(ROOT / "code" / "brs.py")], env=env, stdout=fh,
                           stderr=subprocess.STDOUT)
    return f"{out.relative_to(OUT_ROOT)} {'FAILED' if r.returncode else 'done'}"


def run(workers):
    jobs = []
    halves = inner_halves()
    for f, (tr, te) in enumerate(OUTER):
        fd = OUT_ROOT / f"fold_{f:02d}"
        fd.mkdir(parents=True, exist_ok=True)
        inp = fd / "input.xlsx"
        if not inp.exists():
            out = pd.concat([transform(tr).set_axis(METS, axis=1), lab[TCOLS]], axis=1)[list(lab.columns)]
            out.to_excel(inp, sheet_name="Sheet1", index=False)
        jobs.append((fd / "full", tr, inp))
        for i, h in enumerate(halves[f]):
            jobs.append((fd / f"half_{i:02d}", h, inp))
    log(f"{len(jobs)} univariate BRS runs on fold-specific inputs, workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_uni, d, idx, inp) for d, idx, inp in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            msg = fu.result()
            if "FAILED" in msg or i % 10 == 0 or i == len(futs):
                log(f"{i}/{len(futs)} {msg}")


# ==========================================
# 3) Scorers on a sample subset
# ==========================================
def brs_ranked(xlsx):
    x = pd.ExcelFile(xlsx)
    return {s: pd.read_excel(x, f"Inhibition (pct)-{s}_Uni_All").assign(
        Feature=lambda d: d["Feature"].astype(str).str.strip()).set_index("Feature")["BRS"].sort_values(ascending=False)
        for s in STRAINS}


def prescreen(Z, idx, s):
    X, y = Z.values[idx], AOI[s].values[idx]
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    Xc, yc = X - X.mean(0), y - y.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    r = np.where(np.isfinite(r), r, -np.inf)
    keep = [j for j in np.argsort(-r) if r[j] > R_MIN][:CAP]
    return keep, r, X, y


def simple_ranked(Z, idx, scorer):
    """{strain: Series of scores, best first} for Pearson or Elastic Net (non-zero coefficients only)."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    out = {}
    for s in STRAINS:
        keep, r, X, y = prescreen(Z, idx, s)
        if scorer == "Pearson" or not keep:
            out[s] = pd.Series({KEYS[j]: r[j] for j in keep})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coef = np.abs(ElasticNetCV(l1_ratio=[0.7, 0.85, 1.0], cv=3, random_state=SEED, max_iter=2000)
                          .fit(StandardScaler().fit_transform(X[:, keep]), y).coef_)
        order = sorted(range(len(keep)), key=lambda i: (-coef[i], -r[keep[i]]))
        out[s] = pd.Series({KEYS[keep[i]]: coef[i] + 1e-9 * r[keep[i]] for i in order if coef[i] > 0})
    return out


def pi_from(rankings):
    cnt = {}
    for rk in rankings:
        for s in STRAINS:
            for m in list(rk[s].index[:TOP_K]):
                cnt.setdefault(m, dict.fromkeys(STRAINS, 0))[s] += 1
    return pd.DataFrame(cnt).T.reindex(columns=STRAINS).fillna(0) / len(rankings)


def order_from(pi, full_scores, cols):
    """Metabolites with pi > 0 by max pi over cols, ties by the full-training score (as incremental_signature.py)."""
    p = pi[cols].max(axis=1)
    b = pd.concat([full_scores[s] for s in cols], axis=1).max(axis=1).reindex(pi.index).fillna(-np.inf)
    d = pd.DataFrame({"p": p, "b": b})
    d = d[(d.p > 0) & d.index.isin(KEYS)]
    return list(d.sort_values(["p", "b"], ascending=False, kind="mergesort").index)


def single_top(full_scores, k=FALLBACK_N):
    """Top k of one run over strains: best within-strain rank."""
    best = {}
    for s in STRAINS:
        for i, m in enumerate(full_scores[s].index):
            best[m] = min(best.get(m, np.inf), i)
    return sorted(best, key=best.get)[:k]


def auc(score, yy):
    r = rankdata(score)
    n1 = yy.sum()
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))


def inner_k(pairs, target, order, kmax, Ztr_fn):
    """k from the training half-samples only, for the order it is applied to (the training set's pi order): the
    top-k composite is scored on each half of every pair, with transform and label thresholds from the other half.
    The order uses all training half-samples, so the inner curve is optimistic; this affects only the choice of k,
    never the outer test accessions."""
    curves = []
    for (ia, _), (ib, _) in pairs:
        for i_fit, i_val in ((ia, ib), (ib, ia)):
            yv = labels(i_fit, target)[i_val]       # threshold from the fitting half, as for the outer test
            if yv.sum() in (0, len(yv)):
                continue
            Zf = Ztr_fn(i_fit)
            curves.append([auc(Zf[order[:k]].mean(axis=1).values[i_val], yv) for k in range(1, kmax + 1)])
    c = np.nanmean(np.array(curves), axis=0)
    return int(np.where(c >= c.max() - PARSIMONY)[0].min()) + 1, c


# ==========================================
# 4) Evaluation
# ==========================================
def evaluate():
    rng = np.random.default_rng(SEED)
    rrng = np.random.default_rng(SEED + 1)
    oof, ylab = {}, {}                      # oof[(method, target)][repeat, sample]; ylab[target][repeat, sample]
    lists, ks, fold_rows = {}, {}, []
    blank = lambda: np.full((N_REPEATS, n), np.nan)
    for t in COLS:
        ylab[t] = blank()
    for f, (tr, te) in enumerate(OUTER):
        rep = f // N_SPLITS
        fd = OUT_ROOT / f"fold_{f:02d}"
        Ztr = transform(tr)
        for t in COLS:
            ylab[t][rep, te] = labels(tr, t)[te]
        halves = [(np.loadtxt(h / "samples.txt", dtype=int), h.name) for h in sorted(fd.glob("half_*"))]
        rk = {"BRS": [brs_ranked(fd / name / "univariate_BRS.xlsx") for _, name in halves]}
        full = {"BRS": brs_ranked(fd / "full" / "univariate_BRS.xlsx")}
        for sc in ("Pearson", "Elastic Net"):
            rk[sc] = [simple_ranked(Ztr, idx, sc) for idx, _ in halves]
            full[sc] = simple_ranked(Ztr, tr, sc)
        pairs = [((halves[i][0], rk["BRS"][i]), (halves[i + 1][0], rk["BRS"][i + 1])) for i in range(0, len(halves), 2)]
        # comparators of the consensus list: random sets of 14 prescreened metabolites, best single metabolite
        pre = sorted({KEYS[j] for s in STRAINS for j in prescreen(Ztr, tr, s)[0]})
        yt = labels(tr, "overall")
        for b in range(N_RANDOM):
            oof.setdefault(("random set of 14", b), blank())[rep, te] = Ztr[list(rrng.choice(pre, FALLBACK_N, replace=False))].mean(axis=1).values[te]
        tr_auc = {m: auc(Ztr[m].values[tr], yt[tr]) for m in pre}
        best1 = max(tr_auc, key=tr_auc.get)
        oof.setdefault(("best single metabolite (training AUC)", "overall"), blank())[rep, te] = Ztr[best1].values[te]
        for qq in LABEL_QS:
            q = AOI.iloc[tr].quantile(qq)
            ylab.setdefault(("label q", qq), blank())[rep, te] = (AOI >= q).any(axis=1).astype(int).values[te]
        pm_brs = pi_from(rk["BRS"]).max(axis=1).sort_values(ascending=False)
        for thr in THRESHOLDS:                          # selection-threshold sensitivity of the BRS consensus list
            m = [x for x in pm_brs.index[pm_brs >= thr] if x in KEYS] or [x for x in pm_brs.index if x in KEYS][:FALLBACK_N]
            oof.setdefault((f"consensus list, threshold {thr}", "overall"), blank())[rep, te] = Ztr[m].mean(axis=1).values[te]
            lists.setdefault(f"consensus list, threshold {thr}", []).append(m)
        for sc in ("BRS", "Pearson", "Elastic Net"):
            pi = pi_from(rk[sc])
            pm = pi.max(axis=1).sort_values(ascending=False)
            cons = [x for x in pm.index[pm >= PI_THR] if x in KEYS] or [x for x in pm.index if x in KEYS][:FALLBACK_N]
            one = [x for x in single_top(full[sc]) if x in KEYS]
            for name, m in ((f"{sc} + stability selection (consensus list)", cons), (f"{sc}, single run (top 14)", one)):
                oof.setdefault((name, "overall"), blank())[rep, te] = Ztr[m].mean(axis=1).values[te]
                lists.setdefault(name, []).append(m)
            if sc != "BRS":
                continue
            for t, cols in COLS.items():
                o = order_from(pi, full["BRS"], cols)
                ki, _ = inner_k(pairs, t, o, K_MAX[t], transform)
                ks.setdefault(t, []).append(ki)
                for name, k in (("mixture, k chosen in training", ki), ("first metabolite (k = 1)", 1)):
                    oof.setdefault((name, t), blank())[rep, te] = Ztr[o[:k]].mean(axis=1).values[te]
                for k in range(1, K_MAX[t] + 1):
                    oof.setdefault((f"fixed k = {k}", t), blank())[rep, te] = Ztr[o[:k]].mean(axis=1).values[te]
                fold_rows.append({"fold": f, "repeat": rep, "target": t, "k chosen in training": ki,
                                  "members": "; ".join(o[:ki]), "test AUC": auc(Ztr[o[:ki]].mean(axis=1).values[te],
                                                                                ylab[t][rep, te])
                                  if 0 < ylab[t][rep, te].sum() < len(te) else np.nan})
        log(f"fold {f} evaluated")

    # pooled out-of-fold AUC (mean over repeats) with a stratified bootstrap shared by all methods of a target
    def pooled(o, yl, idx=None):
        idx = np.arange(n) if idx is None else idx
        return float(np.mean([auc(o[r, idx], yl[r, idx]) for r in range(N_REPEATS)]))
    boots = {}
    for t in COLS:
        yg = Y_GLOBAL[t]
        pos, neg = np.where(yg == 1)[0], np.where(yg == 0)[0]
        boots[t] = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(N_BOOT)]

    def summary(name, t, ref=None):
        o, yl = oof[(name, t)], ylab[t]
        b = np.array([pooled(o, yl, i) for i in boots[t]])
        rec = {"target": t, "method": name, "nested-CV AUC": pooled(o, yl),
               "CI low": np.percentile(b, 2.5), "CI high": np.percentile(b, 97.5),
               "AUC with all-accession labels": pooled(o, np.tile(Y_GLOBAL[t], (N_REPEATS, 1)))}
        if ref is not None:
            br = np.array([pooled(oof[(ref, t)], yl, i) for i in boots[t]])
            rec.update({"difference from " + ref: rec["nested-CV AUC"] - pooled(oof[(ref, t)], yl),
                        "difference CI low": np.percentile(b - br, 2.5), "difference CI high": np.percentile(b - br, 97.5)})
        return rec

    main_rows = []
    REF = "BRS + stability selection (consensus list)"
    for name in [REF, "BRS, single run (top 14)", "Pearson + stability selection (consensus list)",
                 "Pearson, single run (top 14)", "Elastic Net + stability selection (consensus list)",
                 "Elastic Net, single run (top 14)"]:
        r = summary(name, "overall", None if name == REF else REF)
        L = lists[name]
        r["list size (mean)"] = float(np.mean([len(x) for x in L]))
        r["membership Jaccard"] = float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in combinations(L, 2)]))
        main_rows.append(r)
    main_rows.append(summary("best single metabolite (training AUC)", "overall", REF))
    rand = np.array([pooled(oof[("random set of 14", b)], ylab["overall"]) for b in range(N_RANDOM)])
    cons = main_rows[0]["nested-CV AUC"]
    main_rows.append({"target": "overall", "method": "random sets of 14 prescreened metabolites (500)",
                      "nested-CV AUC": float(rand.mean()), "CI low": float(np.percentile(rand, 2.5)),
                      "CI high": float(np.percentile(rand, 97.5)),
                      "difference from " + REF: float(rand.mean() - cons),
                      "share of random sets below the consensus list": float((rand < cons).mean())})
    label_rows = [{"label percentile (training accessions)": int(round(100 * qq)),
                   "consensus list AUC": pooled(oof[(REF, "overall")], ylab[("label q", qq)])} for qq in LABEL_QS]
    p_full = pd.read_excel(STAB_XLSX, sheet_name="pi_single")
    p_full = p_full.set_index(p_full["key"].astype(str).str.strip())[[f"pi_{s}" for s in STRAINS]].max(axis=1)
    thr_rows = []
    for thr in THRESHOLDS:
        L = lists[f"consensus list, threshold {thr}"]
        thr_rows.append({"selection threshold": thr, "metabolites selected on all accessions": int((p_full >= thr).sum()),
                         "list size per fold: min": min(map(len, L)), "max": max(map(len, L)),
                         "nested-CV AUC": pooled(oof[(f"consensus list, threshold {thr}", "overall")], ylab["overall"])})
    cons_full = [k for k in p_full.index[p_full >= PI_THR]]
    resel_rows = [{"metabolite key": k, "full-data max pi": float(p_full[k]),
                   "re-selected in outer folds (of 10)": sum(k in L for L in lists[REF])} for k in cons_full]
    mix_rows = []
    for t in COLS:
        r = summary("mixture, k chosen in training", t, "first metabolite (k = 1)")
        r.update({"k per fold": ", ".join(map(str, ks[t])), "k median": float(np.median(ks[t])),
                  "first metabolite AUC": pooled(oof[("first metabolite (k = 1)", t)], ylab[t])})
        mix_rows.append(r)
    curve_rows = []
    for t in COLS:
        for k in range(1, K_MAX[t] + 1):
            o = oof[(f"fixed k = {k}", t)]
            b = [pooled(o, ylab[t], i) for i in boots[t][:500]]
            curve_rows.append({"target": t, "k": k, "nested-CV AUC (fixed k)": pooled(o, ylab[t]),
                               "CI low": np.percentile(b, 2.5), "CI high": np.percentile(b, 97.5)})
    oof_rows = [{"target": t, "method": name, "repeat": r, "sample": i, "y": int(ylab[t][r, i]), "oof_score": oof[(name, t)][r, i]}
                for t in COLS for name in ("mixture, k chosen in training", "first metabolite (k = 1)")
                for r in range(N_REPEATS) for i in range(n)]

    # final model on all accessions: k by the same inner rule on the 25 full-data half-sample pairs
    subs = sorted(d for d in STAB_UNI.glob("sub_*") if (d / "DONE").exists())
    fr = [(np.loadtxt(d / "samples.txt", dtype=int), brs_ranked(d / "univariate_BRS.xlsx")) for d in subs]
    fpairs = [(fr[i], fr[i + 1]) for i in range(0, len(fr), 2)]
    for a, b in fpairs:
        assert not set(a[0]) & set(b[0]), "full-data half-samples are not complementary pairs"
    final_rows = []
    from scipy.stats import spearmanr
    origin = pd.read_csv(ACCESSIONS, index_col=0)["origin"].values        # same row order as the input files
    assert len(origin) == n
    Z_all = transform(np.arange(n))
    p_strain = pd.read_excel(STAB_XLSX, sheet_name="pi_single")
    p_strain = p_strain.set_index(p_strain["key"].astype(str).str.strip())
    brs_full = {s: p_strain[f"BRS_full_{s}"] for s in STRAINS}
    pi_full = p_strain[[f"pi_{s}" for s in STRAINS]].set_axis(STRAINS, axis=1)
    for t, cols in COLS.items():
        o_full = order_from(pi_full, brs_full, cols)
        kf, curve = inner_k(fpairs, t, o_full, K_MAX[t], transform)
        members = o_full[:kf]
        rec = {"target": t, "k for the final mixture (inner rule on 25 full-data pairs)": kf,
               "members with pi >= 0.4": int((pi_full.loc[members, cols].max(axis=1) >= PI_THR).sum()),
               "members": "; ".join(members), "inner AUC curve": "; ".join(f"{v:.3f}" for v in curve)}
        if t != "overall":
            sc = Z_all[members].mean(axis=1).values
            for o in ("Austria", "India"):
                rec[f"Spearman with AOI%, {o}"] = spearmanr(sc[origin == o], AOI[t].values[origin == o]).correlation
        final_rows.append(rec)

    readme = pd.DataFrame([
        ("main", "consensus list and comparators (overall high-BNI label): nested-CV AUC with stratified bootstrap 95% CI, "
                 "paired difference from the BRS consensus list, list size and membership Jaccard across the ten folds"),
        ("mixtures", "mixture with k chosen inside each training set vs the first metabolite alone, per target; paired "
                     "bootstrap difference; k selected in each of the ten outer folds"),
        ("curves", "nested-CV AUC for fixed k with bootstrap 95% CI (descriptive; choosing k on this curve is optimistic)"),
        ("label_thresholds", "consensus list scored against labels defined at the 60th-85th percentile of the training accessions"),
        ("final_mixtures", "k of the final mixtures, chosen with the same inner rule on all accessions"),
        ("folds", "per outer fold: k, members and test AUC"),
        ("preprocessing", "imputation, log1p, autoscaling and labels from the training accessions of each fold; "
                          "'AUC with all-accession labels' uses the 75th percentile of all 44 accessions instead"),
    ], columns=["sheet", "description"])
    # unrounded: Table 1, Table 3 and Fig. 4 round these once to two decimals (rounding a 3-decimal copy again
    # turned 0.64496 into 0.65 and 0.59535 into 0.59)
    with pd.ExcelWriter(OUT_XLSX) as w:
        readme.to_excel(w, sheet_name="README", index=False)
        pd.DataFrame(main_rows).to_excel(w, sheet_name="main", index=False)
        pd.DataFrame(mix_rows).to_excel(w, sheet_name="mixtures", index=False)
        pd.DataFrame(curve_rows).to_excel(w, sheet_name="curves", index=False)
        pd.DataFrame(label_rows).to_excel(w, sheet_name="label_thresholds", index=False)
        pd.DataFrame(thr_rows).to_excel(w, sheet_name="selection_thresholds", index=False)
        pd.DataFrame(resel_rows).to_excel(w, sheet_name="reselection", index=False)
        pd.DataFrame(final_rows).to_excel(w, sheet_name="final_mixtures", index=False)
        pd.DataFrame(fold_rows).to_excel(w, sheet_name="folds", index=False)
        pd.DataFrame(oof_rows).to_excel(w, sheet_name="oof_scores", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(main_rows).round(3).to_string(index=False), "\n")
    print(pd.DataFrame(mix_rows).round(3).to_string(index=False), "\n")
    print(pd.DataFrame(final_rows)[["target", "k for the final mixture (inner rule on 25 full-data pairs)"]].to_string(index=False))
    log(f"Saved {OUT_XLSX}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.run:
        run(a.workers)
    if a.eval:
        evaluate()
