"""
Benchmark BRS vs baselines on two public metabolomics datasets.
Strategy: use data-driven ground truth (statistically significant features
by t-test/Mann-Whitney with FDR correction) as the reference set, then
evaluate which method's top-K features overlap most with this reference.

Additionally, for KOMP, focus on a single well-characterized knockout.
"""
import sys, os, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Reuse existing benchmark functions
from step7_benchmark_comprehensive import (
    brs_univariate, lasso_rank, enet_rank, rf_rank,
    lgbm_shap_rank, permutation_imp_rank, mutual_info_rank,
    recall_at_k, COLORS,
)

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results" / "benchmark_public"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {
    "BRS": lambda X, y, fn: brs_univariate(X, y, fn),
    "LASSO": lasso_rank,
    "ElasticNet": enet_rank,
    "RandomForest": rf_rank,
    "LightGBM+SHAP": lgbm_shap_rank,
    "PermutationImp": permutation_imp_rank,
    "MutualInfo": mutual_info_rank,
}


def compute_ground_truth(X, y, feature_names, fdr_threshold=0.05, top_n=None):
    """
    Compute data-driven ground truth using Mann-Whitney U test with BH FDR.
    Returns set of significant feature names.
    If too few pass FDR, fall back to top_n by p-value.
    """
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    pvals = []
    effects = []  # absolute effect size (rank-biserial)
    for j in range(X.shape[1]):
        x0 = X[idx0, j]
        x1 = X[idx1, j]
        # Remove NaN
        x0 = x0[~np.isnan(x0)]
        x1 = x1[~np.isnan(x1)]
        if len(x0) < 3 or len(x1) < 3:
            pvals.append(1.0)
            effects.append(0.0)
            continue
        try:
            stat, p = stats.mannwhitneyu(x0, x1, alternative='two-sided')
            pvals.append(p)
            # rank-biserial correlation as effect size
            n0, n1 = len(x0), len(x1)
            r = 1 - (2 * stat) / (n0 * n1)
            effects.append(abs(r))
        except Exception:
            pvals.append(1.0)
            effects.append(0.0)

    pvals = np.array(pvals)
    effects = np.array(effects)

    # BH FDR correction
    n_feat = len(pvals)
    sorted_idx = np.argsort(pvals)
    fdr = np.ones(n_feat)
    for rank_i, orig_i in enumerate(sorted_idx):
        fdr[orig_i] = pvals[orig_i] * n_feat / (rank_i + 1)
    # Enforce monotonicity
    fdr_sorted = fdr[sorted_idx]
    for i in range(len(fdr_sorted) - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted
    fdr = np.minimum(fdr, 1.0)

    sig_mask = fdr < fdr_threshold
    n_sig = sig_mask.sum()

    if n_sig < 10 and top_n is not None:
        # Fall back to top_n by combined score (effect * -log10(p))
        score = effects * (-np.log10(pvals + 1e-300))
        top_idx = np.argsort(-score)[:top_n]
        sig_features = set(feature_names[i] for i in top_idx)
        print(f"  FDR<{fdr_threshold}: only {n_sig} features. "
              f"Using top-{top_n} by effect*significance instead.")
    else:
        sig_features = set(feature_names[i] for i in range(n_feat) if sig_mask[i])

    # Also compute pathway-level annotation for ME/CFS
    return sig_features, fdr, effects


# ====================== Dataset 1: ME/CFS ======================

def load_mecfs():
    """Load ME/CFS dataset, compute ground truth from the data."""
    fpath = BASE / "data" / "public_datasets" / "mecfs" / "Supplementary File 1.xlsx"
    df = pd.read_excel(fpath, sheet_name="OrigScale HD4")

    meta_cols = df.columns[:14].tolist()
    sample_cols = [c for c in df.columns if c not in meta_cols
                   and (str(c).startswith('C') or str(c).startswith('P'))]

    biochem = np.array(df["BIOCHEMICAL"].astype(str).values)
    super_pathway = df["SUPER_PATHWAY"].astype(str).values
    sub_pathway = df["SUB_PATHWAY"].astype(str).values
    data = df[sample_cols].values.T.astype(float)  # (n_samples, n_metabolites)
    labels = np.array([1 if str(c).startswith('P') else 0 for c in sample_cols])

    # Median imputation
    for j in range(data.shape[1]):
        col = data[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
            data[:, j] = col

    print(f"  ME/CFS: {data.shape[0]} samples x {data.shape[1]} metabolites")
    print(f"  Cases: {labels.sum()}, Controls: {(1-labels).sum()}")

    # Data-driven ground truth
    gt_set, fdr, effects = compute_ground_truth(
        data, labels, biochem, fdr_threshold=0.1, top_n=30)
    print(f"  Ground truth (FDR<0.1 or top-30): {len(gt_set)} features")

    # Show pathway distribution of ground truth
    gt_idx = [i for i, b in enumerate(biochem) if b in gt_set]
    gt_pathways = pd.Series(super_pathway[gt_idx]).value_counts()
    print(f"  Ground truth pathways:\n{gt_pathways.head(8).to_string()}")

    return data, labels, biochem, gt_set, (super_pathway, sub_pathway)


# ====================== Dataset 2: KOMP ======================

def load_komp():
    """
    Load KOMP: use Pmm2 knockout only (well-characterized mannose metabolism defect).
    """
    fpath = BASE / "data" / "public_datasets" / "komp" / "ST001154_metabolite_matrix.tsv"
    df = pd.read_csv(fpath, sep='\t')

    feat_cols = np.array(df.columns[2:].tolist())

    # Remove "Standard_XX" internal standards
    keep_mask = ~np.array([str(f).startswith("Standard_") for f in feat_cols])
    feat_cols_clean = feat_cols[keep_mask]

    # Pmm2 knockout vs wildtype
    ko_mask = df["group"] == "Pmm2"
    wt_mask = df["group"] == "Null"
    subset = df[ko_mask | wt_mask].copy()

    X = subset[feat_cols_clean].values.astype(float)
    y = np.where(subset["group"] == "Null", 0, 1).astype(float)

    # Impute NaN/zeros
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col) | (col == 0)
        if mask.any():
            med = np.nanmedian(col[~mask]) if (~mask).any() else 1.0
            col[mask] = med
            X[:, j] = col

    print(f"  KOMP (Pmm2 vs WT): {X.shape[0]} samples x {X.shape[1]} metabolites")
    print(f"  Pmm2 KO: {y.sum():.0f}, Wildtype: {(1-y).sum():.0f}")

    # Data-driven ground truth
    gt_set, fdr, effects = compute_ground_truth(
        X, y, feat_cols_clean, fdr_threshold=0.1, top_n=30)
    print(f"  Ground truth (FDR<0.1 or top-30): {len(gt_set)} features")

    return X, y, feat_cols_clean, gt_set


# ====================== Benchmark Runner ======================

def run_benchmark(dataset_name, X, y, feature_names, ground_truth):
    """Run all 7 methods, compute Recall@K against ground truth."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {dataset_name}")
    print(f"  n={X.shape[0]}, p={X.shape[1]}, ground truth={len(ground_truth)}")
    print(f"{'='*60}")

    rows = []
    for mname, mfn in METHODS.items():
        print(f"\n  [{mname}]", end=" ", flush=True)
        t0 = time.time()
        try:
            ranked, scores = mfn(X, y, feature_names)
            r5 = recall_at_k(ranked, 5, ground_truth)
            r10 = recall_at_k(ranked, 10, ground_truth)
            r20 = recall_at_k(ranked, 20, ground_truth)
            r50 = recall_at_k(ranked, 50, ground_truth)
            elapsed = time.time() - t0
            print(f"R@5={r5:.2f} R@10={r10:.2f} R@20={r20:.2f} R@50={r50:.2f} "
                  f"({elapsed:.1f}s)")
            print(f"    top-5: {ranked[:5]}")

            # Hits detail
            hits20 = [b for b in ranked[:20] if b in ground_truth]
            print(f"    Hits in top-20: {hits20}")

            rows.append(dict(
                dataset=dataset_name, method=mname,
                recall_5=r5, recall_10=r10, recall_20=r20, recall_50=r50,
                top5="; ".join(ranked[:5]), elapsed=elapsed,
            ))
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            rows.append(dict(
                dataset=dataset_name, method=mname,
                recall_5=np.nan, recall_10=np.nan, recall_20=np.nan,
                recall_50=np.nan, top5="ERROR", elapsed=0,
            ))

    return pd.DataFrame(rows)


def plot_comparison(df_all):
    """Create comparison figure."""
    datasets = df_all["dataset"].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7*n_ds, 5.5), dpi=150, squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        df = df_all[df_all["dataset"] == ds].copy()
        methods = df["method"].tolist()
        x = np.arange(len(methods))
        w = 0.2

        ax.bar(x - 1.5*w, df["recall_5"], w, label="Recall@5", color="#4C78A8")
        ax.bar(x - 0.5*w, df["recall_10"], w, label="Recall@10", color="#F58518")
        ax.bar(x + 0.5*w, df["recall_20"], w, label="Recall@20", color="#54A24B")
        ax.bar(x + 1.5*w, df["recall_50"], w, label="Recall@50", color="#E45756")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Recall (ground truth features)")
        ax.set_ylim(0, 1.15)
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(axis="y", alpha=0.3)

        for i, v in enumerate(df["recall_20"]):
            if not np.isnan(v):
                ax.text(i + 0.5*w, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

    plt.suptitle(
        "BRS vs. Baselines: Known Biomarker Recovery on Public Datasets\n"
        "(Ground truth = Mann-Whitney FDR<0.1)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "public_benchmark_figure.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "public_benchmark_figure.pdf", bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {OUT_DIR / 'public_benchmark_figure.png'}")


# ====================== Main ======================

def main():
    t_total = time.time()
    all_dfs = []

    # Dataset 1: ME/CFS
    print("\n" + "="*60)
    print("Loading ME/CFS dataset...")
    print("="*60)
    X1, y1, fn1, gt1, pathways1 = load_mecfs()
    df1 = run_benchmark("ME/CFS (n=52, p=768)", X1, y1, fn1, gt1)
    all_dfs.append(df1)

    # Dataset 2: KOMP (Pmm2 only)
    print("\n" + "="*60)
    print("Loading KOMP dataset (Pmm2 KO)...")
    print("="*60)
    X2, y2, fn2, gt2 = load_komp()
    df2 = run_benchmark("KOMP Pmm2 (n=46, p=1119)", X2, y2, fn2, gt2)
    all_dfs.append(df2)

    # Combine & save
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(OUT_DIR / "public_benchmark_table.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'public_benchmark_table.csv'}")

    plot_comparison(df_all)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    cols = ["dataset", "method", "recall_5", "recall_10", "recall_20", "recall_50"]
    print(df_all[cols].to_string(index=False))

    print(f"\nTotal elapsed: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
