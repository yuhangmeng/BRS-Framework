# -*- coding: utf-8 -*-
"""C86: how does feature-ranking recovery degrade with class imbalance?

CLH's comment was not only a criticism: 'KOMP design imbalance: 6 vs 40??? This
is an opportunity to check different level of imbalance'. We subsample the
wildtype pool to 6, 12, 24 and 40 and rerun the benchmark at each ratio, with the
ground truth recomputed inside every resample so no full-data information leaks.
"""
import sys, importlib.util, warnings
import numpy as np, pandas as pd
import os
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("LEGACY_SCRATCH", ROOT / "legacy_scratch"))
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("bpd", ROOT / "benchmark_public_datasets.py")
bpd = importlib.util.module_from_spec(spec)
bpd.__name__ = "bpd"
spec.loader.exec_module(bpd)

# ---- load the full KOMP matrix once, without the module's ground-truth step --
fpath = ROOT / "public_datasets" / "komp" / "ST001154_metabolite_matrix.tsv"
df = pd.read_csv(fpath, sep="\t")
feat_cols = np.array(df.columns[2:].tolist())
feat_cols = feat_cols[~np.array([str(f).startswith("Standard_") for f in feat_cols])]

sub = df[(df["group"] == "Pmm2") | (df["group"] == "Null")].copy()
X_all = sub[feat_cols].values.astype(float)
y_all = np.where(sub["group"] == "Null", 0, 1).astype(float)
for j in range(X_all.shape[1]):
    col = X_all[:, j]
    m = np.isnan(col) | (col == 0)
    if m.any():
        col[m] = np.nanmedian(col[~m]) if (~m).any() else 1.0
        X_all[:, j] = col

ko_idx = np.where(y_all == 1)[0]
wt_idx = np.where(y_all == 0)[0]
print(f"KOMP full: {X_all.shape[0]} samples x {X_all.shape[1]} metabolites "
      f"({len(ko_idx)} Pmm2 KO, {len(wt_idx)} wildtype)\n")

RATIOS = [6, 12, 24, len(wt_idx)]
N_RESAMPLE = {6: 25, 12: 25, 24: 25, len(wt_idx): 1}   # full pool is deterministic
rows = []

for n_wt in RATIOS:
    for rep in range(N_RESAMPLE[n_wt]):
        rng = np.random.default_rng(1000 * n_wt + rep)
        sel_wt = wt_idx if n_wt >= len(wt_idx) else rng.choice(wt_idx, n_wt, replace=False)
        idx = np.concatenate([ko_idx, sel_wt])
        X, y = X_all[idx], y_all[idx]

        gt, _, _ = bpd.compute_ground_truth(X, y, feat_cols, fdr_threshold=0.1, top_n=30)
        if len(gt) == 0:
            continue
        for mname, mfn in bpd.METHODS.items():
            try:
                ranked, _scores = mfn(X, y, feat_cols)
            except Exception:
                continue
            rec = {K: bpd.recall_at_k(ranked, K, gt) for K in (10, 20, 50)}
            rows.append(dict(n_wt=n_wt, ratio=f"6:{n_wt}", rep=rep, method=mname,
                             n_ground_truth=len(gt),
                             recall_10=rec[10], recall_20=rec[20], recall_50=rec[50]))
    print(f"  ratio 6:{n_wt}  done ({N_RESAMPLE[n_wt]} resample(s))")

D = pd.DataFrame(rows)
agg = (D.groupby(["ratio", "n_wt", "method"])
         .agg(n_resamples=("rep", "count"),
              gt_mean=("n_ground_truth", "mean"),
              recall_20_mean=("recall_20", "mean"),
              recall_20_sd=("recall_20", "std"),
              recall_50_mean=("recall_50", "mean"))
         .reset_index().sort_values(["n_wt", "method"]))

pd.set_option("display.width", 220)
print("\n=== Recall@20 by imbalance ratio (mean over resamples) ===")
piv = agg.pivot_table(index="method", columns="ratio", values="recall_20_mean").round(3)
order = ["6:6", "6:12", "6:24", f"6:{len(wt_idx)}"]
piv = piv[[c for c in order if c in piv.columns]]
print(piv.to_string())

print("\n=== Recall@50 by imbalance ratio ===")
piv50 = agg.pivot_table(index="method", columns="ratio", values="recall_50_mean").round(3)
piv50 = piv50[[c for c in order if c in piv50.columns]]
print(piv50.to_string())

out = SCRATCH / "ST_komp_imbalance.xlsx"
with pd.ExcelWriter(out) as w:
    agg.to_excel(w, sheet_name="summary", index=False)
    piv.to_excel(w, sheet_name="recall20_matrix")
    piv50.to_excel(w, sheet_name="recall50_matrix")
    D.to_excel(w, sheet_name="per_resample", index=False)
print(f"\nwrote {out}")
