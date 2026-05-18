from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ===================== Configuration =====================
FILE_PATH = str(_ROOT / "data" / "preprocessing" / "filtered_notfound_removed.xlsx")  # Master feature table
SHEET_NAME = "Sheet1"
DECIMAL_COMMA = True  # Keep True when decimals are stored like 0,0998

OUT_CORR_FILE = str(_ROOT / "data" / "preprocessing" / "corr_results.xlsx")  # Correlation and p-value table
OUT_FILTER_FILE = str(_ROOT / "data" / "preprocessing" / "filtered_metabolites.xlsx")  # Export all processed metabolites

# ===================== 1. Load data =====================
if DECIMAL_COMMA:
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, decimal=",")
else:
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

print("Original columns:")
print(list(df.columns))

# ===================== 2. Detect the four BNI columns =====================
all_cols = list(df.columns)
inhib_cols = [c for c in all_cols if "Inhibition" in str(c)]

print("\nColumns containing 'Inhibition':")
print(inhib_cols)

suffixes = ["NF", "NV", "NU", "NM"]
TARGET_COLS = []
for suf in suffixes:
    matches = [c for c in inhib_cols if suf in str(c)]
    if not matches:
        raise ValueError(f"Cannot find a BNI column containing '{suf}'.")
    TARGET_COLS.append(matches[0])

print("\nDetected BNI columns:")
print(TARGET_COLS)

# ===================== 3. Convert columns to numeric where possible =====================
df_num = df.copy()
for col in df_num.columns:
    if df_num[col].dtype == object:
        # Normalize decimal commas and strip spaces before parsing.
        df_num[col] = (
            df_num[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
    df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

# ===================== 3.1 Replace zeros with half of the minimum non-zero value =====================
for col in df_num.columns:
    series = df_num[col]
    if not pd.api.types.is_numeric_dtype(series):
        continue

    zero_mask = (series == 0) & series.notna()
    if not zero_mask.any():
        continue

    non_zero = series[(series != 0) & series.notna()]
    if len(non_zero) == 0:
        continue

    min_val = non_zero.min()
    fill_val = min_val / 2.0
    df_num.loc[zero_mask, col] = fill_val

print("\nZero values were replaced with half of each column's minimum non-zero value.")

# ===================== 3.2 Keep samples with complete BNI labels =====================
df_num = df_num.dropna(subset=TARGET_COLS).reset_index(drop=True)

# Treat every non-BNI column as a candidate metabolite feature.
feature_cols = [c for c in df_num.columns if c not in TARGET_COLS]
print(f"\nNumber of candidate features: {len(feature_cols)}")

# ===================== 3.3 Apply z-score normalization =====================
for col in df_num.columns:
    series = df_num[col]
    if pd.api.types.is_numeric_dtype(series):
        mean = series.mean(skipna=True)
        std = series.std(skipna=True)
        if std is not None and std > 0:
            df_num[col] = (series - mean) / std

# ===================== 4. Compute correlation and p-values =====================
records = []

for feat in feature_cols:
    if df_num[feat].isna().all():
        continue

    row = {"Metabolite": feat}
    for target in TARGET_COLS:
        x = df_num[feat]
        y = df_num[target]

        mask = x.notna() & y.notna()
        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) < 3:
            corr = np.nan
            pval = np.nan
        else:
            corr, pval = pearsonr(x_valid, y_valid)

        row[f"corr_{target}"] = corr
        row[f"p_{target}"] = pval

    records.append(row)

corr_df = pd.DataFrame(records)
corr_df.to_excel(OUT_CORR_FILE, index=False)
print(f"\nCorrelation and p-value table saved to: {OUT_CORR_FILE}")

# ===================== 5. Export all metabolites after preprocessing =====================
keep_metabolites = corr_df["Metabolite"].tolist()
print(f"\nTotal metabolites to export: {len(keep_metabolites)}")

cols_to_export = keep_metabolites + TARGET_COLS
filtered_df = df_num[cols_to_export]
filtered_df.to_excel(OUT_FILTER_FILE, index=False)
print(f"Processed metabolite table saved to: {OUT_FILTER_FILE}")
