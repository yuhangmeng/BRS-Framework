from pathlib import Path
import pandas as pd

# 1. Load the workbook.
FILE_PATH = str(Path(__file__).resolve().parent.parent / "data" / "preprocessing" / "annotated_features_lc_gc_DL(4).xlsx")
SHEET_NAME = "Sheet12"

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, decimal=",")

# 2. Keep columns that do not contain "Not found" markers.
cols_keep = [
    c for c in df.columns
    if "not found" not in str(c).lower() and "not found kegg" not in str(c).lower()
]

df_clean = df[cols_keep].copy()

# 3. Save the filtered workbook.
OUT_PATH = str(Path(__file__).resolve().parent.parent / "data" / "preprocessing" / "filtered_notfound_removed.xlsx")
df_clean.to_excel(OUT_PATH, index=False)

print("Original column count:", len(df.columns))
print("Filtered column count:", len(df_clean.columns))
print("Saved to:", OUT_PATH)
