# -*- coding: utf-8 -*-
"""
Strain-level phenotype structure of the wheat panel:
  (1) distribution of AOI% for each strain (are AOA phenotypes stronger / more variable than AOB?),
  (2) how much each strain's phenotype depends on single accessions (leverage),
  (3) germplasm origin (Austrian winter vs Indian spring) as a possible confounder of strain-specific signatures.

The origin of each of the 44 analysed accessions is recovered from Sheet3 of the raw annotation workbook by
matching the four AOI% values.
Output: supplementary/ST_strain_phenotypes.xlsx
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr, skew

ROOT = Path(__file__).resolve().parent
LAB = ROOT / "Data Preprocessing" / "filtered_metabolites.xlsx"
ANNOT = ROOT / "Data Preprocessing" / "annotated_features_lc_gc_DL(4).xlsx"
OUT = ROOT / "supplementary" / "ST_strain_phenotypes.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
GUILD = {"NF": "AOA", "NV": "AOA", "NM": "AOB", "NU": "AOB"}


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


lab = pd.read_excel(LAB, sheet_name=0)
aoi_z = pd.DataFrame({s: num(lab[[c for c in lab.columns if "Inhibition" in str(c) and str(c).endswith(s)][0]])
                      for s in STRAINS})            # the analysis file holds autoscaled AOI%
meta = pd.read_excel(ANNOT, sheet_name="Sheet3")
meta = meta[meta["ID"].notna()].reset_index(drop=True)
for s in STRAINS:
    meta[s] = num(meta[f"Inhibition (%)-{s}"])       # raw AOI (fraction inhibited)
meta_z = (meta[STRAINS] - meta[STRAINS].mean()) / meta[STRAINS].std()

# match the 44 analysed rows to Sheet3 by their four autoscaled AOI% values
idx = []
for i, r in aoi_z.iterrows():
    d = (meta_z - r.values).abs().sum(axis=1)
    j = d.idxmin()
    assert d[j] < 1e-4, f"row {i} not matched (min distance {d[j]})"
    idx.append(j)
assert len(set(idx)) == len(idx)
print("rows in the same order as Sheet3:", idx == list(range(len(idx))))
aoi = meta.loc[idx, STRAINS].reset_index(drop=True)
aoi["accession"] = meta.loc[idx, "Name"].values
aoi["origin"] = meta.loc[idx, "Geography"].astype(str).str.split("-").str[0].str.strip().values
print(aoi["origin"].value_counts().to_dict())

rows = []
for s in STRAINS:
    v = aoi[s].values
    top = np.sort(v)[::-1]
    # leverage: drop in the strain's variance when its single most extreme accession is removed
    z = (v - v.mean()) / v.std()
    rows.append({"strain": s, "guild": GUILD[s], "mean_AOI": v.mean(), "median_AOI": np.median(v), "SD_AOI": v.std(ddof=1),
                 "CV": v.std(ddof=1) / v.mean(), "min": v.min(), "max": v.max(), "skewness": skew(v),
                 "max_|z|": np.abs(z).max(), "share_of_variance_top1": (z.max() ** 2) / (z ** 2).sum(),
                 "n_AOI>=0.2": int((v >= 0.2).sum()), "n_AOI<=0.05": int((v <= 0.05).sum()),
                 "mean_Austria": v[aoi.origin == "Austria"].mean(), "mean_India": v[aoi.origin == "India"].mean(),
                 "MannWhitney_p_origin": mannwhitneyu(v[aoi.origin == "Austria"], v[aoi.origin == "India"]).pvalue})
P = pd.DataFrame(rows)
C = aoi[STRAINS].corr(method="spearman")
q75 = aoi[STRAINS].quantile(0.75)
hi = pd.DataFrame({s: (aoi[s] >= q75[s]).astype(int) for s in STRAINS})
H = pd.DataFrame({s: hi.groupby(aoi.origin)[s].sum() for s in STRAINS})
with pd.ExcelWriter(OUT) as w:
    P.round(3).to_excel(w, sheet_name="per_strain", index=False)
    C.round(3).to_excel(w, sheet_name="spearman_between_strains")
    H.to_excel(w, sheet_name="high_labels_by_origin")
    aoi.to_excel(w, sheet_name="accessions", index=True)
pd.set_option("display.width", 250)
print(P.round(3).to_string(index=False))
print(C.round(2))
print("high (>=75th pct) accessions by origin:\n", H)
