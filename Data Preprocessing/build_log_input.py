# -*- coding: utf-8 -*-
"""
Build the log-scale BRS input: same layout as filtered_metabolites.xlsx (metabolite columns + four AOI% columns),
with each metabolite = autoscaled log1p of the raw intensity after replacing zeros by half the smallest non-zero
value. This is the transform used for composite scores, and it limits the leverage of single extreme accessions
on the Pearson correlations that BRS uses for prescreening and Validity.

Inputs : filtered_metabolites.xlsx (layout, AOI% columns), filtered_notfound_removed.xlsx (raw intensities)
Output : filtered_metabolites_log.xlsx (sheet "Sheet1")
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LAYOUT = HERE / "filtered_metabolites.xlsx"
RAW = HERE / "filtered_notfound_removed.xlsx"
OUT = HERE / "filtered_metabolites_log.xlsx"

lab = pd.read_excel(LAYOUT, sheet_name="Sheet1", decimal=",")
raw = pd.read_excel(RAW, sheet_name=0, decimal=",")
targets = [c for c in lab.columns if "Inhibition" in str(c)]
mets = [c for c in lab.columns if c not in targets]
assert len(lab) == len(raw), "row counts differ"

X = raw[mets].apply(pd.to_numeric, errors="coerce")
X = X.apply(lambda s: s.where(s != 0, s[s != 0].min() / 2))
L = np.log1p(X)
L = (L - L.mean()) / L.std()

# rows must be aligned: the existing input is the autoscaled raw matrix
old = lab[mets].apply(pd.to_numeric, errors="coerce")
zr = (X - X.mean()) / X.std()
agree = pd.Series({c: np.corrcoef(old[c], zr[c])[0, 1] for c in mets})
print(f"row alignment check (corr of existing input with autoscaled raw): median {agree.median():.4f}, "
      f"min {agree.min():.4f}")
assert agree.median() > 0.99, "rows of the two files do not seem to be aligned"

out = pd.concat([L, lab[targets]], axis=1)[list(lab.columns)]
out.to_excel(OUT, sheet_name="Sheet1", index=False)
print(f"Saved {OUT} ({out.shape[0]} x {out.shape[1]})")
