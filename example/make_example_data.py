# -*- coding: utf-8 -*-
"""
Small synthetic dataset for testing the installation: 40 samples, 200 metabolites and four phenotypes, in the layout
of the wheat data (data/preprocessing). The phenotypes use the column names that the scripts expect
("Inhibition (%)-NF", ...), but every value is simulated.

Planted signals (on the log scale of the intensities, plus Gaussian noise):
    NF  M01, M02, M03            three single-metabolite effects
    NV  M04, M05                 two single-metabolite effects
    NM  M06; M07 x M08           one single effect and one pair with weak main effects and an interaction
    NU  M09, M10                 two single-metabolite effects
The other 190 metabolites are noise; M101-M200 contain 20% zeros (non-detects), as metabolomics data do.

Outputs (example/data): preprocessing/filtered_notfound_removed.xlsx (raw intensities and AOI%),
preprocessing/filtered_metabolites.xlsx (the same, autoscaled), preprocessing/accessions.csv,
metabolite_names.xlsx, truth.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data"
N, P, SEED = 40, 200, 1
CLASSES = ["Flavonoids", "Phenylpropanoids", "Amino acids and derivatives", "Carbohydrates", "Lipids"]

rng = np.random.default_rng(SEED)
ids = [f"M{j:02d}" for j in range(1, P + 1)]          # M01 ... M99, M100 ... M200
Z = rng.standard_normal((N, P))                                 # log-scale intensities, standardized
z = dict(zip(ids, Z.T))
noise = lambda: rng.standard_normal(N)

pheno = {
    "NF": 1.0 * z["M01"] + 0.8 * z["M02"] + 0.6 * z["M03"] + 0.6 * noise(),
    "NV": 0.9 * z["M04"] + 0.7 * z["M05"] + 0.6 * noise(),
    "NM": 0.8 * z["M06"] + 0.4 * z["M07"] + 0.4 * z["M08"] + 0.6 * z["M07"] * z["M08"] + 0.6 * noise(),
    "NU": 0.9 * z["M09"] + 0.7 * z["M10"] + 0.6 * noise(),
}
aoi = pd.DataFrame({f"Inhibition (%)-{s}": np.clip(20 + 8 * v, 0, None).round(2) for s, v in pheno.items()})

mu, sd = rng.uniform(8, 14, P), rng.uniform(0.4, 1.2, P)
raw = pd.DataFrame(np.exp(mu + sd * Z), columns=ids)
for j in range(P // 2, P):                                      # non-detects in the second half
    raw.iloc[np.argsort(Z[:, j])[: N // 5], j] = 0.0
raw = raw.round(1)

imp = raw.apply(lambda s: s.where(s != 0, s[s != 0].min() / 2))
scaled = (imp - imp.mean()) / imp.std()
aoi_scaled = (aoi - aoi.mean()) / aoi.std()

(OUT / "preprocessing").mkdir(parents=True, exist_ok=True)
pd.concat([raw, aoi], axis=1).to_excel(OUT / "preprocessing" / "filtered_notfound_removed.xlsx", sheet_name="Sheet1",
                                       index=False)
pd.concat([scaled.round(6), aoi_scaled.round(6)], axis=1).to_excel(
    OUT / "preprocessing" / "filtered_metabolites.xlsx", sheet_name="Sheet1", index=False)
pd.DataFrame({"row": range(N), "accession": [f"S{i + 1:02d}" for i in range(N)],
              "origin": ["group A"] * (N // 2) + ["group B"] * (N - N // 2)}).to_csv(
    OUT / "preprocessing" / "accessions.csv", index=False)
pd.DataFrame({"ID": ids, "Compound name": [f"Metabolite {j:02d}" for j in range(1, P + 1)],
              "CHEMICAL CLASS": [CLASSES[j % len(CLASSES)] for j in range(P)]}).to_excel(
    OUT / "metabolite_names.xlsx", sheet_name="Sheet1", index=False)
pd.DataFrame([("NF", "M01", "single", 1.0), ("NF", "M02", "single", 0.8), ("NF", "M03", "single", 0.6),
              ("NV", "M04", "single", 0.9), ("NV", "M05", "single", 0.7), ("NM", "M06", "single", 0.8),
              ("NM", "M07", "single (weak)", 0.4), ("NM", "M08", "single (weak)", 0.4),
              ("NM", "M07 x M08", "interaction", 0.6), ("NU", "M09", "single", 0.9), ("NU", "M10", "single", 0.7)],
             columns=["phenotype", "planted", "effect", "coefficient"]).to_csv(OUT / "truth.csv", index=False)
print(f"Saved {OUT} ({N} samples, {P} metabolites)")
