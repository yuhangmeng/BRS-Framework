# -*- coding: utf-8 -*-
"""
Single-metabolite stability selection summary (reads <STAB_DIR>/sub_*/univariate_BRS.xlsx written by
brs_stability_subsampling.py --orders uni).

1) Selection probability pi(metabolite, strain) = fraction of half-samples in which the metabolite is in that
   strain's BRS top-10; error bound E[V] <= q^2 / ((2 pi_thr - 1) p) for pi_thr > 0.5.
2) Single-accession leverage. With complementary half-samples every accession is in exactly half of them, so a
   metabolite selected only because of one accession has pi close to 0.5. For each metabolite and strain:
       pi_in(i)  = selection probability over the half-samples that contain accession i
       pi_out(i) = selection probability over the half-samples that do not contain it
       leverage  = max_i [pi_in(i) - pi_out(i)]   (about 1 for a single-accession artefact, near 0 otherwise)
       pi_without_top = pi_out at the most influential accession (selection probability without it)
3) Comparison with the previous (raw-intensity) run for every metabolite with pi >= 0.3 in either run.

    BRS_OUT_DIR=results_v3_log python stability_univariate_summary.py
Output: <STAB_DIR>/stability_summary_uni.xlsx (sheet pi_single has the layout of stability_summary.xlsx)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 1) Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
STAB_DIR = ROOT / os.environ.get("BRS_STAB_DIR", "results_stability_uni")
FULL_RUN = ROOT / os.environ.get("BRS_OUT_DIR", "results_v3_log")
OLD_PI = ROOT / "results_stability_v2raw" / "stability_summary.xlsx"
ACCESSIONS = ROOT / "supplementary" / "ST_strain_phenotypes.xlsx"
NAME_MAP = ROOT / "name.xlsx"
OUT_XLSX = STAB_DIR / "stability_summary_uni.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
TOP_K = 10
PI_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
PI_SIG = 0.4
DISPLAY_FIX = {"L-TRYPTOPHAN": "L-Tryptophan", "Uric Acid": "Uric acid", "Ala-ala": "Ala-Ala",
               "Isoshaftoside": "Isoschaftoside", "4-METHYLCATECHOL": "4-Methylcatechol"}


def finished(d: Path) -> bool:
    f = d / "univariate_BRS.xlsx"
    if (d / "DONE").exists():
        return True
    if not f.exists() or f.stat().st_size == 0:
        return False
    try:
        return "heatmap_values_wide" in pd.ExcelFile(f).sheet_names
    except Exception:
        return False


# ==========================================
# 2) Per half-sample top-10 lists
# ==========================================
subs = sorted(d for d in STAB_DIR.glob("sub_*") if finished(d))
B = len(subs)
if B == 0:
    raise SystemExit(f"No finished half-samples in {STAB_DIR}")
n_acc = len(pd.read_excel(ACCESSIONS, sheet_name="accessions", index_col=0))
member = np.zeros((B, n_acc), dtype=bool)             # half-sample x accession
sel = {}                                              # (metabolite, strain) -> bool vector over half-samples
n_scored = []
for b, d in enumerate(subs):
    member[b, np.loadtxt(d / "samples.txt", dtype=int)] = True
    xls = pd.ExcelFile(d / "univariate_BRS.xlsx")
    for s in STRAINS:
        t = pd.read_excel(xls, f"Inhibition (pct)-{s}_Uni_All")
        t["Feature"] = t["Feature"].astype(str).str.strip()
        t = t.sort_values(["BRS", "Feature"], ascending=[False, True])
        n_scored.append(len(t))
        for m in t["Feature"].head(TOP_K):
            sel.setdefault((m, s), np.zeros(B, dtype=bool))[b] = True
print(f"{B} finished half-samples; median metabolites scored per strain {int(np.median(n_scored))}")

keys = sorted({m for m, _ in sel})
pi = pd.DataFrame({s: [sel.get((m, s), np.zeros(B, bool)).mean() for m in keys] for s in STRAINS}, index=keys)
pi.columns = [f"pi_{s}" for s in STRAINS]
pi["pi_max"] = pi.max(axis=1)

# ==========================================
# 3) Single-accession leverage
# ==========================================
acc_names = pd.read_excel(ACCESSIONS, sheet_name="accessions", index_col=0)["accession"].values
lev_rows = []
for (m, s), v in sel.items():
    if v.mean() < 0.2:
        continue
    pin = np.array([v[member[:, i]].mean() for i in range(n_acc)])
    pout = np.array([v[~member[:, i]].mean() for i in range(n_acc)])
    i = int(np.argmax(pin - pout))
    lev_rows.append({"key": m, "strain": s, "pi": v.mean(), "leverage": pin[i] - pout[i],
                     "top_accession_row": i, "top_accession": acc_names[i],
                     "pi_with_top": pin[i], "pi_without_top": pout[i]})
LEV = pd.DataFrame(lev_rows).sort_values(["strain", "pi"], ascending=[True, False])

# ==========================================
# 4) Names, full-data BRS, error bounds
# ==========================================
nm = pd.read_excel(NAME_MAP, sheet_name=0, dtype=str).set_index("ID")
names = {k.strip(): DISPLAY_FIX.get(str(v).strip(), str(v).strip()) for k, v in nm["Compound name"].items()}
classes = {k.strip(): str(v).strip() for k, v in nm["CHEMICAL CLASS"].items()}
dup = pd.Series(names).value_counts()


def display(k):
    b = names.get(k, k)
    return f"{b} [{k.split('_')[1]}]" if dup.get(b, 0) > 1 and "_" in k else b


pi["set"] = [display(k) for k in pi.index]
pi["chemical_class"] = [classes.get(k, "") for k in pi.index]
xls = pd.ExcelFile(FULL_RUN / "univariate_BRS.xlsx")
for s in STRAINS:
    t = pd.read_excel(xls, f"Inhibition (pct)-{s}_Uni_All").sort_values("BRS", ascending=False).reset_index(drop=True)
    t["Feature"] = t["Feature"].astype(str).str.strip()
    pi[f"BRS_full_{s}"] = pi.index.map(dict(zip(t["Feature"], t["BRS"])))
    pi[f"rank_full_{s}"] = pi.index.map(dict(zip(t["Feature"], t.index + 1)))
lev_best = LEV.sort_values("pi", ascending=False).drop_duplicates("key").set_index("key")
for s in STRAINS:
    d = LEV[LEV.strain == s].set_index("key")
    pi[f"leverage_{s}"] = pi.index.map(d["leverage"])
    pi[f"pi_without_top_{s}"] = pi.index.map(d["pi_without_top"])
pi = pi.sort_values("pi_max", ascending=False)
pi.insert(0, "key", pi.index)

p = int(np.median(n_scored))
EB = pd.DataFrame([{"pi_thr": t, "n_selected_any_strain": int((pi["pi_max"] >= t).sum()), "p_candidates": p,
                    "q_per_halfsample": TOP_K, "E[V]_bound_per_strain": TOP_K ** 2 / ((2 * t - 1) * p)}
                   for t in PI_THRESHOLDS])

# ==========================================
# 5) Comparison with the previous raw-intensity run
# ==========================================
old = pd.read_excel(OLD_PI, sheet_name="pi_single")
old["key"] = old["key"].astype(str).str.strip()
old = old.set_index("key")
cmp_keys = sorted(set(pi.index[pi.pi_max >= 0.3]) | set(old.index[old.pi_max >= 0.3]))
CMP = pd.DataFrame({"metabolite": [display(k) for k in cmp_keys],
                    **{f"pi_{s}_old_raw": [old[f"pi_{s}"].get(k, 0.0) for k in cmp_keys] for s in STRAINS},
                    **{f"pi_{s}_new_log": [pi[f"pi_{s}"].get(k, 0.0) for k in cmp_keys] for s in STRAINS},
                    "key": cmp_keys})
CMP["pi_max_old"] = CMP[[f"pi_{s}_old_raw" for s in STRAINS]].max(axis=1)
CMP["pi_max_new"] = CMP[[f"pi_{s}_new_log" for s in STRAINS]].max(axis=1)
CMP = CMP.sort_values("pi_max_new", ascending=False)

with pd.ExcelWriter(OUT_XLSX) as w:
    pd.DataFrame([("half-samples", f"{B} complementary half-samples (single-metabolite BRS only)"),
                  ("input", os.environ.get("BRS_INPUT", "see run log")),
                  ("pi", "fraction of half-samples in which the metabolite is in the strain's BRS top-10"),
                  ("leverage", "max over accessions of pi(with accession) - pi(without); ~1 = single-accession artefact"),
                  ("signature", f"metabolites with pi >= {PI_SIG} for at least one strain")],
                 columns=["field", "description"]).to_excel(w, sheet_name="README", index=False)
    pi.round(4).to_excel(w, sheet_name="pi_single", index=False)
    EB.round(3).to_excel(w, sheet_name="error_bounds", index=False)
    LEV.round(3).to_excel(w, sheet_name="leverage", index=False)
    CMP.round(3).to_excel(w, sheet_name="old_vs_new", index=False)

pd.set_option("display.width", 250)
show = ["set", "pi_NF", "pi_NV", "pi_NM", "pi_NU", "pi_max"] + [f"leverage_{s}" for s in STRAINS]
print(pi[pi.pi_max >= 0.3][show].round(2).to_string(index=False))
print(EB.round(2).to_string(index=False))
print(f"signature (pi >= {PI_SIG}): {(pi.pi_max >= PI_SIG).sum()} metabolites -> {OUT_XLSX}")
