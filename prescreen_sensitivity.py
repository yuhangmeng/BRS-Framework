# -*- coding: utf-8 -*-
"""
Sensitivity of the single-metabolite BRS ranking to the prescreen settings (correlation threshold and cap).

Reruns Main Method.py on the full panel with r > 0.1, r > 0.3 and a cap of 50 metabolites, and compares each
per-strain top-10 with the reference run (r > 0.2, cap 100; results_v2_deterministic). Also reports the rank of
the 14 consensus-signature metabolites under each setting.

    python prescreen_sensitivity.py            # runs (cached) + summary
Output: supplementary/ST_prescreen_sensitivity.xlsx
"""
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

# ==========================================
# 1) Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
REF_DIR = ROOT / os.environ.get("BRS_OUT_DIR", "results_v2_deterministic")   # reference full-data run
OUT_ROOT = ROOT / "results_prescreen"
SETTINGS = {"r>0.1, cap 100": ("0.1", "100"), "r>0.3, cap 100": ("0.3", "100"), "r>0.2, cap 50": ("0.2", "50")}
STRAINS = ["NF", "NV", "NM", "NU"]
TOP_K = 10
SIG_XLSX = ROOT / "supplementary" / "ST_signature.xlsx"
OUT_XLSX = ROOT / "supplementary" / "ST_prescreen_sensitivity.xlsx"


def run_dir(label):
    return OUT_ROOT / label.replace(">", "gt").replace(", ", "_").replace(" ", "")


def run(label):
    out = run_dir(label)
    f = out / "univariate_BRS.xlsx"
    if f.exists() and f.stat().st_size > 0:
        return f"{label}: cached"
    out.mkdir(parents=True, exist_ok=True)
    r_min, cap = SETTINGS[label]
    env = dict(os.environ, BRS_OUT_DIR=str(out), BRS_R_MIN=r_min, BRS_MAX_FEATS=cap, MPLBACKEND="Agg",
               PYTHONIOENCODING="utf-8", LOKY_MAX_CPU_COUNT="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with open(out / "run.log", "w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, "-u", "Main Method.py"], cwd=ROOT, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    return f"{label}: {'FAILED' if rc else 'done'}"


def ranking(path, strain):
    t = pd.read_excel(path, sheet_name=f"Inhibition (pct)-{strain}_Uni_All").sort_values("BRS", ascending=False)
    return [str(f).strip() for f in t["Feature"]]


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=len(SETTINGS)) as ex:
        for msg in ex.map(run, SETTINGS):
            print(msg, flush=True)

    sig = pd.read_excel(SIG_XLSX, sheet_name="signature")
    rows, sig_rows = [], []
    for s in STRAINS:
        ref = ranking(REF_DIR / "univariate_BRS.xlsx", s)
        for label in SETTINGS:
            alt = ranking(run_dir(label) / "univariate_BRS.xlsx", s)
            rows.append({"setting": label, "strain": s, "n_scored_reference": len(ref), "n_scored": len(alt),
                         "top10_overlap_with_reference": len(set(ref[:TOP_K]) & set(alt[:TOP_K]))})
    for _, m in sig.iterrows():
        s = m["signature_strains"].split(", ")[0]
        rec = {"metabolite": m["metabolite"], "strain": s,
               "rank_reference": ranking(REF_DIR / "univariate_BRS.xlsx", s).index(str(m["key"]).strip()) + 1
               if str(m["key"]).strip() in ranking(REF_DIR / "univariate_BRS.xlsx", s) else None}
        for label in SETTINGS:
            alt = ranking(run_dir(label) / "univariate_BRS.xlsx", s)
            k = str(m["key"]).strip()
            rec[f"rank ({label})"] = alt.index(k) + 1 if k in alt else None
        sig_rows.append(rec)
    A, Bm = pd.DataFrame(rows), pd.DataFrame(sig_rows)
    with pd.ExcelWriter(OUT_XLSX) as w:
        pd.DataFrame([("reference", "r > 0.2, cap 100 (results_v2_deterministic)"),
                      ("top10_overlap_with_reference", "metabolites shared by the per-strain top-10 lists"),
                      ("signature ranks", "full-data rank of each signature metabolite for its signature strain")],
                     columns=["field", "description"]).to_excel(w, sheet_name="README", index=False)
        A.to_excel(w, sheet_name="top10_overlap", index=False)
        Bm.to_excel(w, sheet_name="signature_ranks", index=False)
    pd.set_option("display.width", 200)
    print(A.to_string(index=False))
    print(A.groupby("setting")["top10_overlap_with_reference"].agg(["min", "max", "mean"]))
    print(Bm.to_string(index=False))
