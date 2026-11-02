# -*- coding: utf-8 -*-
"""
Sensitivity check of the strict nested cross-validation (strict_nested_cv.py) to one step that is not repeated
inside the training folds: the AOI% response in Data Preprocessing/filtered_metabolites.xlsx is autoscaled once with
the mean and SD of all 44 accessions. Supplementary Methods S4.

    --run   repeat the 210 BRS runs (20 half-samples + full training set per outer fold) on a copy of each fold's
            input whose response columns are autoscaled with training-accession constants only
    --eval  (a) labels from the autoscaled response == labels from raw AOI% (outer folds and inner fits)
            (b) per-strain top ten of every run, compared with the stored runs
            (c) the complete strict evaluation with these runs (strict_nested_cv.evaluate, unrounded), compared with
                supplementary/ST_strict_nested_cv.xlsx

Output: results_nestedcv_strict_response_check/ (BRS runs, ST_strict_nested_cv_training_scaled_response.xlsx)
Run:    python check_response_scaling.py --run --workers 6    (about 40 min)
        python check_response_scaling.py --eval
"""
import argparse
import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_nestedcv_strict_response_check"
spec = importlib.util.spec_from_file_location("snc", ROOT / "strict_nested_cv.py")
snc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snc)
STORED_ROOT, STORED_XLSX = snc.OUT_ROOT, snc.OUT_XLSX
RAW_AOI = pd.read_excel(snc.RAW_PATH, sheet_name=0, decimal=",")[snc.TCOLS].apply(pd.to_numeric, errors="coerce")


def run_one(out, idx, inp):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "univariate_BRS.xlsx").exists():
        return
    np.savetxt(out / "samples.txt", np.sort(idx), fmt="%d")
    env = dict(os.environ, BRS_INPUT=str(inp), BRS_OUT_DIR=str(out), BRS_SAMPLE_FILE=str(out / "samples.txt"),
               MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", LOKY_MAX_CPU_COUNT="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with open(out / "run.log", "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-u", "Main Method.py"], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)


def run(workers):
    jobs = []
    for f, (tr, te) in enumerate(snc.OUTER):
        fd, od = STORED_ROOT / f"fold_{f:02d}", OUT / f"fold_{f:02d}"
        od.mkdir(parents=True, exist_ok=True)
        inp = od / "input.xlsx"
        if not inp.exists():
            d = pd.read_excel(fd / "input.xlsx", sheet_name="Sheet1", decimal=",")
            for c in snc.TCOLS:
                y = RAW_AOI[c]
                d[c] = (y - y.iloc[tr].mean()) / y.iloc[tr].std()
            d.to_excel(inp, sheet_name="Sheet1", index=False)
        jobs += [(od / "full", tr, inp)] + [(od / h.name, np.loadtxt(h / "samples.txt", dtype=int), inp)
                                            for h in sorted(fd.glob("half_*"))]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_one, *zip(*jobs)))


def evaluate():
    yraw = RAW_AOI[[[c for c in snc.TCOLS if c.endswith(s)][0] for s in snc.STRAINS]].set_axis(snc.STRAINS, axis=1)

    def lab_raw(idx, t):
        hi = yraw >= yraw.iloc[idx].quantile(snc.Q)
        return (hi.any(axis=1) if t == "overall" else hi[t]).astype(int).values
    same = all(np.array_equal(snc.labels(i, t), lab_raw(i, t))
               for f, (tr, _) in enumerate(snc.OUTER)
               for i in [tr] + [np.loadtxt(h / "samples.txt", dtype=int) for h in sorted((STORED_ROOT / f"fold_{f:02d}").glob("half_*"))]
               for t in snc.COLS)
    print("labels identical (outer folds and inner fits):", same)
    agree = total = 0
    for f in range(len(snc.OUTER)):
        for nm in ["full"] + [h.name for h in sorted((STORED_ROOT / f"fold_{f:02d}").glob("half_*"))]:
            A = snc.brs_ranked(STORED_ROOT / f"fold_{f:02d}" / nm / "univariate_BRS.xlsx")
            B = snc.brs_ranked(OUT / f"fold_{f:02d}" / nm / "univariate_BRS.xlsx")
            for s in snc.STRAINS:
                agree += set(A[s].index[:10]) == set(B[s].index[:10])
                total += 1
    print(f"per-strain top ten identical in {agree} of {total} runs x strains")
    snc.OUT_ROOT, snc.OUT_XLSX = OUT, OUT / "ST_strict_nested_cv_training_scaled_response.xlsx"
    snc.evaluate()
    for s, key in (("main", "method"), ("mixtures", "target")):
        a, b = pd.read_excel(STORED_XLSX, s).set_index(key), pd.read_excel(snc.OUT_XLSX, s).set_index(key)
        num = [c for c in a.columns if a[c].dtype.kind == "f" and c in b.columns]
        print(f"{s}: max |difference| over all statistics {np.nanmax(np.abs(a[num].values - b[num].values)):.4f}")
    fa, fb = pd.read_excel(STORED_XLSX, "folds"), pd.read_excel(snc.OUT_XLSX, "folds")
    print("k changed in", int((fa["k chosen in training"] != fb["k chosen in training"]).sum()), "of", len(fa),
          "fold x target combinations")


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
