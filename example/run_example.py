# -*- coding: utf-8 -*-
"""
Run the pipeline on the small synthetic dataset in example/data, to check an installation.

The scripts in code/ read data/ and write results/ next to the code folder, so this script copies code/ and the
example data into a separate working folder (default example/run) and runs the steps there; the wheat data and
results of the repository are not touched.

    python example/run_example.py           Steps 1 and 2: 51 BRS runs (about 5 min with 6 workers)
    python example/run_example.py --full    all steps: also the nested cross-validation (210 BRS runs, about
                                            20 min), the combination test (about 40 min) and the candidate list
At the end, the reproducible metabolites (pi >= 0.4) are compared with the planted signals in example/data/truth.csv
and, with --full, the reproducible complementarity and interaction calls of the combination test are listed.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def step(cmd, cwd):
    print(f"\n$ python {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable] + cmd, cwd=cwd, check=True)
    print(f"  ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run the nested cross-validation, the combination test and the candidate list")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=HERE / "run", help="working folder (default example/run)")
    a = ap.parse_args()

    out = a.out.resolve()
    shutil.copytree(REPO / "code", out / "code", dirs_exist_ok=True)
    shutil.copytree(HERE / "data", out / "data", dirs_exist_ok=True)
    w = str(a.workers)
    step(["code/prepare_input.py"], out)
    step(["code/stability_selection.py", "--run", "--summary", "--workers", w], out)
    if a.full:
        step(["code/nested_cv.py", "--run", "--eval", "--workers", w], out)
        step(["code/combination_test.py"], out)
        step(["code/candidate_list.py"], out)

    pi = pd.read_excel(out / "results" / "stability_selection.xlsx", sheet_name="pi_single")
    truth = pd.read_csv(out / "data" / "truth.csv")
    print("\nReproducible metabolites (pi >= 0.4) and planted single effects:")
    for s in ["NF", "NV", "NM", "NU"]:
        found = pi.loc[pi[f"pi_{s}"] >= 0.4].sort_values(f"pi_{s}", ascending=False)
        got = ", ".join(f"{k} ({p:.2f})" for k, p in zip(found["key"], found[f"pi_{s}"])) or "none"
        planted = ", ".join(truth.loc[(truth.phenotype == s) & truth.effect.str.startswith("single"), "planted"]) or "none"
        print(f"  {s}: selected {got}; planted {planted}")
    if a.full:
        c = pd.read_excel(out / "results" / "combination_test.xlsx", sheet_name="combinations")
        print(f"\nCombination test: {len(c)} pairs and triads within the pools; reproducible calls "
              "(planted interaction: M07 x M08 for NM):")
        for kind in ("complementarity", "interaction"):
            hit = c[c[f"reproducible_{kind}"].astype(bool)]
            print(f"  {kind}: " + ("; ".join(f"{r.strain} {r.combination}" for r in hit.itertuples()) or "none"))
    print(f"\nAll outputs are in {out / 'results'}")


if __name__ == "__main__":
    main()
