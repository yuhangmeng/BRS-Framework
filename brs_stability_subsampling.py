# -*- coding: utf-8 -*-
"""
Complementary-pairs subsampling for BRS stability selection.

Each draw splits the 44 accessions into two disjoint halves (22 + 22) and runs the full,
unchanged BRS pipeline (Main Method.py -> Main Method(pair).py -> Main Method(triple)_1.py)
on each half. Selection frequencies, error bounds, multi-strain classification and metabolite
modules are computed afterwards by brs_stability_summary.py.

    python brs_stability_subsampling.py --pairs 25 --workers 4      # 50 half-samples
    python brs_stability_subsampling.py --pairs 1 --workers 2       # pilot / timing
    python brs_stability_subsampling.py --orders uni --out results_stability_uni   # single metabolites only

Finished subsamples are skipped, so the script can be stopped and restarted.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 1) Configuration
# ==========================================
ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "results_stability"
DATA_PATH = ROOT / "Data Preprocessing" / "filtered_metabolites.xlsx"
SCRIPTS = ["Main Method.py", "Main Method(pair).py", "Main Method(triple)_1.py"]
FINAL_FILE = "triple_BRS.xlsx"          # last output of the chain = subsample complete
SEED = 2026
THREADS_PER_WORKER = 2


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==========================================
# 2) Subsample definitions
# ==========================================
def make_halves(n_samples: int, n_pairs: int, seed: int = SEED):
    """n_pairs random splits into two disjoint halves -> 2 * n_pairs index sets."""
    rng = np.random.default_rng(seed)
    halves = []
    for _ in range(n_pairs):
        perm = rng.permutation(n_samples)
        k = n_samples // 2
        halves += [np.sort(perm[:k]), np.sort(perm[k:2 * k])]
    return halves


# ==========================================
# 3) One subsample = the three BRS scripts on a row subset
# ==========================================
def is_done(out: Path, final_file: str = FINAL_FILE) -> bool:
    """Finished = DONE marker, or a complete final workbook (its last sheet is written at the very end).
    The BRS scripts create their workbooks empty at the start of each stage, so existence alone is not enough."""
    if (out / "DONE").exists():
        return True
    f = out / final_file
    if not f.exists() or f.stat().st_size == 0:
        return False
    try:
        return "heatmap_values_wide" in pd.ExcelFile(f).sheet_names
    except Exception:
        return False


def run_subsample(b: int, idx: np.ndarray, out_root: Path = OUT_ROOT, scripts=tuple(SCRIPTS),
                  final_file: str = FINAL_FILE) -> str:
    out = out_root / f"sub_{b:03d}"
    out.mkdir(parents=True, exist_ok=True)
    if is_done(out, final_file):
        return f"sub_{b:03d} already done"
    np.savetxt(out / "samples.txt", idx, fmt="%d")
    env = dict(os.environ, BRS_OUT_DIR=str(out), BRS_SAMPLE_FILE=str(out / "samples.txt"),
               MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", LOKY_MAX_CPU_COUNT="1",
               OMP_NUM_THREADS=str(THREADS_PER_WORKER), MKL_NUM_THREADS=str(THREADS_PER_WORKER))
    t0 = time.time()
    with open(out / "run.log", "w", encoding="utf-8") as log:
        for s in scripts:
            r = subprocess.run([sys.executable, "-u", s], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                return f"sub_{b:03d} FAILED in {s} (see {out / 'run.log'})"
    (out / "DONE").write_text(f"{time.time() - t0:.0f} s\n", encoding="utf-8")
    return f"sub_{b:03d} done in {time.time() - t0:.0f}s"


# ==========================================
# 4) Main
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=25, help="number of complementary pairs (2x half-samples)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--orders", choices=["all", "uni"], default="all",
                    help="all: single metabolites, pairs and triads; uni: single metabolites only")
    ap.add_argument("--out", default=str(OUT_ROOT), help="output folder")
    args = ap.parse_args()
    # passed to the workers explicitly: spawned processes re-import this module with the defaults
    scripts, final_file = (("Main Method.py",), "univariate_BRS.xlsx") if args.orders == "uni" \
        else (tuple(SCRIPTS), FINAL_FILE)
    out_root = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out

    n = len(pd.read_excel(DATA_PATH, sheet_name=0).dropna(how="all"))
    halves = make_halves(n, args.pairs)
    out_root.mkdir(exist_ok=True)
    pd.DataFrame({"subsample": [f"sub_{b:03d}" for b in range(len(halves))],
                  "pair": [b // 2 for b in range(len(halves))],
                  "samples": [" ".join(map(str, h)) for h in halves]}).to_csv(out_root / "subsamples.csv", index=False)
    _log(f"{n} accessions, {len(halves)} half-samples of {len(halves[0])}, orders={args.orders}, "
         f"input={os.environ.get('BRS_INPUT', 'default')}, workers={args.workers}")

    t_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_subsample, b, h, out_root, scripts, final_file) for b, h in enumerate(halves)]
        for f in as_completed(futs):
            _log(f.result())
    _log(f"All done in {time.time() - t_all:.0f}s -> {out_root}")


if __name__ == "__main__":
    main()
