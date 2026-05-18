#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BRS-BNI: One-click reproduction script
=======================================
Runs the full analysis pipeline from raw data to final results.

Usage:
    python run_all.py              # Run everything
    python run_all.py --step 3     # Run from step 3 onward
    python run_all.py --only 7     # Run only step 7

Steps:
    1  Data preprocessing (remove unannotated features)
    2  Dimension reduction (correlation filtering, export analysis-ready matrix)
    3  Univariate BRS scoring
    4  Pairwise BRS scoring
    5  Triplet BRS scoring
    6  Generate synthetic benchmark dataset
    7  Comprehensive benchmark (synthetic data, 7 methods)
    8  Public dataset benchmark (ME/CFS + KOMP)
"""
import sys
import os
import time
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"

STEPS = [
    (1, "step1_preprocessing.py",            "Data preprocessing"),
    (2, "step2_dimension_reduction.py",       "Dimension reduction & correlation filtering"),
    (3, "step3_univariate_brs.py",            "Univariate BRS scoring"),
    (4, "step4_pairwise_brs.py",              "Pairwise BRS scoring"),
    (5, "step5_triplet_brs.py",               "Triplet BRS scoring"),
    (6, "step6_build_synthetic.py",           "Generate synthetic benchmark dataset"),
    (7, "step7_benchmark_comprehensive.py",   "Comprehensive benchmark (7 methods x 11 scenarios)"),
    (8, "step8_benchmark_public.py",          "Public dataset benchmark (ME/CFS + KOMP)"),
]


def run_step(num, script, desc, python_exe):
    """Run a single pipeline step."""
    script_path = SRC / script
    if not script_path.exists():
        print(f"  [SKIP] {script} not found")
        return False

    print(f"\n{'='*70}")
    print(f"  Step {num}: {desc}")
    print(f"  Script: src/{script}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(
        [python_exe, str(script_path)],
        cwd=str(ROOT),
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [FAILED] Step {num} exited with code {result.returncode} ({elapsed:.1f}s)")
        return False
    else:
        print(f"\n  [OK] Step {num} completed ({elapsed:.1f}s)")
        return True


def main():
    parser = argparse.ArgumentParser(description="BRS-BNI reproduction pipeline")
    parser.add_argument("--step", type=int, default=1,
                        help="Start from this step (default: 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this step")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter to use")
    args = parser.parse_args()

    # Create output directories
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "benchmark_comprehensive").mkdir(exist_ok=True)
    (ROOT / "results" / "benchmark_public").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    print("BRS-BNI Reproduction Pipeline")
    print(f"Python: {args.python}")
    print(f"Root:   {ROOT}")

    t_total = time.time()
    failed = []

    for num, script, desc in STEPS:
        if args.only is not None and num != args.only:
            continue
        if num < args.step:
            continue

        ok = run_step(num, script, desc, args.python)
        if not ok:
            failed.append(num)
            if args.only is None:
                print(f"\n  Pipeline stopped at step {num}. Fix the error and re-run with --step {num}")
                break

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    if failed:
        print(f"  Pipeline finished with errors in step(s): {failed}")
    else:
        print(f"  All steps completed successfully!")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*70}")

    # Summary of outputs
    print("\nKey output files:")
    outputs = [
        ("results/univariate_BRS.xlsx",                          "Univariate BRS rankings"),
        ("results/pairwise_BRS.xlsx",                            "Pairwise BRS rankings"),
        ("results/triple_BRS.xlsx",                              "Triplet BRS rankings"),
        ("results/benchmark_comprehensive/full_benchmark_table.csv", "Benchmark comparison table"),
        ("results/benchmark_comprehensive/full_benchmark_figure.png", "Benchmark comparison figure"),
        ("results/benchmark_public/public_benchmark_table.csv",  "Public dataset benchmark table"),
        ("results/benchmark_public/public_benchmark_figure.png", "Public dataset benchmark figure"),
    ]
    for path, desc in outputs:
        full = ROOT / path
        status = "OK" if full.exists() else "--"
        print(f"  [{status}] {path}")
        if full.exists():
            print(f"         {desc}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
