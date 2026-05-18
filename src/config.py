# -*- coding: utf-8 -*-
"""
Shared configuration: paths, targets, hyperparameters.
All paths are relative to the repository root.
"""
import os
from pathlib import Path

# Repository root (parent of src/)
ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ──────────────────────────────────────────────
RAW_EXCEL       = ROOT / "data" / "preprocessing" / "annotated_features_lc_gc_DL(4).xlsx"
FILTERED_EXCEL  = ROOT / "data" / "preprocessing" / "filtered_metabolites.xlsx"
NAME_MAP_PATH   = ROOT / "data" / "name.xlsx"

# Public benchmark datasets
MECFS_DATA      = ROOT / "data" / "public_datasets" / "mecfs"
KOMP_DATA       = ROOT / "data" / "public_datasets" / "komp" / "ST001154_metabolite_matrix.tsv"

# Synthetic dataset
SYNTHETIC_DIR   = ROOT / "data" / "synthetic"

# ── Output paths ────────────────────────────────────────────
RESULTS_DIR     = ROOT / "results"
FIGURES_DIR     = ROOT / "figures"
BENCH_DIR       = RESULTS_DIR / "benchmark_comprehensive"
BENCH_PUB_DIR   = RESULTS_DIR / "benchmark_public"

# ── Biological targets ──────────────────────────────────────
TARGET_COLS  = [
    "Inhibition (%)-NF",
    "Inhibition (%)-NV",
    "Inhibition (%)-NM",
    "Inhibition (%)-NU",
]
TARGET_SHORT = ["NF", "NV", "NM", "NU"]

# ── Hyperparameters ─────────────────────────────────────────
# Strength weights (three-model ensemble)
W_LGBM, W_ENET, W_MLP = 0.1, 0.5, 0.4

# BRS weights
ALPHA, BETA, GAMMA = 0.2, 0.2, 0.6

# Consistency gating threshold parameter
CONS_Q = 0.5

# Cross-validation
FOLDS = 5
SEED  = 42

# Name mapping columns
NAME_MAP_SHEET  = "Sheet1"
ID_COL_IN_MAP   = "ID"
NAME_COL_IN_MAP = "Compound name"


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [RESULTS_DIR, FIGURES_DIR, BENCH_DIR, BENCH_PUB_DIR]:
        d.mkdir(parents=True, exist_ok=True)
