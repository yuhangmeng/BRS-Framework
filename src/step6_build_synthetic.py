from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
# -*- coding: utf-8 -*-
"""
Artificial dataset generator for validating that BRS can identify:
1) true univariate signals
2) true synergistic pairwise and triplet effects

Design summary
--------------
- Response y is continuous.
- True univariate informative features:
    X_lin_1, X_lin_2, X_lin_3, X_nonlin_1, X_nonlin_2
- True pairwise synergy:
    (X_lin_1, X_nonlin_1)
- True triplet synergy:
    (X_lin_2, X_lin_3, X_nonlin_2)
- Remaining features are pure noise.
- Pairwise and triplet ground truth are built from the same known univariate signals.

Outputs
-------
The script saves the generated dataset and metadata to a target directory:
- artificial_brs_synergy_dataset.xlsx
- dataset.csv
- X.npy
- y.npy
- ground_truth.json
- signal_components.csv
- quality_check.csv

This file is intended to be used as an upstream data generator before running
BRS or other ranking / screening algorithms.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class GroundTruth:
    true_linear_features: List[str]
    true_nonlinear_features: List[str]
    true_univariate_features: List[str]
    true_pairwise_synergy: List[Tuple[str, str]]
    true_triplet_synergy: List[Tuple[str, str, str]]
    noise_features: List[str]
    y_type: str
    total_features: int
    n_samples: int


@dataclass
class GenerationConfig:
    n_samples: int = 50
    n_noise: int = 195               # 5 signal + 195 noise = 200 features total
    seed: int = 42
    noise_std: float = 0.4          # lower -> easier recovery
    save_dir: str = "./artificial_brs_synergy_dataset"
    excel_name: str = "artificial_brs_synergy_dataset.xlsx"

    # Effect sizes
    coef_lin_1: float = 1.8
    coef_lin_2: float = -1.5
    coef_lin_3: float = 1.3
    coef_nonlin_1: float = 2.2
    coef_nonlin_2: float = 1.8
    coef_pair: float = 2.4
    coef_triplet: float = 2.8

    # Nonlinear shapes
    nonlin1_shift: float = 0.8       # for shifted quadratic effect
    nonlin2_threshold: float = 0.1   # for hinge effect


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_feature_names(n_noise: int) -> Tuple[List[str], List[str], List[str]]:
    linear_feats = ["X_lin_1", "X_lin_2", "X_lin_3"]
    nonlinear_feats = ["X_nonlin_1", "X_nonlin_2"]
    noise_feats = [f"X_noise_{i+1}" for i in range(n_noise)]
    return linear_feats, nonlinear_feats, noise_feats


def generate_dataset(config: GenerationConfig):
    rng = np.random.default_rng(config.seed)

    linear_feats, nonlinear_feats, noise_feats = build_feature_names(config.n_noise)
    true_single_feats = linear_feats + nonlinear_feats
    all_features = true_single_feats + noise_feats

    # ---------------------------------------------------------
    # 1) Generate raw features
    # ---------------------------------------------------------
    X_raw = rng.normal(loc=0.0, scale=1.0, size=(config.n_samples, len(all_features)))
    X_df = pd.DataFrame(X_raw, columns=all_features)

    # Standardize features to make coefficients easier to interpret
    scaler = StandardScaler()
    X_df.loc[:, all_features] = scaler.fit_transform(X_df[all_features])

    # Aliases for signal variables
    x1 = X_df["X_lin_1"].to_numpy()
    x2 = X_df["X_lin_2"].to_numpy()
    x3 = X_df["X_lin_3"].to_numpy()
    xn1 = X_df["X_nonlin_1"].to_numpy()
    xn2 = X_df["X_nonlin_2"].to_numpy()

    # ---------------------------------------------------------
    # 2) Main effects: recoverable univariate signals
    # ---------------------------------------------------------
    z_lin = (
        config.coef_lin_1 * x1
        + config.coef_lin_2 * x2
        + config.coef_lin_3 * x3
    )

    # Nonlinear main effects designed to be detectable but still nonlinear
    tmp1 = (xn1 - config.nonlin1_shift) ** 2
    z_nonlin_1 = config.coef_nonlin_1 * (tmp1 - np.mean(tmp1))

    tmp2 = np.maximum(0.0, xn2 - config.nonlin2_threshold)
    z_nonlin_2 = config.coef_nonlin_2 * (tmp2 - np.mean(tmp2))

    z_nonlin = z_nonlin_1 + z_nonlin_2

    # ---------------------------------------------------------
    # 3) True synergistic effects built from the same true singles
    # ---------------------------------------------------------
    # True pairwise synergy: (X_lin_1, X_nonlin_1)
    z_pair = config.coef_pair * (x1 * xn1)

    # True triplet synergy: (X_lin_2, X_lin_3, X_nonlin_2)
    z_triplet = config.coef_triplet * (x2 * x3 * xn2)

    # ---------------------------------------------------------
    # 4) Output noise and final response
    # ---------------------------------------------------------
    eps = rng.normal(loc=0.0, scale=config.noise_std, size=config.n_samples)
    y = z_lin + z_nonlin + z_pair + z_triplet + eps

    # ---------------------------------------------------------
    # 5) Package outputs
    # ---------------------------------------------------------
    component_df = pd.DataFrame({
        "z_lin": z_lin,
        "z_nonlin_1": z_nonlin_1,
        "z_nonlin_2": z_nonlin_2,
        "z_nonlin": z_nonlin,
        "z_pair": z_pair,
        "z_triplet": z_triplet,
        "eps": eps,
        "y": y,
    })

    full_df = X_df.copy()
    full_df["y"] = y

    truth = GroundTruth(
        true_linear_features=linear_feats,
        true_nonlinear_features=nonlinear_feats,
        true_univariate_features=true_single_feats,
        true_pairwise_synergy=[("X_lin_1", "X_nonlin_1")],
        true_triplet_synergy=[("X_lin_2", "X_lin_3", "X_nonlin_2")],
        noise_features=noise_feats,
        y_type="continuous",
        total_features=len(all_features),
        n_samples=config.n_samples,
    )

    quality_df = build_quality_check(full_df, component_df, truth)

    return full_df, component_df, truth, quality_df


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def build_quality_check(full_df: pd.DataFrame, component_df: pd.DataFrame, truth: GroundTruth) -> pd.DataFrame:
    y = full_df["y"].to_numpy()
    rows = []

    for feat in truth.true_univariate_features:
        x = full_df[feat].to_numpy()
        rows.append({
            "feature": feat,
            "type": "true_signal",
            "corr_x_y": safe_corr(x, y),
            "corr_x2_y": safe_corr(x**2, y),
        })

    for feat in truth.noise_features[:10]:
        x = full_df[feat].to_numpy()
        rows.append({
            "feature": feat,
            "type": "noise_example",
            "corr_x_y": safe_corr(x, y),
            "corr_x2_y": safe_corr(x**2, y),
        })

    # Explicit check for the true pair and true triplet terms
    pair_cols = truth.true_pairwise_synergy[0]
    pair_term = full_df[list(pair_cols)].prod(axis=1).to_numpy()
    rows.append({
        "feature": " * ".join(pair_cols),
        "type": "true_pair_term",
        "corr_x_y": safe_corr(pair_term, y),
        "corr_x2_y": np.nan,
    })

    triplet_cols = truth.true_triplet_synergy[0]
    triplet_term = full_df[list(triplet_cols)].prod(axis=1).to_numpy()
    rows.append({
        "feature": " * ".join(triplet_cols),
        "type": "true_triplet_term",
        "corr_x_y": safe_corr(triplet_term, y),
        "corr_x2_y": np.nan,
    })

    # Component-level correlations
    for col in ["z_lin", "z_nonlin", "z_pair", "z_triplet", "eps"]:
        rows.append({
            "feature": col,
            "type": "component",
            "corr_x_y": safe_corr(component_df[col].to_numpy(), y),
            "corr_x2_y": np.nan,
        })

    quality_df = pd.DataFrame(rows)
    return quality_df


def save_outputs(
    full_df: pd.DataFrame,
    component_df: pd.DataFrame,
    truth: GroundTruth,
    quality_df: pd.DataFrame,
    config: GenerationConfig,
) -> Dict[str, str]:
    ensure_dir(config.save_dir)

    excel_path = os.path.join(config.save_dir, config.excel_name)
    csv_path = os.path.join(config.save_dir, "dataset.csv")
    x_npy_path = os.path.join(config.save_dir, "X.npy")
    y_npy_path = os.path.join(config.save_dir, "y.npy")
    truth_json_path = os.path.join(config.save_dir, "ground_truth.json")
    components_csv_path = os.path.join(config.save_dir, "signal_components.csv")
    quality_csv_path = os.path.join(config.save_dir, "quality_check.csv")
    config_json_path = os.path.join(config.save_dir, "generation_config.json")

    feature_cols = [c for c in full_df.columns if c != "y"]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        full_df.to_excel(writer, sheet_name="dataset", index=False)
        component_df.to_excel(writer, sheet_name="signal_components", index=False)
        quality_df.to_excel(writer, sheet_name="quality_check", index=False)

        truth_rows = [[k, json.dumps(v, ensure_ascii=False)] for k, v in asdict(truth).items()]
        truth_df = pd.DataFrame(truth_rows, columns=["item", "value"])
        truth_df.to_excel(writer, sheet_name="ground_truth", index=False)

        cfg_rows = [[k, v] for k, v in asdict(config).items()]
        cfg_df = pd.DataFrame(cfg_rows, columns=["item", "value"])
        cfg_df.to_excel(writer, sheet_name="generation_config", index=False)

    full_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    component_df.to_csv(components_csv_path, index=False, encoding="utf-8-sig")
    quality_df.to_csv(quality_csv_path, index=False, encoding="utf-8-sig")

    np.save(x_npy_path, full_df[feature_cols].to_numpy())
    np.save(y_npy_path, full_df["y"].to_numpy())

    with open(truth_json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(truth), f, ensure_ascii=False, indent=2)

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    return {
        "excel_path": excel_path,
        "csv_path": csv_path,
        "x_npy_path": x_npy_path,
        "y_npy_path": y_npy_path,
        "truth_json_path": truth_json_path,
        "components_csv_path": components_csv_path,
        "quality_csv_path": quality_csv_path,
        "config_json_path": config_json_path,
    }


def main():
    # Save under the current working directory by default.
    # You can modify save_dir here before running.
    config = GenerationConfig(
        n_samples=50,
        n_noise=195,
        seed=42,
        noise_std=0.45,
        save_dir="./artificial_brs_synergy_dataset",
        excel_name="artificial_brs_synergy_dataset.xlsx",
    )

    full_df, component_df, truth, quality_df = generate_dataset(config)
    saved = save_outputs(full_df, component_df, truth, quality_df, config)

    print("Dataset generated successfully.")
    print(f"Samples: {truth.n_samples}")
    print(f"Total features: {truth.total_features}")
    print(f"True univariate features: {truth.true_univariate_features}")
    print(f"True pairwise synergy: {truth.true_pairwise_synergy}")
    print(f"True triplet synergy: {truth.true_triplet_synergy}")
    print("\nSaved files:")
    for k, v in saved.items():
        print(f"- {k}: {v}")

    print("\nTop quality-check rows:")
    print(quality_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
