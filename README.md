# BRS-BNI

**An interpretable deep-learning framework identifies synergistic root-exudate metabolite combinations underlying biological nitrification inhibition in wheat**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Balanced Relevance Score (BRS) is an interpretable multi-model ensemble framework that prioritizes individual metabolites and small metabolite combinations (pairs and triads) associated with biological nitrification inhibition (BNI) from untargeted root-exudate metabolomics.

BRS integrates three complementary learners:

| Component | Role |
|-----------|------|
| **Attention-gated MLP** | Captures non-linear feature relevance via learned gate activations |
| **LightGBM + SHAP** | Provides model-agnostic gradient-boosted attributions |
| **Stability-selected Elastic Net** | Ensures sparse, reproducible linear-signal estimates |

The final score combines **Strength** (weighted model consensus), **Consistency** (cross-model agreement), and **Validity** (predictive gain) into a single ranking:

```
BRS = α · Strength + β · Consistency + γ · Validity
```

## Repository Structure

```
BRS-BNI/
├── run_all.py                  # One-click reproduction script
├── requirements.txt            # Python dependencies
├── src/
│   ├── config.py               # Shared paths and hyperparameters
│   ├── step1_preprocessing.py          # Remove unannotated features
│   ├── step2_dimension_reduction.py    # Correlation filtering
│   ├── step3_univariate_brs.py         # Single-metabolite BRS
│   ├── step4_pairwise_brs.py           # Pair BRS (interaction terms)
│   ├── step5_triplet_brs.py            # Triad BRS (3rd-order interactions)
│   ├── step6_build_synthetic.py        # Generate synthetic benchmark data
│   ├── step7_benchmark_comprehensive.py # 7-method benchmark (11 scenarios)
│   └── step8_benchmark_public.py       # Public dataset validation
├── data/
│   ├── preprocessing/          # Raw and preprocessed metabolomics
│   │   ├── annotated_features_lc_gc_DL(4).xlsx
│   │   └── filtered_metabolites.xlsx
│   ├── name.xlsx               # Metabolite ID → compound name mapping
│   ├── public_datasets/
│   │   ├── mecfs/              # ME/CFS metabolomics (Germain et al., 2020)
│   │   └── komp/               # KOMP Pmm2 knockout (Metabolomics Workbench ST001154)
│   └── synthetic/              # Pre-generated synthetic benchmark dataset
├── results/                    # Output tables and figures (auto-created)
└── figures/                    # Publication figures (auto-created)
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<username>/BRS-BNI.git
cd BRS-BNI
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_all.py
```

This executes all eight steps sequentially (approximately 30–60 minutes depending on hardware). To run a specific step:

```bash
python run_all.py --only 7    # Run only the comprehensive benchmark
python run_all.py --step 3    # Resume from step 3 onward
```

### 3. Key outputs

| File | Description |
|------|-------------|
| `results/univariate_BRS.xlsx` | Univariate BRS rankings for all metabolites × 4 conditions |
| `results/pairwise_BRS.xlsx` | Pairwise BRS rankings for top metabolite pairs |
| `results/triple_BRS.xlsx` | Triplet BRS rankings for top metabolite triads |
| `results/benchmark_comprehensive/full_benchmark_table.csv` | Benchmark: BRS vs. 6 baselines across 11 synthetic scenarios |
| `results/benchmark_public/public_benchmark_table.csv` | Benchmark: BRS vs. 6 baselines on ME/CFS and KOMP datasets |

## Pipeline Steps

### Step 1–2: Data preprocessing

Loads the raw annotated feature table (GC-TOF-MS + UHPLC-HRMS), removes features lacking compound annotations, handles decimal-comma formatting, imputes missing values, and exports the analysis-ready metabolite matrix with four BNI target columns (NF, NV, NM, NU).

### Step 3: Univariate BRS

For each metabolite, a three-model ensemble (attention-gated MLP, LightGBM, ElasticNetCV) is trained under k-fold cross-validation. Model-specific relevance scores are normalized, aggregated into Strength, Consistency, and Validity components, and combined into the final BRS.

### Step 4: Pairwise BRS

Top univariate candidates are combined into pairs. The design matrix `[x₁, x₂, x₁·x₂]` is z-standardized, then scored by the same three-model ensemble with bootstrap stability selection. Validity measures incremental R² gain over the best single-metabolite baseline.

### Step 5: Triplet BRS

Top candidates are combined into triads. Interaction terms use mean-centred products with tanh saturation (τ = 3.0) to prevent multiplicative blow-up. The design matrix includes all main effects, pairwise, and third-order interactions (7 columns). Validity is the incremental R² beyond the best embedded pair.

### Step 6–7: Synthetic benchmark

A controlled dataset (n = 50, p = 200) with 5 planted univariate signals, 1 synergistic pair, and 1 synergistic triplet is generated. BRS is compared against LASSO, Elastic Net, Random Forest, LightGBM+SHAP, permutation importance, and mutual information across 11 scenarios varying noise level, sample size, dimensionality, and signal strength. Ablation and weight-sensitivity analyses are also performed.

### Step 8: Public dataset validation

BRS is validated on two independent public metabolomics datasets:
- **ME/CFS** (n = 52, p = 768; Germain et al., 2020): circulating metabolomics of chronic fatigue syndrome
- **KOMP Pmm2** (n = 46, p = 1,092; Metabolomics Workbench ST001154): Pmm2-knockout mouse plasma metabolomics

Ground truth is defined as features reaching statistical significance (Mann–Whitney U, BH-FDR < 0.10).

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α, β, γ | 0.2, 0.2, 0.6 | BRS component weights (Strength, Consistency, Validity) |
| w_LGBM, w_ENet, w_MLP | 0.1, 0.5, 0.4 | Strength sub-model weights |
| τ (consistency) | 0.5 | Gating threshold = median + τ · IQR |
| τ (saturation) | 3.0 | tanh saturation scale for interaction terms |
| Bootstrap iterations | 40 (pairs), 30 (triads) | Elastic Net stability selection |
| CV folds | 3 (univariate), 5 (pairs/triads) | Cross-validation scheme |

## Data Availability

- **Wheat BNI metabolomics**: included in `data/preprocessing/` (44 accessions × 4 BNI conditions)
- **ME/CFS dataset**: Germain et al. (2020), *Metabolites* 10(1):34. Supplementary data included in `data/public_datasets/mecfs/`
- **KOMP dataset**: Metabolomics Workbench Study ST001154. Data included in `data/public_datasets/komp/`

## System Requirements

- Python 3.10 or later
- PyTorch 2.0+ (CPU sufficient; GPU optional)
- 8 GB RAM recommended
- Tested on Windows 11 and Ubuntu 22.04

## Citation

If you use BRS in your research, please cite:

```
[Citation will be added upon publication]
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
