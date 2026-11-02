# A pipeline based on stability selection for prioritizing metabolite mixtures in small-sample metabolomics

Code, input data and summary outputs for the manuscript of the same title (Meng *et al.*, submitted to *Bioinformatics*).

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What the pipeline does

Untargeted metabolomics studies often measure hundreds of metabolites in fewer than 50 samples. Rankings computed once
from such data change when a few samples are replaced, and rankings of metabolite combinations are the least stable.
The pipeline places stability selection between ranking and combination search:

1. **Ranking.** The Balanced Relevance Score (BRS) scores each metabolite for each phenotype. It combines a Pearson
   prescreen, an Elastic Net, LightGBM, a perceptron with feature attention and the squared correlation.
2. **Stability selection.** BRS is rerun on 50 complementary half-samples. Each metabolite receives a selection
   probability π̂ for every phenotype. Metabolites with π̂ ≥ 0.4 form the reproducible list (output A). A
   single-accession leverage flags candidates that depend on one sample.
3. **Mixture assembly.** Metabolites are added in order of π̂. The mixture size is chosen inside the training folds
   of a strict nested cross-validation (output B).
4. **Combination test.** Pairs and triads are scored only within the stability-selected pool, separating
   complementarity from interaction (output C).

BRS is one exchangeable scorer: Pearson correlation or the Elastic Net in its place give similar results. The outputs
are computational candidates and hypotheses for experiments. A selection probability measures reproducible
association in the cohort, not inhibitory activity, and an interaction gain is a statistical association, not
evidence of synergy.

## Repository layout

The scripts use paths relative to the repository root and run unchanged from there.

```
├── Main Method.py, Main Method(pair).py, Main Method(triple)_1.py   BRS for single metabolites, pairs, triads
├── brs_stability_*.py, stability_univariate_summary.py               Step 2: stability selection
├── nested_cv_signature.py, strict_nested_cv.py, incremental_signature.py, cv_checks.py,
│   check_response_scaling.py                                         nested cross-validation, Step 3
├── gated_combinations.py, gated_combination_auc.py, candidate_cards.py,
│   candidate_prioritization.py                                       Step 4 and the candidate lists
├── benchmark_*.py, sample_size_simulation.py, treeshap_stability.py  benchmarks and simulations
├── Fig*.py, Table*.py, build_supplementary_tables.py                 figures and tables of the manuscript
├── analysis_supplementary_legacy/   scripts behind Tables S2, S3 and S6 (see its README)
├── Data Preprocessing/      wheat input data and preprocessing scripts
├── public_datasets/komp/    Pmm2 knockout plasma metabolomics (Metabolomics Workbench ST001154)
├── mecfs_data/              ME/CFS plasma metabolomics (Germain et al. 2020, supplementary files)
├── supplementary/           summary workbooks of every analysis (sources of the Supplementary Tables)
├── results*/                summary workbooks read by the figure and table scripts
└── 论文/overleaf_v6/         Supplementary_Tables.xlsx; figures/ and tables/ are written here
```

Per-run outputs are not included: the BRS run of every half-sample and outer fold, which takes about 400 MB. The
scripts regenerate them (see below). The summary workbooks are included, so every figure and table script can be
run directly.

## Installation

```bash
git clone https://github.com/yuhangmeng/BRS-Framework.git
cd BRS-Framework
pip install -r requirements.txt     # tested with Python 3.12.2, CPU only
```

## Reproducing the analyses

All random components use fixed seeds:
- 42 for BRS, the outer cross-validation and the Elastic Net;
- 43 for the random comparator sets;
- 2026 for the half-sample partitions;
- 42 and 1–9 for the ten synthetic datasets.

Times are for a desktop CPU.

| Step | Command | Main outputs | Manuscript |
|---|---|---|---|
| Preprocessing | `python "Data Preprocessing/Data Preprocessing_1.py"`, `python "Data Preprocessing/Data Preprocessing_Dimension Reduction.py"`, `python "Data Preprocessing/build_log_input.py"` | `filtered_metabolites*.xlsx`, `filtered_notfound_removed.xlsx` (already included) | Methods 2.2 |
| BRS on all accessions | `python "Main Method.py"` (inputs and outputs set by `BRS_INPUT`, `BRS_OUT_DIR`, `BRS_SAMPLE_FILE`); pairs and triads with `Main Method(pair).py`, `Main Method(triple)_1.py` | `univariate_BRS.xlsx`, `pairwise_BRS.xlsx`, `triple_BRS.xlsx` | Step 1, Fig. S5 |
| Stability selection | `python brs_stability_subsampling.py --orders uni --out results_stability_uni` (about 15 min); `stability_univariate_summary.py` | `results_stability_uni/stability_summary_uni.xlsx` | Fig. 3, Table 2, Table S10 |
| Direct enumeration (all orders, raw input) | `brs_stability_subsampling.py` (about 18 h), `brs_stability_summary.py`, `brs_stability_diagnostics.py`, `treeshap_stability.py` | `results_stability_v2raw/*.xlsx`, `results_treeshap_stability/` | Fig. 2a, Tables S10, S17 |
| Strict nested cross-validation | `python strict_nested_cv.py --run --workers 6` (210 BRS runs, about 40 min), then `--eval` | `supplementary/ST_strict_nested_cv.xlsx` | Table 1, Section 3.3–3.4, Table S19 |
| Response-scaling check | `python check_response_scaling.py --run --eval` (about 40 min) | `results_nestedcv_strict/ST_strict_nested_cv_training_scaled_response.xlsx` | Methods 2.4, Supplementary Methods S4 |
| Earlier nested CV and mixtures | `nested_cv_signature.py` (about 2 h), `incremental_signature.py`, `cv_checks.py` | `results_nestedcv/`, `ST_incremental_signature.xlsx`, `ST_cv_checks.xlsx` | Tables S11, S13, Fig. S8 |
| Combination test (Step 4) | `python gated_combinations.py --data wheat` and `--data synthetic`; `gated_combination_auc.py` | `ST_gated_combinations.xlsx`, `results/gated_synthetic.xlsx`, `ST_gated_combination_auc.xlsx` | Section 3.5, Table S15 |
| Candidate lists | `candidate_cards.py`, `candidate_prioritization.py` | `ST_candidate_cards.xlsx`, `ST_candidate_prioritization.xlsx` | Table S16 |
| Synthetic benchmark | `benchmark_comprehensive.py`, `benchmark_interaction_baselines.py` | `results/benchmark_comprehensive/`, `results/benchmark_interaction/` | Section 3.1, Figs S1–S2, Table S8 |
| Public datasets | `benchmark_public_datasets.py` | `results/benchmark_public/` | Tables S5, S6 |
| Sample-size simulation | `sample_size_simulation.py`, `benchmark_scorers_sim.py`, `benchmark_pipeline_sim.py` | `results_sample_size/*.xlsx` | Fig. 2b, Table 1b, Tables S18, S20 |
| Pipeline benchmark (wheat) | `benchmark_pipeline_wheat.py` | `ST_pipeline_benchmark.xlsx` | Table 1a, Table S20 |
| Sensitivity and phenotypes | `prescreen_sensitivity.py`, `prescreen_sign.py`, `strain_phenotype_check.py` | `ST_prescreen_*.xlsx`, `ST_strain_phenotypes.xlsx` | Tables S3, S14 |
| Weight grid, KOMP imbalance | `analysis_supplementary_legacy/` (partly needs intermediate files of the original runs) | `ST_weight_grid36.xlsx`, `ST_komp_imbalance.xlsx` | Tables S2, S6 |
| Main figures | `Fig1_pipeline.py`, `Fig3_reproducibility_final.py`, `Fig_stability_map.py`, `Fig5_incremental_auc_final.py` | `论文/overleaf_v6/figures/` | Figs 1–4 |
| Main tables | `Table1_pipeline_benchmark.py`, `Table3_signature.py`, `Table3_mixtures.py` | `论文/overleaf_v6/tables/` | Tables 1–3 |
| Supplementary figures and tables | `FigS8_list_nestedcv.py`, `FigS9_gated_network.py`, `gated_combination_roc_fig.py`, `Fig2_heatmap_final.py`, `Fig4_single_vs_combo_final.py`, `build_candidate_sets.py`, `Table3_mixture_candidates.py`, `build_supplementary_tables.py` | figures; `Supplementary_Tables.xlsx` | Figs S5–S10, Tables S1–S20 |

Most scripts accept the environment variables or options described in their docstrings.

## Data

- **Wheat (44 accessions).** Root-exudate metabolomics (GC-TOF-MS and LC-Orbitrap; 413 annotated metabolites) and
  ammonia-oxidation inhibition (AOI%) against four strains: NF, NV, NM and NU. Data generated as described by
  Ghatak et al. (2025), *Plant Biotechnology Journal* 23: 4755–4772.
  - `annotated_features_lc_gc_DL(4).xlsx`: the annotated feature table.
  - `filtered_notfound_removed.xlsx`: the filtered intensities.
  - `filtered_metabolites.xlsx` and `filtered_metabolites_log.xlsx`: the BRS inputs (autoscaled and log-transformed).
- **Pmm2 knockout.** Metabolomics Workbench study ST001154 (Barupal et al. 2019).
- **ME/CFS.** Supplementary files of Germain et al. (2020), *Metabolites* 10: 34.

Single-compound assays of individual metabolites are not part of this study and are not included.

## Earlier version

The code of the earlier BRS framework (`src/step1–8`, May 2026) is kept under the tag
[`v1-2026-05`](https://github.com/yuhangmeng/BRS-Framework/tree/v1-2026-05).

## Citation

Meng *et al.* A pipeline based on stability selection for prioritizing metabolite mixtures in small-sample
metabolomics (submitted). The citation and archive DOI will be added on publication.

## License

MIT (see [LICENSE](LICENSE)).
