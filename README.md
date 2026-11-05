# A pipeline based on stability selection for prioritizing metabolite mixtures in small-sample metabolomics

Code and data of the pipeline described in the manuscript of the same title (Meng *et al.*, submitted to
*Bioinformatics*), applied to root-exudate metabolomics of 44 wheat accessions phenotyped for biological
nitrification inhibition (BNI) against four ammonia-oxidizing strains.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Contents

1. [The pipeline](#the-pipeline)
2. [Repository](#repository)
3. [Installation](#installation)
4. [Quick test with the example data](#quick-test-with-the-example-data)
5. [Running the pipeline on the wheat data](#running-the-pipeline-on-the-wheat-data)
6. [Reproducing the figures and tables](#reproducing-the-figures-and-tables)
7. [Using your own data](#using-your-own-data)
8. [Data](#data)
9. [Citation and license](#citation-and-license)

## The pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `code/brs.py` | Balanced Relevance Score (BRS) of every metabolite for every phenotype: Pearson prescreen, Elastic Net, LightGBM, perceptron with feature attention and squared correlation |
| 2 | `code/stability_selection.py` | BRS on 50 complementary half-samples; selection probability π̂ (reproducible metabolites: π̂ ≥ 0.4) and single-accession leverage (≥ 0.7 flags dependence on one sample) |
| 3 | `code/nested_cv.py` | Strict nested cross-validation (5 folds × 2 repeats); preprocessing, labels, stability selection and mixture size are all derived inside the training folds. Also builds the final mixtures, whose size is chosen with the same rule on all accessions |
| 4 | `code/combination_test.py` | Pairs and triads only within each strain's pool of reproducible metabolites: complementarity and interaction gains, recurrence over the half-samples and under label permutation |
| – | `code/candidate_list.py` | Candidate list for experiments: reproducible metabolites, candidate pairs and exploratory candidates in three tiers of computational evidence |

`code/prepare_input.py` builds the input: half-minimum imputation of zeros, log1p and autoscaling.

The outputs are computational candidates and hypotheses for experiments. A selection probability measures
reproducible association in this cohort, not inhibitory activity, and an interaction gain is a statistical
association, not evidence of synergy.

## Repository

```
code/                         the pipeline (Steps 1-4 and the candidate list)
data/
  metabolite_names.xlsx       identifier -> compound name and chemical class
  preprocessing/
    filtered_notfound_removed.xlsx   raw intensities of the 413 annotated metabolites and AOI% of the four strains
    filtered_metabolites.xlsx        the same accessions and columns, autoscaled (AOI% autoscaled over all accessions)
    accessions.csv                   accession name and origin (Austria, India), in the row order of the data files
results/                      outputs of the pipeline for the manuscript
  stability_selection.xlsx    Step 2: selection probabilities, error bounds, leverage
  nested_cv.xlsx              Step 3: nested cross-validation (consensus list, comparators, mixtures, final mixtures)
  combination_test.xlsx       Step 4: pools and all pairs and triads with their gains
  candidate_list.xlsx         candidate list (Supplementary Table S16 of the manuscript)
  source_data/                benchmark values behind Fig. 2 and Table 1 (see below)
figures/                      scripts that redraw Figs 2-4 and Tables 1-3 of the manuscript
example/                      small synthetic dataset with planted signals, and a script that runs the pipeline on it
```

## Installation

The code was developed and tested with Python 3.12.2 on Windows 11 (CPU only). It uses only cross-platform
scientific packages but has not been tested on Linux or macOS. No GPU is needed.

```bash
git clone https://github.com/yuhangmeng/BRS-Framework.git
cd BRS-Framework
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the manuscript. On Linux, the default PyTorch wheel includes CUDA
libraries (several GB); the CPU build is sufficient and much smaller:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Installation takes a few minutes. The figures use the Arial font; where it is missing, matplotlib substitutes
another font and prints a warning.

## Quick test with the example data

`example/data` holds a small synthetic dataset in the layout of the wheat data: 40 samples, 200 metabolites and
four phenotypes in the columns the scripts expect (`Inhibition (%)-NF`, `-NV`, `-NM`, `-NU`). Signals were planted in
ten metabolites (`example/data/truth.csv`; generator: `example/make_example_data.py`):

| Phenotype | Planted signal |
|---|---|
| NF | M01, M02, M03 (single effects) |
| NV | M04, M05 (single effects) |
| NM | M06 (single effect); M07 and M08 (weak single effects and an interaction) |
| NU | M09, M10 (single effects) |

`example/run_example.py` copies `code/` and the example data into a separate folder (`example/run/`) and runs the
pipeline there, so the wheat data and results are not touched:

```bash
python example/run_example.py                 # Steps 1 and 2: about 5 min
python example/run_example.py --full          # all steps: about 70 min
```

Run times with 6 workers on a 12-core PC: Steps 1 and 2 (51 BRS runs) about 5 min. With `--full`, the nested
cross-validation (210 BRS runs) adds about 20 min and the combination test about 40 min. Use `--workers` to match
your CPU (each worker uses two threads).

On our machine, Step 2 selected the following metabolites (π̂ ≥ 0.4):

| Phenotype | Planted metabolites (π̂) | Other metabolites selected (π̂) |
|---|---|---|
| NF | M01 (1.00), M02 (0.98), M03 (0.62) | M12 (0.76) |
| NV | M04 (1.00); M05 not selected (0.16) | M136, M48, M25, M163 (0.40-0.48) |
| NM | M07 (0.64), M08 (0.60), M06 (0.40) | M100, M39 (0.54), M113 (0.44) |
| NU | M09 (1.00), M10 (0.64) | M54 (0.74), M61 (0.50), M117 (0.46) |

Nine of the ten planted metabolites were selected. With 40 samples and 190 noise metabolites, some noise
metabolites with chance correlations are selected as well; the selected metabolites are candidates for experiments,
not confirmed effects. In the full run, the combination test (85 pairs and triads within these pools) called one
reproducible interaction, the planted pair M07 + M08 (interaction gain in 80% of half-samples, 3.5% under label
permutation), and reproducible complementarity for M09 + M10 and for pairs and one triad of NF metabolites, two of
them with the noise metabolite M12. Results can differ slightly between platforms and library versions.

## Running the pipeline on the wheat data

```bash
python code/prepare_input.py                  # seconds
python code/stability_selection.py --run --summary --workers 6   # 51 BRS runs, about 15 min
python code/nested_cv.py --run --eval --workers 6                # 210 BRS runs, about 40 min
python code/combination_test.py               # about 40 min
python code/candidate_list.py                 # seconds
```

Run times are for a 12-core Windows PC. Runs are resumable (finished BRS runs are skipped), and all random components
use fixed seeds:
- 2026 for the half-sample partitions;
- 42 for BRS, the cross-validation and the combination test.

The files in `results/` were computed with these scripts from the BRS runs of the original analysis. The per-run
BRS outputs themselves are not distributed.

| File | Agreement with the manuscript |
|---|---|
| `stability_selection.xlsx` | same values as Supplementary Table S10, part A |
| `nested_cv.xlsx` | same values as the strict nested cross-validation in Supplementary Table S19 |
| `combination_test.xlsx`, `candidate_list.xlsx` | same values as the wheat part of Supplementary Table S15 and Parts A and B of Table S16, except one thing for the seven-metabolite NU pool: the permutation fractions differ by at most 0.013, and some combinations list their members in a different order |

The NU difference arises because metabolites with tied selection probabilities are now ordered by their
full-data BRS, so the label permutations are drawn in a different order. Observed gains and recurrences are
unchanged.

**Rerunning from scratch** with the versions in `requirements.txt`:
- The nested cross-validation runs of an outer fold reproduced the original BRS values exactly.
- The full-data and half-sample runs gave the same reproducible metabolites and the same top ten for every strain.
- Individual BRS values of lower-ranked metabolites can differ slightly between library versions or thread
  settings. In our rerun, the selection probability of six metabolites with π̂ ≤ 0.16 changed by 0.02.

## Reproducing the figures and tables

```bash
python figures/make_all.py                    # about 30 s; output in figures/output/
```

or one script at a time (for example `python figures/fig3_stability_map.py`). Figures are written as PDF and PNG,
tables as Markdown and CSV.

| Manuscript | Script | Inputs | Where the inputs come from |
|---|---|---|---|
| Fig. 1 | – | – | schematic of the pipeline, drawn by hand (no data) |
| Fig. 2 | `figures/fig2_combinations.py` | `results/source_data/fig2_source_data.xlsx` | benchmark analyses: direct enumeration of pairs and triads on the wheat half-samples, TreeSHAP, sample-size simulation (Supplementary Tables S10 part B, S17, S18) |
| Fig. 3 | `figures/fig3_stability_map.py` | `data/`, `results/stability_selection.xlsx` | Steps 1-2 |
| Fig. 4 | `figures/fig4_mixtures.py` | `data/`, `results/stability_selection.xlsx`, `results/nested_cv.xlsx` | Steps 2-3 |
| Table 1 | `figures/table1_benchmark.py` | `results/nested_cv.xlsx`, `results/source_data/table1_source_data.xlsx` | panel (a): Step 3 and the wheat benchmark; panels (b, c): 60 synthetic datasets (Supplementary Table S20) |
| Table 2 | `figures/table2_reproducible_metabolites.py` | `data/`, `results/stability_selection.xlsx`, `results/nested_cv.xlsx` | Steps 2-3 |
| Table 3 | `figures/table3_mixtures.py` | `results/stability_selection.xlsx`, `results/nested_cv.xlsx` | Steps 2-3 |

To reproduce Figs 3-4 and Tables 2-3 from the raw data, run the pipeline first (previous section) and then
`figures/make_all.py`; the scripts read whatever is in `results/`. The values behind Fig. 2 and Table 1 come from
benchmark analyses (several thousand BRS runs on synthetic data) whose code is not part of this repository; their
outputs are provided in `results/source_data/`, each workbook with a README sheet describing its columns.

With the files in this repository, the scripts reproduce the published figures and the values of the published
tables (checked against the manuscript files: Fig. 2 pixel-identical; Figs 3-4 differ only in anti-aliasing).

## Using your own data

The scripts expect the layout of `data/preprocessing`:
- `filtered_notfound_removed.xlsx`: one row per sample, one column per metabolite (raw intensities; zeros are
  treated as non-detects), followed by one column per phenotype;
- `filtered_metabolites.xlsx`: the same rows and columns, autoscaled (only its phenotype columns are used);
- `accessions.csv`: columns `row`, `accession`, `origin`, in the row order of the data files;
- `data/metabolite_names.xlsx`: columns `ID`, `Compound name`, `CHEMICAL CLASS`.

The phenotype columns are named `Inhibition (%)-<ID>` with the four IDs NF, NV, NM and NU, as in the example data
(`example/make_example_data.py` shows how to write the files). For a different number of phenotypes, change the
constants at the top of the scripts (`TARGET_COLS` in `code/brs.py`, `STRAINS` in Steps 2-4, `K_MAX` in
`code/nested_cv.py`). `code/candidate_list.py` writes the candidate list of the manuscript; its strain names and
wording are specific to the wheat study.

## Data

Root exudates of 44 wheat accessions (20 Austrian, 24 Indian) were measured by GC-TOF-MS and LC-Orbitrap
metabolomics. Ammonia-oxidation inhibition (AOI%) was measured against *Candidatus* Nitrosocosmicus franklandianus
(NF), *Nitrososphaera viennensis* (NV), *Nitrosospira multiformis* (NM) and *Nitrosomonas ureae* (NU). The data were
generated as described by Ghatak et al. (2025), *Plant Biotechnology Journal* 23: 4755–4772.

The code of an earlier version of the framework is kept under the tag
[`v1-2026-05`](https://github.com/yuhangmeng/BRS-Framework/tree/v1-2026-05).

## Citation and license

Meng *et al.* A pipeline based on stability selection for prioritizing metabolite mixtures in small-sample
metabolomics (submitted). Each release of this repository is archived at Zenodo; see `CITATION.cff` or the DOI
badge for the software citation. MIT license (see [LICENSE](LICENSE)).
