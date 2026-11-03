# A pipeline based on stability selection for prioritizing metabolite mixtures in small-sample metabolomics

Code and data of the pipeline described in the manuscript of the same title (Meng *et al.*, submitted to
*Bioinformatics*), applied to root-exudate metabolomics of 44 wheat accessions phenotyped for biological
nitrification inhibition (BNI) against four ammonia-oxidizing strains.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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
code/                       the six scripts above
data/
  metabolite_names.xlsx     identifier -> compound name and chemical class
  preprocessing/
    filtered_notfound_removed.xlsx   raw intensities of the 413 annotated metabolites and AOI% of the four strains
    filtered_metabolites.xlsx        the same accessions and columns, autoscaled (AOI% autoscaled over all accessions)
    accessions.csv                   accession name and origin (Austria, India), in the row order of the data files
results/                    outputs for the manuscript
  stability_selection.xlsx  selection probabilities, error bounds, leverage
  nested_cv.xlsx            nested cross-validation (consensus list, comparators, mixtures, final mixtures)
  combination_test.xlsx     pools and all pairs and triads with their gains
  candidate_list.xlsx       candidate list (Supplementary Table S16 of the manuscript)
```

## Running

```bash
pip install -r requirements.txt               # tested with Python 3.12.2, CPU only
python code/prepare_input.py                  # seconds
python code/stability_selection.py --run --summary --workers 6   # 51 BRS runs, about 15 min
python code/nested_cv.py --run --eval --workers 6                # 210 BRS runs, about 40 min
python code/combination_test.py               # about 40 min
python code/candidate_list.py                 # seconds
```

Runs are resumable, and all random components use fixed seeds:
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

## Data

Root exudates of 44 wheat accessions (20 Austrian, 24 Indian) were measured by GC-TOF-MS and LC-Orbitrap
metabolomics. Ammonia-oxidation inhibition (AOI%) was measured against *Candidatus* Nitrosocosmicus franklandianus
(NF), *Nitrososphaera viennensis* (NV), *Nitrosospira multiformis* (NM) and *Nitrosomonas ureae* (NU). The data were
generated as described by Ghatak et al. (2025), *Plant Biotechnology Journal* 23: 4755–4772.

The code of an earlier version of the framework is kept under the tag
[`v1-2026-05`](https://github.com/yuhangmeng/BRS-Framework/tree/v1-2026-05).

## Citation and license

Meng *et al.* A pipeline based on stability selection for prioritizing metabolite mixtures in small-sample
metabolomics (submitted). The citation and archive DOI will be added on publication. MIT license (see
[LICENSE](LICENSE)).
