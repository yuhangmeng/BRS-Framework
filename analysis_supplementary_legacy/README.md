# Scripts behind Supplementary Tables S2, S3 and S6

These scripts were written on 2026-10-05 in a temporary working directory and copied here so that the computation
of the tables stays documented. Paths are relative to the repository root. Several scripts summarize intermediate
files of the original runs that are not distributed. Set `LEGACY_SCRATCH` to the folder that holds them (default:
`legacy_scratch/`). The resulting summary workbooks are included in `supplementary/`.

| Script | Output (in `supplementary/`) | Manuscript table | Intermediate files needed |
|---|---|---|---|
| `komp_imbalance.py`, then `save_komp.py` | `ST_komp_imbalance.xlsx` | S6 | none: `komp_imbalance.py` writes them from `public_datasets/komp/` |
| `save_c56.py` | `ST_prescreen_correlation_sign.xlsx` | S3 | none (also produced by `prescreen_sign.py`) |
| `save_batch2.py` | `ST_weight_grid36.xlsx` (and the superseded `ST_cross_validated_auc.xlsx`) | S2 | `grid36/sensitivity_table.csv`, `ST_cross_validated_auc.xlsx` from the original runs |
| `pool_nondeterminism.py` | console demonstration that `list(set(...))[:cap]` made the old pair/triad pool depend on the hash seed | none | none (motivated the deterministic pool in `Main Method(pair).py` and `Main Method(triple)_1.py`) |
