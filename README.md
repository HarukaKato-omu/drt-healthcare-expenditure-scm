# Impact of Demand Responsive Transportation introduction on healthcare expenditures

Replication code for:

> Impact of Demand Responsive Transportation introduction on healthcare
> expenditures: A natural experiment in Ikuno and Hirano wards, Osaka City.
> *Manuscript under review.*

This repository contains the Python pipeline used to estimate the association
between the March 2021 introduction of Demand Responsive Transportation (DRT)
in the Ikuno and Hirano wards of Osaka City and per-patient administrative
ward-level healthcare expenditure (ward-level HE), using the synthetic control
method (`SyntheticControlMethods`) with the 20 non-exposure wards of Osaka
City as donors over a ward × month panel (April 2017–March 2024; 48 baseline
and 36 follow-up months). The pipeline reproduces all model-based quantities
in the article: donor weights, covariate balance and predictor importance,
pre- and post-intervention RMSPE, the monthly and cumulative gaps between the
observed and synthetic series, and the leave-one-out re-estimations over the
donor pool, together with the descriptive exposure/non-exposure series.

## Repository structure

```
├── scm_drt_pipeline.py     # Full estimation pipeline (v1.0.0)
├── data/
│   └── schema.csv          # Variable definitions and sources (no raw data)
├── .gitignore              # Prevents accidental data/output commits
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

Running the pipeline creates `output/` (estimation CSVs and diagnostics) and
`report/` (figures). These are not tracked in the repository.

## Data availability

The ward × month panel (`data/he_panel.csv`) is **not** redistributed here,
and — unlike panels built entirely from official statistics — it **cannot be
reconstructed from public sources**: the ward-level HE series were provided
to the authors by a third-party data holder under a data-use agreement that
does not permit redistribution. Requests for access should be directed to the
data provider identified in the article's Data availability statement. The
covariates (age-group population shares, total population, land area, and the
COVID-19 state-of-emergency indicator) can be reconstructed from the public
sources listed in `data/schema.csv`, which documents every variable (name,
type, unit, definition, source, and access conditions) so that researchers
who obtain an equivalent HE extract can run the pipeline without
modification.

## Requirements

- Python ≥ 3.9 (the article's analyses were run on Google Colab; syntax
  verified on Python 3.12) [author check: Colab Python version]
- Required packages: `pandas`, `numpy`, `SyntheticControlMethods`
  (Engelbrektson, 2020)
- Optional package: `matplotlib` (figures)

```
pip install -r requirements.txt
```

## How to run

1. Obtain an equivalent HE extract, arrange it as documented in
   `data/schema.csv`, and save it as `data/he_panel.csv`.
2. `python scm_drt_pipeline.py` — loads and validates the panel, writes the
   descriptive series, then prints the checkpoint status.
3. `python scm_drt_pipeline.py --run-all --force` — full pipeline (main model
   and 20 leave-one-out re-estimations: 200 optimizer initializations each,
   `pen = "auto"`, `random_seed = 0`).
   For a quick smoke test (does **not** reproduce the published values):
   `SCM_N_OPTIM=10 python scm_drt_pipeline.py --run-all --force`.

**Runtime.** Steps S1–S3 take roughly [X] minutes; S4 (leave-one-out
re-estimations) takes roughly [Y] minutes [author to add]. `--run-all` without
`--force` skips steps whose outputs already exist, so an interrupted run can
be resumed.

## Output ↔ article mapping

| Output | Article element |
|---|---|
| `output/figure3_group_series.csv`, `report/figure3_group_series` | Figure 3 |
| `output/descriptive_group_means.csv` | Section 3.2 (baseline and follow-up means by group) |
| `output/table2_donor_weights.csv` | Table 2 |
| `output/table3_covariates_by_ward.csv` | Table 3 |
| `output/table4_covariate_balance.csv` | Table 4 (pre-intervention means, WMAPE, importance) |
| `output/table5_rmspe.csv` | Table 5 |
| `output/scm_estimates_summary.csv` | Section 3.3 (36-month cumulative gap and its monthly mean; pre-period mean gap) |
| `output/figure4_series.csv`, `report/figure4_synthetic_control` | Figure 4 |
| `output/table6_leave_one_out.csv` | Table 6 |
| `output/figure5_loo_series.csv`, `report/figure5_leave_one_out` | Figure 5 |

Gaps are observed − synthetic in JPY per patient per month, so negative
values indicate lower observed expenditure; the cumulative gap is accumulated
from April 2021 only, as in Figure 4(c). Table 1 (sample characteristics) is
tabulated directly from aggregate counts supplied by the data provider and is
not produced by the pipeline.

## Reproducibility notes

- Every fit — the main model and each leave-one-out re-estimation — uses the
  same specification: `n_optim = 200` optimizer initializations seeded with
  `random_seed = 0`, `pen = "auto"`, and the seven predictors listed in
  `data/schema.csv` (pre-intervention means of the outcome and six
  covariates). Re-runs with the same package versions reproduce the reported
  values at the reported rounding.
- `SyntheticControlMethods` uses every column other than the unit and time
  identifiers as a predictor and reshapes the outcome in dataset order; the
  pipeline therefore restricts the panel to the documented columns and sorts
  it by (`ward`, `month_index`) before every fit.
- Balance diagnostics are reported in the original unit of each predictor
  (WMAPE is the package's Weighted Mean Absolute Pairwise Error, not a
  percentage) and RMSPE in JPY per patient per month. No placebo or
  permutation inference is computed, in line with Section 2.3 of the article.
- `output/run_settings.json` records the Python and package versions and the
  estimation settings of each run.

## Citation

If you use this code, please cite the paper above (citation metadata for
this repository is provided in `CITATION.cff`).

## License

MIT — see `LICENSE`. The license applies to the code only and does not
extend to the HE data, which remain subject to the data-use agreement with
the data provider.

## Contact

Haruka Kato, Associate Professor, Osaka Metropolitan University
