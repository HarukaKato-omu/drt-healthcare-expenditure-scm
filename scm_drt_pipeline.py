#!/usr/bin/env python3
# ==============================================================================
# DRT introduction and ward-level healthcare expenditure - synthetic control
# estimation pipeline (v1.0.0)
#
# Replication code for:
#   Kato, H. et al. "Impact of Demand Responsive Transportation Introduction on
#   Healthcare Expenditures: A Natural Experiment in Ikuno and Hirano Wards,
#   Osaka City" (manuscript under review, Transportation Research
#   Interdisciplinary Perspectives).            # [author check: author list]
#
# NOTE ON THIS PUBLIC VERSION
#   v1.0.0 consolidates the two Google Colab notebooks that produced the
#   manuscript results (main SCM estimation; leave-one-out robustness) into
#   one script. Colab-specific I/O (pip magics, google.colab.files) was
#   removed, the input file and column names were generalized (see
#   data/schema.csv), and console messages were translated into English.
#   The estimation settings are unchanged: SyntheticControlMethods
#   (Engelbrektson, 2020), pen = "auto", n_optim = 200, random_seed = 0.
#   [author check: confirm these settings match the notebooks that produced
#   the submitted results]
#
# WHAT THIS SCRIPT DOES
#   S1  Panel preparation: loads data/he_panel.csv (ward x month panel,
#       April 2017 - March 2024, 84 months), validates columns, units and
#       periods, and maps the integer month index to calendar months.
#   S2  Descriptives: exposure vs. non-exposure monthly means (Fig. 3 data;
#       Section 3.2 means) and per-ward covariate means (Table 3 data).
#   S3  Baseline SCM: synthetic control for the exposure unit from all
#       donor wards; donor weights (Table 2), covariate balance and predictor
#       importance (Table 4), pre/post RMSPE (Table 5), observed/synthetic
#       series with monthly and cumulative gaps (Fig. 4, Section 3.3).
#   S4  Leave-one-out robustness: re-estimates the synthetic control once
#       per donor ward, excluding that ward (Table 6, Fig. 5).
#   S5  Reporting: CSV tables in output/, figures (PDF + PNG) in report/.
#
# INPUT
#   data/he_panel.csv   (ward x month panel; not redistributed)
#   The outcome series was provided by a third-party data holder under a
#   data-use agreement and is not redistributed; see data/schema.csv in this
#   repository for variable definitions and original data sources.
#
# USAGE
#   python scm_drt_pipeline.py                     # S1-S2 only; prints status
#   python scm_drt_pipeline.py --run-all           # full pipeline (n_optim = 200)
#   SCM_N_OPTIM=10 python scm_drt_pipeline.py --run-all --force   # quick test
#   In Python: from scm_drt_pipeline import run_all; run_all(force=True)
#
# DESIGN NOTES (aligned with the Methods section of the paper)
#   * Exposure unit: Ikuno and Hirano wards aggregated into one unit
#     (ward-level intention-to-treat exposure). Donor pool: the 20 wards of
#     Osaka City without DRT during the analysis period (Kita and Fukushima
#     wards excluded).
#   * Outcome: per-patient ward-level healthcare expenditure (JPY per patient
#     per month), assigned to wards by provider location.
#   * Time index: integer month index with 2000 = April 2017. Baseline =
#     April 2017 - March 2021 (48 months, index 2000-2047); treatment period
#     starts at index 2048 = April 2021; follow-up = April 2021 - March 2024
#     (36 months, index 2048-2083).
#   * Predictors: pre-intervention means of the outcome and of six
#     covariates (total, youth, working-age and elderly population; land
#     area; state-of-emergency indicator). SyntheticControlMethods treats
#     every column other than the id and time variables as a predictor, so
#     the panel is restricted to exactly these columns before fitting.
#   * Gap = observed - synthetic (negative = lower observed expenditure).
#     The cumulative gap is accumulated from the treatment period only.
#   * No placebo/permutation inference is computed (Section 2.3 of the
#     paper); RMSPE and WMAPE are reported as descriptive diagnostics.
#   * SyntheticControlMethods reshapes outcomes in dataset order, so the
#     panel is sorted by (ward, month_index) before every fit.
#
# KEY VARIABLES (see data/schema.csv)
#   ward                    unit identifier (exposure unit = "Ikuno_Hirano")
#   month_index             integer month index (2000 = April 2017)
#   he_per_patient          per-patient ward-level HE (JPY/patient/month)
#   population              total registered population (persons)
#   youth_population        population share aged 0-19 (%)
#   working_age_population  population share aged 20-64 (%)
#   elderly_population      population share aged 65+ (%)
#   land_area_km2           ward land area (km2)
#   lockdown                state-of-emergency month indicator (1/0)
#   [author check: youth/working-age/elderly are shares (%) as in Table 3/4]
# ==============================================================================

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# S0 Configuration
# ==============================================================================
VERSION = "1.0.0"

DIR_DATA = "data"                       # folder with input CSVs
INPUT_PANEL = "he_panel.csv"         # place the panel at data/he_panel.csv
DIR_OUT = "output"
DIR_REP = "report"

ID_VAR = "ward"
TIME_VAR = "month_index"
OUTCOME_VAR = "he_per_patient"
TREATED_UNIT = "Ikuno_Hirano"
COVARIATES = ["population", "youth_population", "working_age_population",
              "elderly_population", "land_area_km2", "lockdown"]
DONORS_EXPECTED = 20                    # 24 wards - 2 exposure - 2 excluded (Kita, Fukushima)

BASE_INDEX = 2000                       # month_index of the first baseline month
BASE_MONTH = "2017-04"                  # = month_index 2000
TREATMENT_MONTH = "2021-04"             # start of DRT operation (April 2021)
END_MONTH = "2024-03"                   # last follow-up month

N_OPTIM = int(os.environ.get("SCM_N_OPTIM", 200))   # optimizer initializations
PEN = "auto"                                         # pairwise-difference penalty
RANDOM_SEED = 0

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
os.makedirs(DIR_OUT, exist_ok=True)
os.makedirs(DIR_REP, exist_ok=True)


def month_index_to_period(idx):
    """Integer month index -> pandas monthly Period (2000 -> 2017-04)."""
    return pd.Period(BASE_MONTH, "M") + (int(idx) - BASE_INDEX)


def period_to_month_index(p):
    """Calendar month (e.g. '2021-04') -> integer month index."""
    return BASE_INDEX + (pd.Period(p, "M") - pd.Period(BASE_MONTH, "M")).n


TREATMENT_PERIOD = period_to_month_index(TREATMENT_MONTH)   # 2048
END_INDEX = period_to_month_index(END_MONTH)                # 2083
assert TREATMENT_PERIOD == 2048 and END_INDEX == 2083, "month index convention changed"


def _msg(*parts):
    print(" ".join(str(p) for p in parts), flush=True)


# ==============================================================================
# S1 Panel preparation
# ==============================================================================
def prepare_panel(path: str | None = None) -> pd.DataFrame:
    """Load, validate and sort the ward x month panel.

    Returns a DataFrame restricted to [ID_VAR, TIME_VAR, OUTCOME_VAR] +
    COVARIATES, sorted by (ward, month_index), with each ward observed in
    every month from BASE_MONTH to END_MONTH.
    """
    path = path or os.path.join(DIR_DATA, INPUT_PANEL)
    if not os.path.exists(path):
        sys.exit(f"Input panel not found: {path}\n"
                 f"Reconstruct it following {DIR_DATA}/schema.csv and save it as {path}.")
    raw = pd.read_csv(path, encoding="utf-8")

    need = [ID_VAR, TIME_VAR, OUTCOME_VAR] + COVARIATES
    missing = [c for c in need if c not in raw.columns]
    if missing:
        sys.exit(f"Missing columns in {path}: {missing}\n"
                 f"Expected columns (see data/schema.csv): {need}")
    extra = [c for c in raw.columns if c not in need]
    if extra:
        _msg(f"NOTE: dropping columns not used as predictors: {extra}")

    d = raw[need].copy()
    d[TIME_VAR] = d[TIME_VAR].astype(int)
    d = d.sort_values([ID_VAR, TIME_VAR]).reset_index(drop=True)

    units = list(d[ID_VAR].unique())
    if TREATED_UNIT not in units:
        sys.exit(f"Treated unit '{TREATED_UNIT}' not found. Units: {units}")
    donors = [u for u in units if u != TREATED_UNIT]
    if len(donors) != DONORS_EXPECTED:
        _msg(f"WARNING: {len(donors)} donor units found (paper: {DONORS_EXPECTED}).")

    periods = np.arange(BASE_INDEX, END_INDEX + 1)
    bad = [u for u in units
           if not np.array_equal(d.loc[d[ID_VAR] == u, TIME_VAR].to_numpy(), periods)]
    if bad:
        sys.exit(f"Units not observed in every month {BASE_INDEX}-{END_INDEX}: {bad}")
    if d[need[2:]].isna().any().any():
        sys.exit("Missing values in outcome/covariates; the panel must be complete.")

    n_pre = int((periods < TREATMENT_PERIOD).sum())
    n_post = int((periods >= TREATMENT_PERIOD).sum())
    _msg(f"panel: {len(units)} units (treated + {len(donors)} donors) | "
         f"{len(periods)} months ({month_index_to_period(BASE_INDEX)} to "
         f"{month_index_to_period(END_INDEX)}) | pre = {n_pre}, post = {n_post}")
    return d


def donor_units(d: pd.DataFrame) -> list:
    return [u for u in d[ID_VAR].unique() if u != TREATED_UNIT]


# ==============================================================================
# S2 Descriptives (Fig. 3 data, Section 3.2 means, Table 3 data)
# ==============================================================================
def descriptives(d: pd.DataFrame) -> pd.DataFrame:
    """Exposure vs. non-exposure monthly means and per-ward covariate means."""
    tr = d[d[ID_VAR] == TREATED_UNIT].set_index(TIME_VAR)[OUTCOME_VAR]
    dn = (d[d[ID_VAR] != TREATED_UNIT]
          .groupby(TIME_VAR)[OUTCOME_VAR].mean())       # unweighted mean of donor wards
    g = pd.DataFrame({TIME_VAR: tr.index,
                      "month": [str(month_index_to_period(i)) for i in tr.index],
                      "exposure": tr.to_numpy(),
                      "non_exposure": dn.reindex(tr.index).to_numpy()})
    g["period"] = np.where(g[TIME_VAR] < TREATMENT_PERIOD, "baseline", "follow_up")
    g.to_csv(os.path.join(DIR_OUT, "figure3_group_series.csv"), index=False)

    m = (g.groupby("period")[["exposure", "non_exposure"]].mean()
           .reindex(["baseline", "follow_up"]).round(0))
    m.loc["change"] = m.loc["follow_up"] - m.loc["baseline"]
    m.to_csv(os.path.join(DIR_OUT, "descriptive_group_means.csv"))

    rows = []
    for window, mask in [("pre", d[TIME_VAR] < TREATMENT_PERIOD),
                         ("all", np.ones(len(d), dtype=bool))]:
        cm = d[mask].groupby(ID_VAR)[COVARIATES].mean().round(2)
        cm.insert(0, "window", window)
        rows.append(cm)
    tab3 = pd.concat(rows)
    tab3["group"] = np.where(tab3.index == TREATED_UNIT, "exposure", "non_exposure")
    tab3.to_csv(os.path.join(DIR_OUT, "table3_covariates_by_ward.csv"))
    # [author check: the averaging window used for Table 3 in the manuscript]

    if HAS_MPL:
        _fig3(g)
    _msg("descriptives done: output/figure3_group_series.csv, "
         "descriptive_group_means.csv, table3_covariates_by_ward.csv")
    return g


# ==============================================================================
# S3 Baseline SCM (Tables 2, 4, 5; Fig. 4)
# ==============================================================================
def fit_synth(data: pd.DataFrame, n_optim: int = N_OPTIM) -> dict:
    """Fit one synthetic control and return series and diagnostics."""
    from SyntheticControlMethods import Synth

    data = data.sort_values([ID_VAR, TIME_VAR]).reset_index(drop=True)
    sc = Synth(data, OUTCOME_VAR, ID_VAR, TIME_VAR, TREATMENT_PERIOD, TREATED_UNIT,
               n_optim=n_optim, pen=PEN, random_seed=RANDOM_SEED)
    od = sc.original_data

    sub = data[data[ID_VAR] == TREATED_UNIT]
    yr = sub[TIME_VAR].to_numpy()
    actual = sub[OUTCOME_VAR].to_numpy(dtype=float)
    synth = np.asarray(od.synth_outcome, dtype=float).ravel()
    assert len(synth) == len(actual) == len(yr), "length mismatch"

    gap = actual - synth                         # observed - synthetic (negative = lower)
    pre, post = yr < TREATMENT_PERIOD, yr >= TREATMENT_PERIOD
    cum = np.cumsum(np.where(post, gap, 0.0))    # accumulated from treatment start
    weights = od.weight_df["Weight"].sort_values(ascending=False)

    return dict(yr=yr, actual=actual, synth=synth, gap=gap, cum_gap=cum,
                pre_rmspe=float(np.sqrt(np.mean(gap[pre] ** 2))),
                post_rmspe=float(np.sqrt(np.mean(gap[post] ** 2))),
                pre_mean_gap=float(gap[pre].mean()),
                cum_post=float(gap[post].sum()),        # 36-month cumulative gap
                mean_post=float(gap[post].mean()),      # = cum_post / n_post
                n_post=int(post.sum()),
                weights=weights, comparison_df=od.comparison_df.copy(),
                rmspe_df=od.rmspe_df.copy())


def run_baseline(d: pd.DataFrame, n_optim: int = N_OPTIM) -> dict:
    _msg(f"S3 baseline SCM: {len(donor_units(d))} donors, n_optim = {n_optim}, pen = {PEN}")
    base = fit_synth(d, n_optim)

    # Table 2 - donor weights
    (base["weights"].rename_axis("donor").reset_index()
        .round(3).to_csv(os.path.join(DIR_OUT, "table2_donor_weights.csv"), index=False))

    # Table 4 - covariate balance (pre-intervention means) and predictor importance
    cmp = base["comparison_df"].rename(
        columns={TREATED_UNIT: "exposure_unit",
                 f"Synthetic {TREATED_UNIT}": "synthetic_exposure_unit"})
    cmp.rename_axis("predictor").to_csv(os.path.join(DIR_OUT, "table4_covariate_balance.csv"))

    # Table 5 - RMSPE
    base["rmspe_df"].round(2).to_csv(os.path.join(DIR_OUT, "table5_rmspe.csv"), index=False)

    # Observed / synthetic series with gaps (Fig. 4 data)
    s = pd.DataFrame({TIME_VAR: base["yr"],
                      "month": [str(month_index_to_period(i)) for i in base["yr"]],
                      "period": np.where(base["yr"] < TREATMENT_PERIOD, "baseline", "follow_up"),
                      "observed": base["actual"], "synthetic": base["synth"],
                      "gap": base["gap"], "cumulative_gap": base["cum_gap"]})
    s.round(2).to_csv(os.path.join(DIR_OUT, "figure4_series.csv"), index=False)

    summary = pd.Series({
        "pre_rmspe": base["pre_rmspe"], "post_rmspe": base["post_rmspe"],
        "post_pre_ratio": base["post_rmspe"] / base["pre_rmspe"],
        "pre_mean_gap": base["pre_mean_gap"],
        "cumulative_gap_36": base["cum_post"], "monthly_mean_gap": base["mean_post"],
        "n_post": base["n_post"], "n_optim": n_optim, "pen": PEN, "random_seed": RANDOM_SEED})
    summary.rename_axis("quantity").rename("value").to_csv(
        os.path.join(DIR_OUT, "scm_estimates_summary.csv"))

    if HAS_MPL:
        _fig4(base)

    _msg("=== BASELINE (all donors) ===")
    _msg(f"Pre-RMSPE            : {base['pre_rmspe']:.1f}")
    _msg(f"Post-RMSPE           : {base['post_rmspe']:.1f}  "
         f"(post/pre = {base['post_rmspe'] / base['pre_rmspe']:.2f})")
    _msg(f"pre-period mean gap  : {base['pre_mean_gap']:.1f}  (observed - synthetic)")
    _msg(f"cumulative gap (36)  : {base['cum_post']:.1f}  (negative = lower observed expenditure)")
    _msg(f"monthly mean gap     : {base['mean_post']:.1f}  (= cumulative / {base['n_post']})")
    _msg(f"donor weights        : {base['weights'].round(3).to_dict()}")
    return base


# ==============================================================================
# S4 Leave-one-out robustness (Table 6; Fig. 5)
# ==============================================================================
def run_leave_one_out(d: pd.DataFrame, base: dict, n_optim: int = N_OPTIM) -> pd.DataFrame:
    donors = donor_units(d)
    _msg(f"S4 leave-one-out: {len(donors)} re-estimations, n_optim = {n_optim}")
    rows, curves = [], {}
    for i, dn in enumerate(donors, 1):
        try:
            r = fit_synth(d[d[ID_VAR] != dn].copy(), n_optim)
            curves[dn] = r
            rows.append(dict(excluded_donor=dn,
                             base_weight=float(base["weights"].get(dn, 0.0)),
                             pre_rmspe=round(r["pre_rmspe"], 1),
                             cumulative_gap_36=round(r["cum_post"], 1),
                             monthly_mean_gap=round(r["mean_post"], 1)))
        except Exception as e:                     # keep going; report the failure
            rows.append(dict(excluded_donor=dn,
                             base_weight=float(base["weights"].get(dn, 0.0)),
                             pre_rmspe=np.nan, cumulative_gap_36=np.nan, monthly_mean_gap=np.nan))
            _msg(f"  [skip {dn}] {e}")
        _msg(f"  ... {i}/{len(donors)} done ({dn})")

    loo = (pd.DataFrame(rows).sort_values("base_weight", ascending=False)
             .reset_index(drop=True))
    loo.to_csv(os.path.join(DIR_OUT, "table6_leave_one_out.csv"), index=False)

    long = pd.concat([pd.DataFrame({"excluded_donor": dn, TIME_VAR: r["yr"],
                                    "synthetic": r["synth"], "gap": r["gap"],
                                    "cumulative_gap": r["cum_gap"]})
                      for dn, r in curves.items()], ignore_index=True)
    long.round(2).to_csv(os.path.join(DIR_OUT, "figure5_loo_series.csv"), index=False)

    v = loo["cumulative_gap_36"].dropna().to_numpy()
    _msg("=== LEAVE-ONE-OUT (drop each donor, refit) ===")
    _msg(loo.to_string(index=False))
    _msg("--- robustness summary: cumulative 36-month gap ---")
    _msg(f"baseline                  : {base['cum_post']:.0f}")
    _msg(f"LOO range                 : [{v.min():.0f}, {v.max():.0f}]  "
         f"(x{v.min() / base['cum_post']:.2f} to x{v.max() / base['cum_post']:.2f} of baseline)")
    _msg(f"all LOO same sign as base : {bool(np.all(np.sign(v) == np.sign(base['cum_post'])))}")
    _msg(f"pre-RMSPE range           : [{loo['pre_rmspe'].min():.0f}, {loo['pre_rmspe'].max():.0f}]")

    if HAS_MPL:
        _fig5(base, curves)
    return loo


# ==============================================================================
# S5 Figures (matplotlib optional) and reporting
# ==============================================================================
def _month_ticks(ax):
    ticks = list(range(BASE_INDEX, END_INDEX + 2, 12))          # every April
    ax.set_xticks(ticks)
    ax.set_xticklabels([month_index_to_period(t).strftime("%B %Y") for t in ticks],
                       rotation=90)
    ax.set_xlim(BASE_INDEX - 1, END_INDEX + 2)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(DIR_REP, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _fig3(g: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(g[TIME_VAR], g["non_exposure"], color="C2", lw=1.2, label="Non-exposure group")
    ax.plot(g[TIME_VAR], g["exposure"], color="C0", lw=1.2, label="Exposure group")
    ax.axvline(TREATMENT_PERIOD, color="k", lw=1)
    ax.set_ylabel("Ward-level healthcare expenditure per patient (JPY)")
    ax.legend(frameon=False); _month_ticks(ax)
    fig.tight_layout(); _save(fig, "figure3_group_series")


def _fig4(b: dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True)
    yr = b["yr"]
    axes[0].plot(yr, b["synth"], "r--", lw=1, label="Counterfactual prediction")
    axes[0].plot(yr, b["actual"], "b-", lw=1, label="Exposure group")
    axes[0].set_title("(a) Original graph", loc="left"); axes[0].legend(frameon=False)
    axes[1].plot(yr, b["gap"], "b-", lw=1); axes[1].axhline(0, color="r", ls="--", lw=0.8)
    axes[1].set_title("(b) Pointwise graph (observed - synthetic)", loc="left")
    axes[2].plot(yr, b["cum_gap"], "b-", lw=1); axes[2].axhline(0, color="r", ls="--", lw=0.8)
    axes[2].set_title("(c) Cumulative graph (accumulated from April 2021)", loc="left")
    for ax in axes:
        ax.axvline(TREATMENT_PERIOD, color="k", lw=1)
    fig.supylabel("Ward-level healthcare expenditure per patient (JPY)")
    _month_ticks(axes[2]); fig.tight_layout(); _save(fig, "figure4_synthetic_control")


def _fig5(b: dict, curves: dict):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in curves.values():
        ax.plot(r["yr"], r["cum_gap"], color="0.75", lw=0.8, zorder=1)
    ax.plot(b["yr"], b["cum_gap"], color="C0", lw=2, label="Cumulative gap (all donors)", zorder=3)
    ax.plot([], [], color="0.75", lw=0.8, label="Leave-one-out re-estimations")
    ax.axvline(TREATMENT_PERIOD, color="k", lw=1); ax.axhline(0, color="r", ls="--", lw=0.8)
    ax.set_ylabel("Cumulative gap (JPY per patient)"); ax.legend(frameon=False)
    _month_ticks(ax); fig.tight_layout(); _save(fig, "figure5_leave_one_out")


def write_run_settings(n_optim: int):
    try:
        from importlib.metadata import version
        pkg = version("SyntheticControlMethods")
    except Exception:
        pkg = "unknown"
    json.dump({"pipeline_version": VERSION, "SyntheticControlMethods": pkg,
               "python": sys.version.split()[0], "n_optim": n_optim, "pen": PEN,
               "random_seed": RANDOM_SEED, "treatment_period": int(TREATMENT_PERIOD),
               "baseline_months": [BASE_MONTH, str(month_index_to_period(TREATMENT_PERIOD - 1))],
               "follow_up_months": [TREATMENT_MONTH, END_MONTH]},
              open(os.path.join(DIR_OUT, "run_settings.json"), "w"), indent=2)


# ==============================================================================
# Execution (checkpoint style)
# ==============================================================================
def pipeline_status() -> pd.DataFrame:
    need = ["figure3_group_series.csv", "descriptive_group_means.csv",
            "table3_covariates_by_ward.csv", "table2_donor_weights.csv",
            "table4_covariate_balance.csv", "table5_rmspe.csv", "figure4_series.csv",
            "scm_estimates_summary.csv", "table6_leave_one_out.csv", "figure5_loo_series.csv"]
    return pd.DataFrame({"file": need,
                         "exists": [os.path.exists(os.path.join(DIR_OUT, f)) for f in need]})


def run_all(n_optim: int = N_OPTIM, force: bool = False, path: str | None = None) -> pd.DataFrame:
    """Run S1-S5. Steps whose outputs exist are skipped unless force=True."""
    d = prepare_panel(path)
    descriptives(d)
    base = None
    if force or not os.path.exists(os.path.join(DIR_OUT, "scm_estimates_summary.csv")):
        base = run_baseline(d, n_optim)
    else:
        _msg("SKIP S3 baseline (output exists; use force=True to re-run)")
    if force or not os.path.exists(os.path.join(DIR_OUT, "table6_leave_one_out.csv")):
        base = base or run_baseline(d, n_optim)
        run_leave_one_out(d, base, n_optim)
    else:
        _msg("SKIP S4 leave-one-out (output exists; use force=True to re-run)")
    write_run_settings(n_optim)
    _msg("run_all done: output/, report/")
    return pipeline_status()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SCM pipeline for DRT and ward-level HE (v%s)" % VERSION)
    ap.add_argument("--run-all", action="store_true", help="run S3-S4 (baseline SCM and leave-one-out)")
    ap.add_argument("--force", action="store_true", help="re-run steps even if outputs exist")
    ap.add_argument("--n-optim", type=int, default=N_OPTIM, help="optimizer initializations (default 200)")
    ap.add_argument("--panel", default=None, help="path to the panel CSV (default data/he_panel.csv)")
    a = ap.parse_args()
    if a.run_all:
        print(run_all(n_optim=a.n_optim, force=a.force, path=a.panel).to_string(index=False))
    else:
        d = prepare_panel(a.panel)
        descriptives(d)
        print(pipeline_status().to_string(index=False))
        _msg(f"v{VERSION} loaded. Full run: python scm_drt_pipeline.py --run-all --force "
             f"(n_optim = {N_OPTIM}). Quick test: SCM_N_OPTIM=10 python scm_drt_pipeline.py --run-all --force")
