"""
Post-hoc analyses for the administrative friction study.

Given a patient-level dataset (derived from the main pipeline), this script
recomputes the secondary tables/figures used in the manuscript:
  - prevalence_summary.csv
  - incidence_per_100py.csv
  - cost_among_with_barrier.csv
  - propensity_barrier_acute.csv
  - cost_effectiveness.csv
  - pmpm_analysis.json
  - intensity_* outputs (optional, count-based proxy)

The script assumes the patient-level file contains:
  patient_id, time_cost_lower, minutes_total,
  has_{scheduling,transportation,paperwork,authorization}_barrier,
  count_{scheduling,transportation,documentation,authorization}_flag,
  encounter_count, barrier_count, prior_ed_visits, acute_event,
  gender, ethnicity, race, age

Usage:
  python posthoc_analyses.py \
    --patient-data notebooks/admin_friction/submission/patient_level_data.csv \
    --output-dir notebooks/admin_friction/submission

Optional arguments:
  --patient-months <float>   Override patient-months denominator (default: n_patients*36)
  --intensity                Emit intensity-derived outputs from barrier counts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def prevalence_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["any_barrier"] = (
        df["has_scheduling_barrier"]
        | df["has_transportation_barrier"]
        | df["has_paperwork_barrier"]
        | df["has_authorization_barrier"]
    ).astype(int)

    groups = {
        "Overall": df,
        "Any Barrier": df[df["any_barrier"] == 1],
        "No Barrier": df[df["any_barrier"] == 0],
    }
    rows = []
    for name, g in groups.items():
        n = len(g)
        rows.append(
            {
                "group": name,
                "n": n,
                "age_mean": g["age"].mean(),
                "age_sd": g["age"].std(),
                "female_pct": (g["gender"] == "Female").mean() * 100,
                "prior_ed_mean": g["prior_ed_visits"].mean(),
                "encounter_count_mean": g["encounter_count"].mean(),
                "acute_event_pct": g["acute_event"].mean() * 100,
                "time_cost_mean": g["time_cost_lower"].mean(),
                "scheduling_prev_pct": g["has_scheduling_barrier"].mean() * 100,
                "transportation_prev_pct": g["has_transportation_barrier"].mean() * 100,
                "paperwork_prev_pct": g["has_paperwork_barrier"].mean() * 100,
                "authorization_prev_pct": g["has_authorization_barrier"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def incidence_per_100py(df: pd.DataFrame, patient_months: float) -> pd.DataFrame:
    patient_years = patient_months / 12.0
    barriers = {
        "scheduling": "has_scheduling_barrier",
        "transportation": "has_transportation_barrier",
        "paperwork": "has_paperwork_barrier",
        "authorization": "has_authorization_barrier",
    }
    rows = []
    for name, col in barriers.items():
        n_with = int(df[col].sum())
        rate = (n_with / patient_years) * 100 if patient_years > 0 else np.nan
        # Poisson exact CI
        ci_low, ci_hi = stats.poisson.interval(0.95, n_with)
        rows.append(
            {
                "barrier": name,
                "n_patients": len(df),
                "n_with_barrier": n_with,
                "patient_years": patient_years,
                "incidence_per_100py": rate,
                "ci_lower": (ci_low / patient_years) * 100 if patient_years > 0 else np.nan,
                "ci_upper": (ci_hi / patient_years) * 100 if patient_years > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cost_among_with_barrier(df: pd.DataFrame) -> pd.DataFrame:
    barriers = {
        "scheduling": "has_scheduling_barrier",
        "transportation": "has_transportation_barrier",
        "paperwork": "has_paperwork_barrier",
        "authorization": "has_authorization_barrier",
    }
    rows = []
    for name, col in barriers.items():
        subset = df[df[col] == 1]
        if subset.empty:
            continue
        costs = subset["time_cost_lower"].values
        # Bootstrap CI for mean
        rng = np.random.default_rng(13)
        boots = []
        for _ in range(1000):
            boots.append(rng.choice(costs, size=len(costs), replace=True).mean())
        rows.append(
            {
                "barrier": name,
                "n_with_barrier": len(subset),
                "mean_cost": costs.mean(),
                "mean_ci_lower": float(np.percentile(boots, 2.5)),
                "mean_ci_upper": float(np.percentile(boots, 97.5)),
                "median_cost": float(np.median(costs)),
                "p25": float(np.percentile(costs, 25)),
                "p75": float(np.percentile(costs, 75)),
                "total_cost": float(costs.sum()),
            }
        )
    return pd.DataFrame(rows)


def propensity_barrier(df: pd.DataFrame, treatment_col: str, outcome_col: str = "acute_event", caliper: float = 0.05):
    covariates = ["age", "prior_ed_visits", "encounter_count"]
    eth = pd.get_dummies(df["ethnicity"].fillna("Unknown"), prefix="eth", drop_first=True)
    X = pd.concat([df[covariates].fillna(0), eth], axis=1)
    treat = df[treatment_col].astype(int).values

    # Propensity scores
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, treat)
    ps = model.predict_proba(X)[:, 1]

    treated_idx = np.where(treat == 1)[0]
    control_idx = np.where(treat == 0)[0]
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(ps[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
    mask = distances.flatten() <= caliper
    matched_treated = treated_idx[mask]
    matched_control = control_idx[indices.flatten()[mask]]

    treated_outcome = df.iloc[matched_treated][outcome_col].mean()
    control_outcome = df.iloc[matched_control][outcome_col].mean()
    ate = treated_outcome - control_outcome

    # Bootstrap CI
    rng = np.random.default_rng(13)
    boots = []
    for _ in range(1000):
        sel = rng.choice(len(matched_treated), size=len(matched_treated), replace=True)
        t = df.iloc[matched_treated].iloc[sel][outcome_col].mean()
        c = df.iloc[matched_control].iloc[sel][outcome_col].mean()
        boots.append(t - c)
    ci_lower, ci_upper = np.percentile(boots, [2.5, 97.5])

    def smd(a, b, var):
        m1, m2 = a[var].mean(), b[var].mean()
        v1, v2 = a[var].var(), b[var].var()
        denom = np.sqrt((v1 + v2) / 2)
        return 0 if denom == 0 else abs(m1 - m2) / denom

    treated_all = df[df[treatment_col] == 1]
    control_all = df[df[treatment_col] == 0]
    max_smd_before = max(smd(treated_all, control_all, v) for v in covariates)
    matched_t = df.iloc[matched_treated]
    matched_c = df.iloc[matched_control]
    max_smd_after = max(smd(matched_t, matched_c, v) for v in covariates)

    return {
        "n_treated": int((treat == 1).sum()),
        "n_control": int((treat == 0).sum()),
        "n_matched_pairs": int(len(matched_treated)),
        "ate_pct_points": float(ate * 100),
        "ci_lower": float(ci_lower * 100),
        "ci_upper": float(ci_upper * 100),
        "p_value": float(stats.ttest_ind(df.iloc[matched_treated][outcome_col], df.iloc[matched_control][outcome_col]).pvalue),
        "max_smd_before": float(max_smd_before),
        "max_smd_after": float(max_smd_after),
    }


def cost_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["any_barrier"] = (
        df["has_scheduling_barrier"]
        | df["has_transportation_barrier"]
        | df["has_paperwork_barrier"]
        | df["has_authorization_barrier"]
    ).astype(int)
    grp1 = df[df["any_barrier"] == 1]
    grp0 = df[df["any_barrier"] == 0]

    inc_cost = grp1["time_cost_lower"].sum() - grp0["time_cost_lower"].sum()
    inc_events = grp1["acute_event"].sum() - grp0["acute_event"].sum()
    cost_per_event = abs(inc_cost / inc_events) if inc_events != 0 else np.nan

    rng = np.random.default_rng(13)
    boots = []
    for _ in range(1000):
        b1 = grp1.sample(frac=1, replace=True, random_state=None)
        b0 = grp0.sample(frac=1, replace=True, random_state=None)
        inc_c = b1["time_cost_lower"].sum() - b0["time_cost_lower"].sum()
        inc_e = b1["acute_event"].sum() - b0["acute_event"].sum()
        if inc_e != 0:
            boots.append(abs(inc_c / inc_e))
    ci_lower, ci_upper = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)

    return pd.DataFrame(
        [
            {
                "incremental_cost": inc_cost,
                "incremental_events": inc_events,
                "cost_per_event": cost_per_event,
                "cost_per_event_ci_lower": ci_lower,
                "cost_per_event_ci_upper": ci_upper,
            }
        ]
    )


def pmpm_analysis(df: pd.DataFrame, patient_months: float) -> dict:
    total_time = float(df["time_cost_lower"].sum())
    lower = total_time / patient_months if patient_months > 0 else np.nan
    return {
        "patient_months": patient_months,
        "time_cost_pmpm": lower,
        "context": {
            "capitation_reference": "$300-$600 PMPM typical Medicaid comprehensive rate",
            "share_of_400_pmpm": lower / 400 if patient_months > 0 else np.nan,
        },
    }


def intensity_outputs(df: pd.DataFrame, out_dir: Path):
    """Count-based intensity proxy (low/moderate/high) using barrier counts."""
    weights = {
        "scheduling": {"low": 15, "moderate": 45, "high": 90},
        "transportation": {"low": 30, "moderate": 90, "high": 180},
        "paperwork": {"low": 15, "moderate": 30, "high": 60},
        "authorization": {"low": 20, "moderate": 45, "high": 90},
    }
    count_cols = {
        "scheduling": "count_scheduling_flag",
        "transportation": "count_transportation_flag",
        "paperwork": "count_documentation_flag",
        "authorization": "count_authorization_flag",
    }
    df = df.copy()

    def bucket(v):
        if pd.isna(v) or v <= 0:
            return "none"
        if v == 1:
            return "low"
        if v in (2, 3):
            return "moderate"
        return "high"

    for b, count_col in count_cols.items():
        inten_col = f"{b}_intensity"
        df[inten_col] = df[count_col].apply(bucket)
        df[f"{b}_intensity_minutes"] = df.apply(
            lambda row: 0
            if row[f"has_{b}_barrier"] != 1 or row[inten_col] == "none"
            else weights[b][row[inten_col]],
            axis=1,
        )
        df[f"{b}_intensity_cost"] = df[f"{b}_intensity_minutes"] * 7.25 / 60

    df["total_intensity_cost"] = df[[f"{b}_intensity_cost" for b in count_cols]].sum(axis=1)
    df.to_csv(out_dir / "patient_level_data_with_intensity.csv", index=False)

    # Uniform vs intensity comparison
    rows = []
    for b in count_cols:
        subset = df[df[f"has_{b}_barrier"] == 1]
        if subset.empty:
            continue
        dist = subset[f"{b}_intensity"].value_counts(normalize=True)
        rows.append(
            {
                "barrier": b,
                "n_with_barrier": len(subset),
                "uniform_mean_cost": subset["time_cost_lower"].mean(),
                "uniform_total_cost": subset["time_cost_lower"].sum(),
                "intensity_mean_cost": subset[f"{b}_intensity_cost"].mean(),
                "intensity_total_cost": subset[f"{b}_intensity_cost"].sum(),
                "pct_low": dist.get("low", 0) * 100,
                "pct_moderate": dist.get("moderate", 0) * 100,
                "pct_high": dist.get("high", 0) * 100,
                "percent_difference": (
                    (subset[f"{b}_intensity_cost"].mean() - subset["time_cost_lower"].mean())
                    / subset["time_cost_lower"].mean()
                    * 100
                )
                if subset["time_cost_lower"].mean() != 0
                else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "intensity_cost_comparison.csv", index=False)

    # Disparity by ethnicity
    disp_rows = []
    major_eth = ["African American", "Caucasian", "Hispanic", "Asian"]
    for b in count_cols:
        for eth in major_eth:
            subset = df[(df["ethnicity"] == eth) & (df[f"has_{b}_barrier"] == 1)]
            if subset.empty:
                continue
            n_high = (subset[f"{b}_intensity"] == "high").sum()
            disp_rows.append(
                {
                    "barrier": b,
                    "ethnicity": eth,
                    "n_with_barrier": len(subset),
                    "n_high_intensity": n_high,
                    "pct_high_intensity": n_high / len(subset) * 100,
                    "mean_intensity_cost": subset[f"{b}_intensity_cost"].mean(),
                    "median_intensity_cost": subset[f"{b}_intensity_cost"].median(),
                }
            )
    pd.DataFrame(disp_rows).to_csv(out_dir / "intensity_disparity_analysis.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run post-hoc analyses on patient-level friction data.")
    parser.add_argument("--patient-data", type=Path, required=True, help="Path to patient_level_data.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write analysis outputs")
    parser.add_argument(
        "--patient-months",
        type=float,
        default=None,
        help="Patient-months denominator (default: n_patients * 36 months)",
    )
    parser.add_argument("--intensity", action="store_true", help="Emit intensity-based outputs from counts")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.patient_data)
    patient_months = args.patient_months or (len(df) * 36.0)

    prevalence_df = prevalence_summary(df)
    prevalence_df.to_csv(out_dir / "prevalence_summary.csv", index=False)

    incidence_df = incidence_per_100py(df, patient_months)
    incidence_df.to_csv(out_dir / "incidence_per_100py.csv", index=False)

    cost_df = cost_among_with_barrier(df)
    cost_df.to_csv(out_dir / "cost_among_with_barrier.csv", index=False)

    # Propensity for each barrier
    prop_rows = []
    for name, col in [
        ("Scheduling", "has_scheduling_barrier"),
        ("Transportation", "has_transportation_barrier"),
        ("Paperwork", "has_paperwork_barrier"),
        ("Authorization", "has_authorization_barrier"),
    ]:
        res = propensity_barrier(df, col)
        res["barrier"] = name
        prop_rows.append(res)
    pd.DataFrame(prop_rows).to_csv(out_dir / "propensity_barrier_acute.csv", index=False)

    ce_df = cost_effectiveness(df)
    ce_df.to_csv(out_dir / "cost_effectiveness.csv", index=False)

    pmpm = pmpm_analysis(df, patient_months)
    (out_dir / "pmpm_analysis.json").write_text(json.dumps(pmpm, indent=2))

    if args.intensity:
        intensity_outputs(df, out_dir)


if __name__ == "__main__":
    main()
