#!/usr/bin/env python3
"""Reanalysis pipeline for R1 manuscript.

Reads patient_level_data.csv and produces reanalysis_R1_results.json
containing all numbers reported in the manuscript and supplement.

Steps:
1. Person-time: fixed 36-month window (enrollment dates unavailable)
2. Three-tier wage valuation: federal minimum, MIT living wage, RBRVS
3. Per-note minute costs mapped to patient-level totals
4. Encounter-normalized barrier rates
5. Sensitivity: restrict to patients with >=3 encounters
6. Disparities: unadjusted prevalence ratios, payer-stratified, missing ethnicity excluded
7. Adult-only sensitivity
8. Bootstrap CIs with 1,000 resamples
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-note minute costs (as implemented in run_friction_analysis.py)
# These represent the time burden per flagged encounter note, NOT per episode.
MINUTES_PER_NOTE = {
    "scheduling": 8.1,
    "transportation": 37.5,
    "documentation": 23.0,
    "authorization": 20.0,
}

# Three-tier wage valuation
WAGES = {
    "minimum_wage": 7.25,       # Federal minimum wage (2009)
    "living_wage": 22.00,       # MIT Living Wage, state-weighted WA/VA/OH
    "rbrvs": 33.40,             # CY2026 RBRVS conversion factor
}

BARRIER_COLS = [
    "has_scheduling_barrier",
    "has_transportation_barrier",
    "has_paperwork_barrier",
    "has_authorization_barrier",
]
COUNT_COLS = [
    "count_scheduling_flag",
    "count_transportation_flag",
    "count_documentation_flag",
    "count_authorization_flag",
]
BARRIER_NAMES = ["Scheduling", "Transportation", "Paperwork", "Authorization"]

N_BOOT = 1000
RNG = np.random.default_rng(42)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["has_any_barrier"] = (df["barrier_count"] > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Person-time
# ---------------------------------------------------------------------------

def compute_person_time(df: pd.DataFrame) -> dict:
    """Compute person-time using a fixed 36-month window.

    Enrollment start/end dates are not available in the patient-level
    extract. Each patient is assigned 36 months (3.0 years), matching
    the observation period January 2023 - December 2025.
    """
    original_patient_months = len(df) * 36
    original_patient_years = original_patient_months / 12

    results = {
        "n_patients": len(df),
        "original_patient_years": original_patient_years,
        "original_patient_months": original_patient_months,
        "note": "Person-time uses fixed 36-month window per patient. "
                "Actual enrollment spans may be shorter for some patients, "
                "which would increase incidence rate estimates.",
    }
    return results


# ---------------------------------------------------------------------------
# Cost calculation at three wage tiers
# ---------------------------------------------------------------------------

def compute_costs_three_tiers(df: pd.DataFrame) -> dict:
    """Compute patient-level costs at three wage levels."""
    results = {}

    for wage_name, wage in WAGES.items():
        cost_col = f"cost_{wage_name}"
        df[cost_col] = df["minutes_total"] / 60.0 * wage

        with_barrier = df[df["has_any_barrier"] == 1]
        total_cost = df[cost_col].sum()
        mean_per_affected = with_barrier[cost_col].mean()
        median_per_affected = with_barrier[cost_col].median()

        # Bootstrap CIs (1000 resamples)
        boot_means = []
        for _ in range(N_BOOT):
            sample = RNG.choice(
                with_barrier[cost_col].values,
                size=len(with_barrier),
                replace=True,
            )
            boot_means.append(np.mean(sample))
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)

        results[wage_name] = {
            "wage_per_hour": wage,
            "total_cohort_cost": round(total_cost, 0),
            "mean_per_affected_patient": round(mean_per_affected, 2),
            "mean_95ci": [round(ci_lower, 2), round(ci_upper, 2)],
            "median_per_affected_patient": round(median_per_affected, 2),
            "n_affected": len(with_barrier),
        }

    results["total_patient_hours"] = round(df["minutes_total"].sum() / 60, 0)
    results["total_patient_years_of_time"] = round(
        df["minutes_total"].sum() / 60 / 8760, 2
    )
    return results


def compute_barrier_specific_costs(df: pd.DataFrame) -> dict:
    """Per-barrier-type costs at three wage tiers."""
    results = {}
    for bname, count_col, has_col in zip(
        BARRIER_NAMES, COUNT_COLS, BARRIER_COLS
    ):
        sub = df[df[has_col] == 1]
        min_key = {
            "Scheduling": "scheduling",
            "Transportation": "transportation",
            "Paperwork": "documentation",
            "Authorization": "authorization",
        }[bname]
        mins_per_note = MINUTES_PER_NOTE[min_key]

        tier_results = {}
        for wage_name, wage in WAGES.items():
            cost = sub[count_col] * mins_per_note / 60.0 * wage
            # Bootstrap
            boot_means = []
            for _ in range(N_BOOT):
                sample = RNG.choice(cost.values, size=len(cost), replace=True)
                boot_means.append(np.mean(sample))

            tier_results[wage_name] = {
                "mean": round(cost.mean(), 2),
                "median": round(cost.median(), 2),
                "mean_95ci": [
                    round(np.percentile(boot_means, 2.5), 2),
                    round(np.percentile(boot_means, 97.5), 2),
                ],
                "total": round(cost.sum(), 0),
            }
        results[bname] = {
            "n_affected": len(sub),
            "prevalence_pct": round(len(sub) / len(df) * 100, 1),
            "minutes_per_flagged_note": mins_per_note,
            "mean_flagged_notes_per_patient": round(sub[count_col].mean(), 2),
            "costs": tier_results,
        }
    return results


# ---------------------------------------------------------------------------
# Encounter-normalized barrier rates
# ---------------------------------------------------------------------------

def compute_encounter_normalized_rates(df: pd.DataFrame) -> dict:
    """Barriers per 100 encounters (not just per 100 PY)."""
    results = {}
    total_encounters = df["encounter_count"].sum()

    for bname, count_col in zip(BARRIER_NAMES, COUNT_COLS):
        total_flags = df[count_col].sum()
        rate_per_100_enc = total_flags / total_encounters * 100
        results[bname] = {
            "total_flagged_notes": int(total_flags),
            "total_encounters": int(total_encounters),
            "rate_per_100_encounters": round(rate_per_100_enc, 2),
        }

    any_flags = sum(df[c].sum() for c in COUNT_COLS)
    results["Any barrier"] = {
        "total_flagged_notes": int(any_flags),
        "total_encounters": int(total_encounters),
        "rate_per_100_encounters": round(any_flags / total_encounters * 100, 2),
    }
    return results


# ---------------------------------------------------------------------------
# Sensitivity: patients with >=3 encounters
# ---------------------------------------------------------------------------

def sensitivity_min_encounters(df: pd.DataFrame, min_enc: int = 3) -> dict:
    """Restrict to patients with >= min_enc encounters."""
    sub = df[df["encounter_count"] >= min_enc].copy()
    n_excluded = len(df) - len(sub)

    barrier_prev = sub["has_any_barrier"].mean() * 100
    pt = compute_person_time(sub)

    cost_results = {}
    with_barrier = sub[sub["has_any_barrier"] == 1]
    for wage_name, wage in WAGES.items():
        cost = with_barrier["minutes_total"] / 60.0 * wage
        cost_results[wage_name] = {
            "mean_per_affected": round(cost.mean(), 2),
            "n_affected": len(with_barrier),
        }

    return {
        "min_encounters": min_enc,
        "n_included": len(sub),
        "n_excluded": n_excluded,
        "pct_excluded": round(n_excluded / len(df) * 100, 1),
        "barrier_prevalence_pct": round(barrier_prev, 1),
        "costs": cost_results,
    }


# ---------------------------------------------------------------------------
# Disparities
# ---------------------------------------------------------------------------

def compute_disparities(df: pd.DataFrame) -> dict:
    """Compute barrier prevalence by ethnicity, excluding unknown."""
    known = df[~df["ethnicity"].isin(["Unknown", "unknown", ""])].copy()
    n_unknown = len(df) - len(known)
    pct_unknown = round(n_unknown / len(df) * 100, 1)

    # Harmonize ethnicity labels
    eth_map = {
        "African American": "African American",
        "African  American": "African American",
        "Caucasian": "White",
        "Hispanic": "Hispanic",
        "Asian": "Asian",
        "American Indian or Alaska Native": "AIAN",
        "Native American": "AIAN",
        "Pacific Islander": "NHPI",
        "Native Hawaiian": "NHPI",
    }
    known["eth_clean"] = known["ethnicity"].map(eth_map).fillna("Other")

    # Unadjusted prevalence
    prev_by_eth = known.groupby("eth_clean")["has_any_barrier"].agg(
        ["mean", "count"]
    )
    prev_by_eth.columns = ["prevalence", "n"]
    prev_by_eth["prevalence_pct"] = (prev_by_eth["prevalence"] * 100).round(1)

    # Rate ratios vs White (unadjusted)
    white_prev = prev_by_eth.loc["White", "prevalence"]
    prev_by_eth["rate_ratio_vs_white"] = (
        prev_by_eth["prevalence"] / white_prev
    ).round(3)

    # Bootstrap CIs for rate ratios
    rr_cis = {}
    for eth in prev_by_eth.index:
        if eth == "White":
            rr_cis[eth] = {"rr": 1.0, "ci": [1.0, 1.0]}
            continue
        eth_data = known[known["eth_clean"] == eth]["has_any_barrier"].values
        white_data = known[known["eth_clean"] == "White"][
            "has_any_barrier"
        ].values
        boot_rrs = []
        for _ in range(N_BOOT):
            e_sample = RNG.choice(eth_data, size=len(eth_data), replace=True)
            w_sample = RNG.choice(
                white_data, size=len(white_data), replace=True
            )
            if w_sample.mean() > 0:
                boot_rrs.append(e_sample.mean() / w_sample.mean())
        if boot_rrs:
            rr_cis[eth] = {
                "rr": round(np.mean(boot_rrs), 3),
                "ci": [
                    round(np.percentile(boot_rrs, 2.5), 3),
                    round(np.percentile(boot_rrs, 97.5), 3),
                ],
            }

    # Payer-stratified prevalence (proxy for state adjustment)
    payer_results = {}
    for payer in known["payer"].unique():
        psub = known[known["payer"] == payer]
        payer_prev = psub.groupby("eth_clean")["has_any_barrier"].agg(
            ["mean", "count"]
        )
        payer_results[payer] = {
            eth: {"prev": round(row["mean"] * 100, 1), "n": int(row["count"])}
            for eth, row in payer_prev.iterrows()
            if row["count"] >= 30  # minimum cell size
        }

    # Missing ethnicity sensitivity: assume all unknown are highest-barrier group
    worst_case_prev = (
        df["has_any_barrier"].sum() / len(df) * 100
    )  # unchanged overall

    return {
        "n_known_ethnicity": len(known),
        "n_unknown_ethnicity": n_unknown,
        "pct_unknown": pct_unknown,
        "prevalence_by_ethnicity": {
            eth: {
                "n": int(row["n"]),
                "prevalence_pct": round(row["prevalence_pct"], 1),
                "rate_ratio_vs_white": round(row["rate_ratio_vs_white"], 3),
                "rr_95ci": rr_cis.get(eth, {}).get("ci"),
            }
            for eth, row in prev_by_eth.iterrows()
        },
        "payer_stratified": payer_results,
        "note": f"{pct_unknown}% of patients had unknown ethnicity and were "
                "excluded from disparity analysis. Results should be interpreted "
                "with caution given potential informative missingness.",
    }


# ---------------------------------------------------------------------------
# Adult-only sensitivity
# ---------------------------------------------------------------------------

def adult_only_analysis(df: pd.DataFrame) -> dict:
    """Repeat key metrics for adults (age >= 18) only."""
    adults = df[df["age"] >= 18].copy()
    children = df[df["age"] < 18]

    adult_barrier_prev = adults["has_any_barrier"].mean() * 100
    child_barrier_prev = children["has_any_barrier"].mean() * 100

    with_barrier_adult = adults[adults["has_any_barrier"] == 1]
    cost_results = {}
    for wage_name, wage in WAGES.items():
        cost = with_barrier_adult["minutes_total"] / 60.0 * wage
        cost_results[wage_name] = {
            "mean_per_affected": round(cost.mean(), 2),
            "n_affected": len(with_barrier_adult),
        }

    return {
        "n_adults": len(adults),
        "n_children": len(children),
        "adult_barrier_prevalence_pct": round(adult_barrier_prev, 1),
        "child_barrier_prevalence_pct": round(child_barrier_prev, 1),
        "adult_costs": cost_results,
    }


# ---------------------------------------------------------------------------
# Age breakdown for Table 1
# ---------------------------------------------------------------------------

def age_breakdown(df: pd.DataFrame) -> dict:
    bins = [0, 18, 45, 65, 200]
    labels = ["0-17", "18-44", "45-64", "65+"]
    df["age_cat"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    results = {}
    for cat in labels:
        sub = df[df["age_cat"] == cat]
        with_b = sub[sub["has_any_barrier"] == 1]
        without_b = sub[sub["has_any_barrier"] == 0]
        results[cat] = {
            "overall_n": len(sub),
            "overall_pct": round(len(sub) / len(df) * 100, 1),
            "barrier_n": len(with_b),
            "barrier_pct_of_age_group": round(
                len(with_b) / len(sub) * 100, 1
            ) if len(sub) > 0 else 0,
            "no_barrier_n": len(without_b),
        }

    # Barrier/no-barrier age category counts for Table 1
    barrier_group = df[df["has_any_barrier"] == 1]
    no_barrier_group = df[df["has_any_barrier"] == 0]
    for cat in labels:
        results[cat]["in_barrier_group_n"] = int(
            (barrier_group["age_cat"] == cat).sum()
        )
        results[cat]["in_barrier_group_pct"] = round(
            (barrier_group["age_cat"] == cat).sum()
            / len(barrier_group)
            * 100,
            1,
        )
        results[cat]["in_no_barrier_group_n"] = int(
            (no_barrier_group["age_cat"] == cat).sum()
        )
        results[cat]["in_no_barrier_group_pct"] = round(
            (no_barrier_group["age_cat"] == cat).sum()
            / len(no_barrier_group)
            * 100,
            1,
        )

    return results


# ---------------------------------------------------------------------------
# Incidence rates (with and without encounter normalization)
# ---------------------------------------------------------------------------

def compute_incidence(df: pd.DataFrame) -> dict:
    pt = compute_person_time(df)
    patient_years = pt["original_patient_years"]

    results = {}
    for bname, count_col in zip(BARRIER_NAMES, COUNT_COLS):
        total_events = df[count_col].sum()
        rate = total_events / patient_years * 100
        from scipy.stats import poisson
        ci_lo = poisson.ppf(0.025, total_events) / patient_years * 100
        ci_hi = poisson.ppf(0.975, total_events) / patient_years * 100

        results[bname] = {
            "total_events": int(total_events),
            "patient_years": round(patient_years, 0),
            "rate_per_100py": round(rate, 2),
            "rate_95ci": [round(ci_lo, 2), round(ci_hi, 2)],
            "patients_with_barrier": int(df[BARRIER_COLS[BARRIER_NAMES.index(bname)]].sum()),
            "patient_prevalence_pct": round(
                df[BARRIER_COLS[BARRIER_NAMES.index(bname)]].mean() * 100, 1
            ),
        }

    # Any barrier
    any_events = sum(df[c].sum() for c in COUNT_COLS)
    any_rate = any_events / patient_years * 100
    results["Any barrier"] = {
        "total_events": int(any_events),
        "patient_years": round(patient_years, 0),
        "rate_per_100py": round(any_rate, 2),
        "patients_with_barrier": int(df["has_any_barrier"].sum()),
        "patient_prevalence_pct": round(df["has_any_barrier"].mean() * 100, 1),
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="R1 reanalysis")
    parser.add_argument(
        "--patient-data",
        type=Path,
        default=Path(__file__).parent / "data" / "patient_level_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
    )
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(args.patient_data)
    print(f"  {len(df):,} patients loaded")

    print("\n=== Person-Time ===")
    pt = compute_person_time(df)
    print(f"  Original: {pt['original_patient_years']:,.0f} patient-years "
          f"({pt['original_patient_months']:,} patient-months)")
    print(f"  Note: {pt['note']}")

    print("\n=== Age Breakdown ===")
    ages = age_breakdown(df)
    for cat, vals in ages.items():
        print(f"  {cat}: n={vals['overall_n']:,} ({vals['overall_pct']}%), "
              f"barrier={vals['barrier_pct_of_age_group']}%")

    print("\n=== Incidence Rates (per 100 PY) ===")
    incidence = compute_incidence(df)
    for bname, vals in incidence.items():
        print(f"  {bname}: {vals['rate_per_100py']} "
              f"(prevalence {vals['patient_prevalence_pct']}%)")

    print("\n=== Encounter-Normalized Rates (per 100 encounters) ===")
    enc_rates = compute_encounter_normalized_rates(df)
    for bname, vals in enc_rates.items():
        print(f"  {bname}: {vals['rate_per_100_encounters']} per 100 encounters")

    print("\n=== Costs at Three Wage Tiers ===")
    costs = compute_costs_three_tiers(df)
    print(f"  Total patient-hours: {costs['total_patient_hours']:,.0f}")
    for wage_name in WAGES:
        c = costs[wage_name]
        print(f"  {wage_name} (${WAGES[wage_name]}/hr): "
              f"total=${c['total_cohort_cost']:,.0f}, "
              f"mean/affected=${c['mean_per_affected_patient']:.2f} "
              f"({c['mean_95ci']})")

    print("\n=== Barrier-Specific Costs ===")
    barrier_costs = compute_barrier_specific_costs(df)
    for bname, vals in barrier_costs.items():
        rbrvs = vals["costs"]["rbrvs"]
        print(f"  {bname}: n={vals['n_affected']:,}, "
              f"mean flags/patient={vals['mean_flagged_notes_per_patient']}, "
              f"RBRVS mean=${rbrvs['mean']:.2f}")

    print("\n=== Disparities ===")
    disparities = compute_disparities(df)
    print(f"  Unknown ethnicity: {disparities['pct_unknown']}%")
    for eth, vals in disparities["prevalence_by_ethnicity"].items():
        ci = vals.get("rr_95ci", "")
        print(f"  {eth}: n={vals['n']:,}, prev={vals['prevalence_pct']}%, "
              f"RR={vals['rate_ratio_vs_white']} {ci}")

    print("\n=== Sensitivity: >=3 encounters ===")
    sens_enc = sensitivity_min_encounters(df, min_enc=3)
    print(f"  Excluded: {sens_enc['n_excluded']:,} ({sens_enc['pct_excluded']}%)")
    print(f"  Barrier prevalence: {sens_enc['barrier_prevalence_pct']}%")

    print("\n=== Adult-Only Analysis ===")
    adult = adult_only_analysis(df)
    print(f"  Adults: {adult['n_adults']:,}, children: {adult['n_children']:,}")
    print(f"  Adult barrier prevalence: {adult['adult_barrier_prevalence_pct']}%")
    print(f"  Child barrier prevalence: {adult['child_barrier_prevalence_pct']}%")

    # Save all results
    all_results = {
        "person_time": pt,
        "age_breakdown": ages,
        "incidence": incidence,
        "encounter_normalized_rates": enc_rates,
        "costs_three_tiers": costs,
        "barrier_specific_costs": barrier_costs,
        "disparities": disparities,
        "sensitivity_min_3_encounters": sens_enc,
        "adult_only": adult,
        "methodology_notes": {
            "minutes_per_note": MINUTES_PER_NOTE,
            "wages": WAGES,
            "bootstrap_resamples": N_BOOT,
            "person_time_method": "Fixed 36-month window (original pipeline). "
                                  "Actual enrollment-based person-time not "
                                  "available in patient_level_data.csv.",
            "cost_method": "Per-note costs: each flagged note contributes "
                           "barrier-specific minutes. Patient total = sum "
                           "across all flagged notes. Monetized at three "
                           "wage levels.",
        },
    }

    out_path = args.output_dir / "reanalysis_R1_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
