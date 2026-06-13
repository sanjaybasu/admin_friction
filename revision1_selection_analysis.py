#!/usr/bin/env python3
"""Revision 1 — Selection (engagement) analysis.

Addresses Reviewer 2: compares demographics of the analytic (engaged) cohort
with eligible Medicaid beneficiaries in the same managed care plans who did NOT
engage, to characterize the direction and magnitude of selection into the sample.

Design notes (for consistency with the primary analysis):
- The ENGAGED column uses the same patient-level extract and the same
  ethnicity/age-bin conventions as reanalysis_R1.py, so it reproduces the
  manuscript's Table 1 "Overall" column exactly.
- The NOT-ENGAGED column is drawn from the same source files used to build the
  primary cohort (eligibility + member_attributes), with the identical ethnicity
  map and the same age reference date (study end, 2025-11-30) verified against
  the patient-level `age` field.
- Eligibility is restricted to the four participating managed care plans
  (ABHVA, UHCWA, CHPW, SHPVA) — the plans represented in the analytic cohort.
- riskScore (acuity) is pulled from Lighthouse (public."Patient") when a Vault
  token is available; otherwise it is omitted.

Requires raw inputs (PHI; not in the public repo): ../../data/real_inputs/.
Outputs aggregate-only results (no PHI) to the chosen output dir.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_PAYERS = ["ABHVA", "UHCWA", "CHPW", "SHPVA"]
AGE_REF = pd.Timestamp("2025-11-30")  # study end; verified against patient_level_data.age

# Same ethnicity harmonization as reanalysis_R1.py::compute_disparities
ETH_MAP = {
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
ETH_ORDER = ["African American", "Asian", "Hispanic", "AIAN", "NHPI", "White", "Other", "Unknown"]


def clean_eth(series: pd.Series) -> pd.Series:
    s = series.fillna("Unknown").replace({"": "Unknown", "unknown": "Unknown"})
    out = s.map(lambda x: "Unknown" if str(x).strip().lower() == "unknown" else ETH_MAP.get(x, None))
    # values not in map and not Unknown -> Other (mirror fillna('Other') on known rows)
    out = out.where(s.str.strip().str.lower().eq("unknown"), out.fillna("Other"))
    out = out.fillna("Unknown")
    return out


def age_from_birth(bd: pd.Series, ref: pd.Timestamp) -> pd.Series:
    bd = pd.to_datetime(bd, errors="coerce")
    return ((ref - bd).dt.days / 365.25)


def smd_continuous(x1, x2):
    x1 = pd.Series(x1).dropna(); x2 = pd.Series(x2).dropna()
    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    sp = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    return (m1 - m2) / sp if sp > 0 else 0.0


def smd_prop(p1, p2):
    sp = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / sp if sp > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / ".." / ".." / "data" / "real_inputs")
    ap.add_argument("--patient-data", type=Path, default=Path(__file__).parent / "data" / "patient_level_data.csv")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    args = ap.parse_args()
    dd = args.data_dir

    # --- Engaged (analytic) cohort: manuscript-consistent demographics ---
    pld = pd.read_csv(args.patient_data)
    engaged_pids = set(pld["patient_id"])
    pld["eth_clean"] = clean_eth(pld["ethnicity"])
    pld["female"] = pld["gender"].astype(str).str.upper().str.startswith("F")
    n_eng = len(pld)

    # --- Eligible pool & not-engaged group ---
    elig = pd.read_parquet(dd / "eligibility.parquet")
    mpm = pd.read_parquet(dd / "member_patient_map.parquet")
    ma = pd.read_parquet(dd / "member_attributes.parquet")

    elig4 = elig[elig["payer"].isin(STUDY_PAYERS)].copy()
    eligible_members = elig4.drop_duplicates("member_id")
    n_eligible = eligible_members["member_id"].nunique()

    # map engaged patient_ids -> member_ids
    eng_members = set(mpm.loc[mpm["patient_id"].isin(engaged_pids), "member_id"])

    # not-engaged = eligible members not engaged
    ne = eligible_members[~eligible_members["member_id"].isin(eng_members)].copy()
    # attach attributes (ethnicity, gender, birth_date) from member_attributes
    ma_u = ma.drop_duplicates("member_id").set_index("member_id")
    ne = ne.merge(ma_u[["ethnicity", "gender", "birth_date"]], left_on="member_id", right_index=True, how="left", suffixes=("", "_ma"))
    # prefer member_attributes birth_date/gender/ethnicity; fall back to eligibility
    ne["birth_date_use"] = ne["birth_date"].fillna(ne.get("birth_date"))
    ne["eth_clean"] = clean_eth(ne["ethnicity"])
    g = ne["gender"].astype(str).str.upper()
    ne["female"] = g.str.startswith("F")
    ne["age"] = age_from_birth(ne["birth_date_use"], AGE_REF).round()
    n_ne = len(ne)

    # --- riskScore from Lighthouse (optional) ---
    risk = {"engaged": None, "not_engaged": None, "smd": None, "available": False}
    try:
        import sys
        sys.path.insert(0, str(Path.home() / ".claude/skills/waymark-data-access/scripts"))
        from wm_conn import lighthouse, query
        eng_db = lighthouse("prod")
        # engaged riskScore
        def pull_risk(pids):
            vals = []
            pids = list(pids)
            for i in range(0, len(pids), 20000):
                df = query(eng_db, 'SELECT "riskScore" FROM public."Patient" WHERE id = ANY(%(ids)s) AND "riskScore" IS NOT NULL',
                           ids=pids[i:i + 20000])
                vals.extend(df["riskScore"].tolist())
            return np.array(vals, dtype=float)
        ne_pids = set(mpm.loc[mpm["member_id"].isin(set(ne["member_id"])), "patient_id"].dropna())
        r_eng = pull_risk(engaged_pids)
        r_ne = pull_risk(ne_pids)
        risk = {
            "engaged": {"n": len(r_eng), "mean": round(float(np.mean(r_eng)), 3), "sd": round(float(np.std(r_eng, ddof=1)), 3)},
            "not_engaged": {"n": len(r_ne), "mean": round(float(np.mean(r_ne)), 3), "sd": round(float(np.std(r_ne, ddof=1)), 3)},
            "smd": round(float(smd_continuous(r_eng, r_ne)), 3),
            "available": True,
        }
    except Exception as e:  # pragma: no cover
        risk["error"] = str(e)

    # --- Build comparison ---
    def age_groups(df):
        bins = [0, 18, 45, 65, 200]; labels = ["0-17", "18-44", "45-64", "65+"]
        cat = pd.cut(df["age"], bins=bins, labels=labels, right=False)
        return {l: round((cat == l).mean() * 100, 1) for l in labels}

    comp = {
        "n_eligible_4plans": int(n_eligible),
        "n_engaged": int(n_eng),
        "n_not_engaged": int(n_ne),
        "engagement_rate_pct": round(n_eng / n_eligible * 100, 1),
        "age_mean": {
            "engaged": round(pld["age"].mean(), 1),
            "not_engaged": round(ne["age"].mean(), 1),
            "smd": round(smd_continuous(pld["age"], ne["age"]), 3),
        },
        "age_sd": {"engaged": round(pld["age"].std(ddof=1), 1), "not_engaged": round(ne["age"].std(ddof=1), 1)},
        "age_groups_pct": {"engaged": age_groups(pld), "not_engaged": age_groups(ne)},
        "female_pct": {
            "engaged": round(pld["female"].mean() * 100, 1),
            "not_engaged": round(ne["female"].mean() * 100, 1),
            "smd": round(smd_prop(pld["female"].mean(), ne["female"].mean()), 3),
        },
        "ethnicity_pct": {},
        "payer_pct": {},
        "risk_score": risk,
    }
    for eth in ETH_ORDER:
        p1 = (pld["eth_clean"] == eth).mean()
        p2 = (ne["eth_clean"] == eth).mean()
        comp["ethnicity_pct"][eth] = {
            "engaged": round(p1 * 100, 1), "not_engaged": round(p2 * 100, 1),
            "smd": round(smd_prop(p1, p2), 3),
        }
    for payer in STUDY_PAYERS:
        p1 = (pld["payer"] == payer).mean()
        p2 = (ne["payer"] == payer).mean()
        comp["payer_pct"][payer] = {
            "engaged": round(p1 * 100, 1), "not_engaged": round(p2 * 100, 1),
            "smd": round(smd_prop(p1, p2), 3),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "revision1_selection_results.json"
    with open(out, "w") as f:
        json.dump(comp, f, indent=2)
    print(json.dumps(comp, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
