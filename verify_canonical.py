#!/usr/bin/env python3
"""Audit gate: verify reproducible outputs match CANONICAL_NUMBERS.md.

Run after reanalysis_R1.py, revision1_selection_analysis.py, and
revision1_classifier_uncertainty.py. Exits non-zero if any locked value drifts.
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "output"
# revision outputs may live in the notebooks workspace; allow override via argv[1]
REV = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

checks, fails = [], []


def chk(name, got, want, tol=0.0):
    ok = (got == want) if tol == 0 else (got is not None and abs(got - want) <= tol)
    checks.append((name, got, want, ok))
    if not ok:
        fails.append(name)


r = json.load(open(OUT / "reanalysis_R1_results.json"))
chk("cohort n", r["person_time"]["n_patients"], 49282)
chk("any-burden prevalence %", r["incidence"]["Any barrier"]["patient_prevalence_pct"], 36.8, 0.05)
chk("paperwork prevalence %", r["incidence"]["Paperwork"]["patient_prevalence_pct"], 25.3, 0.05)
chk("transportation prevalence %", r["incidence"]["Transportation"]["patient_prevalence_pct"], 6.1, 0.05)
chk("RBRVS total cost", r["costs_three_tiers"]["rbrvs"]["total_cohort_cost"], 628665, 1)
chk("transportation RBRVS mean", r["barrier_specific_costs"]["Transportation"]["costs"]["rbrvs"]["mean"], 47.58, 0.01)
chk("AfrAm RR", r["disparities"]["prevalence_by_ethnicity"]["African American"]["rate_ratio_vs_white"], 1.216, 0.001)
chk("pct unknown ethnicity", r["disparities"]["pct_unknown"], 23.2, 0.05)
chk(">=3 enc prevalence %", r["sensitivity_min_3_encounters"]["barrier_prevalence_pct"], 58.6, 0.05)

sel_path = REV / "revision1_selection_results.json"
if sel_path.exists():
    s = json.load(open(sel_path))
    chk("eligible (4 MCOs)", s["n_eligible_4plans"], 142473)
    chk("engaged", s["n_engaged"], 49282)
    chk("not engaged", s["n_not_engaged"], 93191)
    chk("engagement rate %", s["engagement_rate_pct"], 34.6, 0.05)
    chk("engaged age (matches Table 1)", s["age_mean"]["engaged"], 33.7, 0.05)
    chk("engaged female % (matches Table 1)", s["female_pct"]["engaged"], 60.5, 0.05)
    chk("AfrAm SMD engaged-vs-not", s["ethnicity_pct"]["African American"]["smd"], 0.206, 0.01)
else:
    print(f"[skip] {sel_path} not found (needs raw inputs + Lighthouse)")

cls_path = REV / "revision1_classifier_uncertainty.json"
if cls_path.exists():
    c = json.load(open(cls_path))
    chk("classifier key ordering robust", c["key_ordering_preserved_across_specificity"], True)
    chk("paperwork central adj %", c["barriers"]["Paperwork"]["central_adjusted_pct"], 36.2, 0.1)
else:
    print(f"[skip] {cls_path} not found")

print(f"\n{'CHECK':45s} {'GOT':>12s} {'WANT':>12s}  OK")
for name, got, want, ok in checks:
    print(f"{name:45s} {str(got):>12s} {str(want):>12s}  {'✓' if ok else '✗ FAIL'}")
print(f"\n{len(checks)-len(fails)}/{len(checks)} passed")
sys.exit(1 if fails else 0)
