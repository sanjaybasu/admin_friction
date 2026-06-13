#!/usr/bin/env python3
"""Revision 1 — Misclassification sensitivity analysis (revised eTable 6).

Addresses Reviewer 2 (classifier performance / presentation of results).

The original eTable 6 applied the Rogan-Gladen correction using an assumed
specificity of 0.95 and sensitivities that were not the same as those reported
in eTable 1. This revised analysis:
  * uses the SAME sensitivities reported in eTable 1 (the validated classifier
    recall on the held-out temporal test split), fixing that internal
    inconsistency, and
  * presents the misclassification-adjusted ("true") prevalence across a RANGE
    of plausible specificities (0.90, 0.95, 0.98) rather than a single assumed
    value, so the dependence of the adjustment on the specificity assumption is
    explicit.

It uses only published summary statistics (no model artifacts, no patient data),
so it is fully reproducible by an auditor with this file alone. NB: the archived
classifier pickles were produced under scikit-learn 1.5.2; re-running them under
a different version yields unfaithful predictions, so eTable 1 is taken as the
authoritative, version-stable source of operating characteristics.

Rogan-Gladen estimator:  P_true = (P_obs + Sp - 1) / (Se + Sp - 1),  clipped to [0,1].
"""
import argparse
import json
from pathlib import Path

# Patient-level observed prevalence (%) — primary analysis (Table 2)
OBSERVED_PREVALENCE = {
    "Scheduling": 16.2,
    "Transportation": 6.1,
    "Paperwork": 25.3,
    "Prior authorization": 9.8,
}
# Classifier sensitivity (recall) and precision — eTable 1 (temporal test split)
SENSITIVITY = {"Scheduling": 0.84, "Transportation": 0.44, "Paperwork": 0.61, "Prior authorization": 0.75}
PRECISION  = {"Scheduling": 0.63, "Transportation": 0.42, "Paperwork": 0.62, "Prior authorization": 0.82}
SPECIFICITY_GRID = [0.90, 0.95, 0.98]


def rogan_gladen(p_obs_pct, se, sp):
    v = (p_obs_pct / 100.0 + sp - 1.0) / (se + sp - 1.0)
    return round(max(0.0, min(1.0, v)) * 100.0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    args = ap.parse_args()

    rows = {}
    for b in OBSERVED_PREVALENCE:
        se = SENSITIVITY[b]
        adj = {f"sp_{sp}": rogan_gladen(OBSERVED_PREVALENCE[b], se, sp) for sp in SPECIFICITY_GRID}
        lo = min(adj.values()); hi = max(adj.values())
        rows[b] = {
            "observed_prevalence_pct": OBSERVED_PREVALENCE[b],
            "sensitivity": se,
            "precision": PRECISION[b],
            "adjusted_prevalence_by_specificity_pct": adj,
            "adjusted_prevalence_range_pct": [lo, hi],
            "central_adjusted_pct": adj["sp_0.95"],
        }

    # Robustness check: is the key qualitative ordering (paperwork most prevalent,
    # transportation least prevalent) preserved under every specificity?
    order_obs = sorted(OBSERVED_PREVALENCE, key=OBSERVED_PREVALENCE.get, reverse=True)
    key_ordering_preserved = True
    for sp in SPECIFICITY_GRID:
        adj_vals = {b: rogan_gladen(OBSERVED_PREVALENCE[b], SENSITIVITY[b], sp) for b in rows}
        if max(adj_vals, key=adj_vals.get) != "Paperwork" or adj_vals["Transportation"] != min(adj_vals.values()):
            key_ordering_preserved = False
    result = {
        "method": "Rogan-Gladen P_true=(P_obs+Sp-1)/(Se+Sp-1); Se from eTable 1 recall; Sp grid 0.90/0.95/0.98",
        "barriers": rows,
        "observed_rank_order": order_obs,
        "key_ordering_preserved_across_specificity": key_ordering_preserved,
        "note": "Adjustment applies note-level operating characteristics to patient-level prevalence and is "
                "therefore approximate; for patients with multiple encounters, patient-level sensitivity is "
                "likely higher than note-level recall, so the correction is conservative. Across all specificity "
                "assumptions, paperwork remains the most prevalent burden type and transportation among the least "
                "prevalent; absolute prevalences (especially for low-sensitivity types) remain uncertain.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "revision1_classifier_uncertainty.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
