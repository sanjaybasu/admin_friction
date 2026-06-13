# Administrative Burden in Medicaid: Replication Code

Analysis code for the JAMA Health Forum study of administrative burden in
Medicaid measured by natural language processing of care coordination notes.

**This repository contains code only.** Patient-level data, intermediate
outputs, results files, figures, and the manuscript/appendix are intentionally
excluded (see `.gitignore`). The deidentified analytic extract and raw inputs
require institutional data use agreements and managed care plan approval.

## Repository structure

```
reanalysis_R1.py                   # Reproduces primary manuscript numbers
generate_figures.py                # Produces Figure 1 (main text)
run_friction_analysis.py           # Upstream NLP pipeline (requires raw notes)
posthoc_analyses.py                # Exploratory acute-care analyses (eResults)
export_submission_assets.py        # Assembles submission assets
revision1_selection_analysis.py    # R1: engaged vs eligible-nonengaged comparison (eTable 1)
revision1_classifier_uncertainty.py# R1: misclassification sensitivity (eTable 4)
verify_canonical.py                # Audit gate: checks outputs vs locked numbers
requirements.txt
README.md
```

## Revision 1 note (eligibility restatement)

During revision the eligibility/engagement funnel was re-derived from the
authoritative managed care enrollment and program-status data. Consistent with
routine restatement of Medicaid eligibility as claims run out, the reproducible
counts (142,473 eligible in the four participating plans; 49,282 [34.6%]
engaged) supersede the original submission's funnel. The analytic cohort
(49,282) and all downstream estimates are unchanged. Locked reference values
live in `CANONICAL_NUMBERS.md` (kept with the analysis workspace, not in this
public code repo); `verify_canonical.py` checks every reproducible output
against them.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.10 and 3.11.

## Reproduce manuscript numbers

```bash
python reanalysis_R1.py
```

Reads `data/patient_level_data.csv` (not included). Writes `output/reanalysis_R1_results.json` (git-ignored).

Runtime: under 2 minutes. The bootstrap uses 1,000 resamples with a fixed seed (`np.random.default_rng(42)`).

## Revision 1 analyses

```bash
# Selection comparison (engaged vs eligible-not-engaged); needs raw inputs + Lighthouse
python revision1_selection_analysis.py --output-dir output

# Misclassification sensitivity (no data required; uses published eTable-1 operating characteristics)
python revision1_classifier_uncertainty.py --output-dir output

# Audit gate: verify reproducible outputs against locked canonical numbers
python verify_canonical.py output
```

Output JSON contains:

| Key | Manuscript reference |
|-----|---------------------|
| `person_time` | Methods, Study Population |
| `age_breakdown` | Table 1 |
| `incidence` | Table 2, event rates |
| `encounter_normalized_rates` | Table 2, per-100-encounter rates |
| `costs_three_tiers` | Table 2, total cohort costs |
| `barrier_specific_costs` | Table 2, per-barrier costs |
| `disparities` | Table 3 |
| `sensitivity_min_3_encounters` | Results, Sensitivity Analyses |
| `adult_only` | Results, Sensitivity Analyses |

## Reproduce Figure 1

```bash
python generate_figures.py
```

Reads `data/patient_level_data.csv`. Writes `output/figure1_barrier_prevalence_vs_cost.pdf`.

## Data dictionary: patient_level_data.csv

| Column | Description |
|--------|-------------|
| `patient_id` | Deidentified patient identifier |
| `time_cost_lower` | Total time cost at $7.25/hr ($) |
| `minutes_total` | Total barrier-related minutes across all note types |
| `has_scheduling_barrier` | Binary: any scheduling barrier flagged |
| `has_transportation_barrier` | Binary: any transportation barrier flagged |
| `has_paperwork_barrier` | Binary: any paperwork barrier flagged |
| `has_authorization_barrier` | Binary: any authorization barrier flagged |
| `count_scheduling_flag` | Number of encounter notes flagged for scheduling |
| `count_transportation_flag` | Number of encounter notes flagged for transportation |
| `count_documentation_flag` | Number of encounter notes flagged for paperwork |
| `count_authorization_flag` | Number of encounter notes flagged for authorization |
| `encounter_count` | Total care coordination encounters |
| `barrier_count` | Number of distinct barrier types experienced (0-4) |
| `prior_ed_visits` | Prior emergency department visits |
| `prior_inpatient` | Prior inpatient admissions |
| `payer` | Managed care payer code (ABHVA, UHCWA, CHPW, SHPVA) |
| `plan` | Plan identifier (all "Unknown" in deidentified extract) |
| `acute_event` | Binary: any ED visit or inpatient admission during observation |
| `gender` | Patient gender |
| `ethnicity` | Self-reported ethnicity from enrollment files |
| `race` | Race (all "Unknown" in deidentified extract) |
| `age` | Age in years |

## Per-note minute costs

Each flagged encounter note contributes barrier-type-specific minutes:

| Barrier type | Minutes per flagged note | Source |
|-------------|--------------------------|--------|
| Scheduling | 8.1 | Care coordinator time-use surveys |
| Transportation | 37.5 | Care coordinator time-use surveys |
| Paperwork | 23.0 | Care coordinator time-use surveys |
| Authorization | 20.0 | Care coordinator time-use surveys |

Patient-level `minutes_total` = sum of (count of flagged notes per barrier type x minutes per note).

## Wage tiers

| Tier | $/hour | Rationale |
|------|--------|-----------|
| Federal minimum wage | 7.25 | Lower bound; set in 2009 |
| MIT Living Wage | 22.00 | State-population-weighted average (WA, VA, OH), single adult |
| RBRVS conversion factor | 33.40 | CY 2026 Medicare PFS; values patient time = physician time |

## Upstream pipeline

`run_friction_analysis.py` is the NLP classification pipeline that processes raw encounter notes and produces `patient_level_data.csv`. It requires access to the organization's encounter note database, which is not included. The script is provided for methodological transparency.

