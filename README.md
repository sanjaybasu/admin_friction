# Administrative Friction in Medicaid: Replication Code


## Repository structure

```
packaging/admin_friction/
  reanalysis_R1.py              # Reproduces all manuscript numbers
  generate_figures.py           # Produces Figure 1 (main text)
  run_friction_analysis.py      # Upstream NLP pipeline (requires raw encounter notes, not included)
  requirements.txt
  README.md
```

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

Reads `data/patient_level_data.csv`. Writes `output/reanalysis_R1_results.json`.

Runtime: under 2 minutes. The bootstrap uses 1,000 resamples with a fixed seed (`np.random.default_rng(42)`).

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

## Contact

Sanjay Basu, MD, PhD — sanjay.basu@waymark.co
