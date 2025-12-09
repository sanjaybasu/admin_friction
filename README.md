# Administrative Friction Analysis

## Environment
- Python ≥3.10
- Install dependencies:
  ```bash
  pip install -r packaging/admin_friction/requirements.txt
  ```

## Reproducible pipeline
```bash
python packaging/admin_friction/run_friction_analysis.py \
  --data-dir data/real_inputs \
  --output-dir notebooks/admin_friction/outputs \
  --score-annotations
```

Key outputs (in `notebooks/admin_friction/outputs/`):
- `friction_summary.json` — headline metrics
- `friction_events.parquet` — per-note barrier flags and probabilities
- `kaplan_meier_friction.png` and `_180.png` — survival curves
- `roc_curves.png` — ROC curves for all classifiers
- `data_linkage.png` — data linkage schematic
- `cox_model_summary.csv` — full Cox coefficients
- `annotation_sample_adjudicated.csv`, `annotation_metrics.json` — gold standard and validation metrics
- `friction_analysis.md` — narrative summary stub

Expected runtime: a few minutes on a modern laptop (TF-IDF + logistic regression). Memory: <4 GB.

## Export submission-ready artifacts (no PHI)
To copy non-PHI, aggregate artifacts into the submission folder:
```bash
python packaging/admin_friction/export_submission_assets.py \
  --output-dir notebooks/admin_friction/outputs \
  --submission-dir notebooks/admin_friction/submission
```

## Post-hoc analyses (tables/figures for manuscript)
Once `patient_level_data.csv` is available (see submission folder), run:
```bash
python packaging/admin_friction/posthoc_analyses.py \
  --patient-data notebooks/admin_friction/submission/patient_level_data.csv \
  --output-dir notebooks/admin_friction/submission \
  --intensity
```
This regenerates prevalence/incident tables, propensity-score estimates, PMPM summary, cost-effectiveness, and count-based intensity sensitivity outputs.
