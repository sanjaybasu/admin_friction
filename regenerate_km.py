"""
Regenerate KM figure (eFigure 3) from existing pipeline outputs + raw data.
Does NOT retrain classifiers — uses saved friction_events.parquet.
"""
import sys
from pathlib import Path

# Add parent so we can import survival_analysis from the main module
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from run_friction_analysis import survival_analysis

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "real_inputs"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "admin_friction" / "outputs"
SUBMISSION_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "admin_friction" / "submission"

# Load friction events (ML-flagged notes)
fe = pd.read_parquet(OUTPUT_DIR / "friction_events.parquet")
fe = fe.rename(columns={"note_dt": "note_date"})
fe["note_date"] = pd.to_datetime(fe["note_date"], errors="coerce")

# Load member attributes for age/gender/ethnicity
attr = pd.read_parquet(DATA_DIR / "member_attributes.parquet",
                       columns=["member_id", "gender", "birth_date", "ethnicity"])
mpm = pd.read_parquet(DATA_DIR / "member_patient_map.parquet",
                      columns=["member_id", "patient_id"]).drop_duplicates()
elig = pd.read_parquet(DATA_DIR / "eligibility.parquet",
                       columns=["person_id", "member_id"]).drop_duplicates()

# Build patient → attributes
person_member = elig[["person_id", "member_id"]].drop_duplicates()
pmp = person_member.merge(mpm, on="member_id", how="left")
patient_attr = pmp.merge(attr, on="member_id", how="left").drop_duplicates("patient_id")
patient_attr["birth_date"] = pd.to_datetime(patient_attr["birth_date"], errors="coerce")
patient_attr["age"] = (pd.Timestamp("2024-06-01") - patient_attr["birth_date"]).dt.days / 365.25

# Merge onto notes
notes = fe.merge(patient_attr[["patient_id", "age", "gender", "ethnicity"]], on="patient_id", how="left")

# Load hospital visits
hosp = pd.read_parquet(DATA_DIR / "hospital_visits.parquet",
                       columns=["patient_id", "admit_date", "patient_class_code"])

# Regenerate all KM variants
km_path = OUTPUT_DIR / "kaplan_meier_friction.png"

print("Generating primary KM (combined, landmark=180)...")
survival_analysis(notes, hosp, km_path, outcome="combined")

print("Generating ED-only KM...")
survival_analysis(notes, hosp, km_path.with_name("kaplan_meier_friction_ed.png"), outcome="ed")

print("Generating inpatient-only KM...")
survival_analysis(notes, hosp, km_path.with_name("kaplan_meier_friction_inpatient.png"), outcome="inpatient")

# Sensitivity landmarks
for lm in [90, 180, 365]:
    print(f"Generating landmark={lm} KM...")
    survival_analysis(notes, hosp, km_path.with_name(f"kaplan_meier_friction_{lm}.png"), landmark_days=lm, outcome="combined")

# Copy primary to submission as eFigure 3
import shutil
src = OUTPUT_DIR / "kaplan_meier_friction.png"
dst = SUBMISSION_DIR / "efigure3_kaplan_meier.png"
shutil.copy2(src, dst)
print(f"\nCopied {src} -> {dst}")
print("Done.")
