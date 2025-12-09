"""
Utility script to copy non-PHI, aggregate artifacts into the submission folder.

Copies:
- Kaplan-Meier plots (full and 0–180 days)
- ROC curves
- Data linkage schematic
- Aggregated metrics (friction_summary.json)
- Cox model summary (coefficients only)
"""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export submission-ready artifacts (no PHI).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory with analysis outputs")
    parser.add_argument("--submission-dir", type=Path, required=True, help="Target submission directory")
    args = parser.parse_args()

    src = args.output_dir
    dst = args.submission_dir
    dst.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "kaplan_meier_friction.png",
        "kaplan_meier_friction_180.png",
        "roc_curves.png",
        "data_linkage.png",
        "friction_summary.json",
        "cox_model_summary.csv",
    ]

    for fname in files_to_copy:
        src_path = src / fname
        if src_path.exists():
            shutil.copy2(src_path, dst / fname)
            print(f"Copied {src_path} -> {dst / fname}")
        else:
            print(f"Skipping missing file: {src_path}")


if __name__ == "__main__":
    main()
