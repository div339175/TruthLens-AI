"""Data preparation and validation script for TruthLens AI."""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.datasets.split_dataset import split_metadata_csv

METADATA_DIR = BASE_DIR / "ml" / "data" / "metadata"
DATA_DIR = BASE_DIR / "ml" / "data"


def validate_and_report_metadata(csv_path: Path):
    """Validates records in the CSV, removes broken paths, ensures splits, and prints statistics."""
    if not csv_path.exists():
        print(f"[ERROR] Metadata file does not exist at {csv_path}")
        return

    valid_rows: List[Dict[str, Any]] = []
    removed_count = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_rel = row.get("image_path", "")
            if img_rel:
                full_img_path = DATA_DIR / img_rel if not os.path.isabs(img_rel) else Path(img_rel)
                if not full_img_path.exists():
                    removed_count += 1
                    continue
            valid_rows.append(row)

    if removed_count > 0:
        print(f"[WARN] Removed {removed_count} records due to missing referenced files.")

    # Rewrite validated records
    if valid_rows:
        fieldnames = list(valid_rows[0].keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_rows)

    # Compute actual statistics
    total = len(valid_rows)
    real_count = sum(1 for r in valid_rows if str(r.get("label", "")).strip() in ["0", "REAL", "real"])
    fake_count = sum(1 for r in valid_rows if str(r.get("label", "")).strip() in ["1", "FAKE", "fake"])
    uncertain_count = sum(1 for r in valid_rows if str(r.get("label", "")).strip() in ["2", "UNCERTAIN", "uncertain"])

    train_count = sum(1 for r in valid_rows if str(r.get("split", "")).lower().strip() == "train")
    val_count = sum(1 for r in valid_rows if str(r.get("split", "")).lower().strip() in ["val", "validation", "valid"])
    test_count = sum(1 for r in valid_rows if str(r.get("split", "")).lower().strip() == "test")

    print("\nDataset Statistics")
    print("------------------")
    print(f"Total samples: {total}")
    print(f"REAL: {real_count}")
    print(f"FAKE: {fake_count}")
    if uncertain_count > 0:
        print(f"UNCERTAIN: {uncertain_count}")
    print(f"Train: {train_count}")
    print(f"Validation: {val_count}")
    print(f"Test: {test_count}")
    print("------------------\n")


def main():
    parser = argparse.ArgumentParser(description="TruthLens AI Data Preparation")
    parser.add_argument("--metadata", type=str, default=str(METADATA_DIR / "dataset.csv"), help="Path to metadata CSV")
    parser.add_argument("--resplit", action="store_true", help="Perform new stratified train/val/test split")

    args = parser.parse_args()
    meta_path = Path(args.metadata)

    if not meta_path.exists():
        print("[INFO] Metadata CSV not found. Generating sample development data first...")
        from scripts.download_dataset import create_development_samples
        create_development_samples()

    if args.resplit:
        print("[INFO] Recomputing stratified train/val/test splits...")
        split_metadata_csv(meta_path, meta_path)

    validate_and_report_metadata(meta_path)


if __name__ == "__main__":
    main()
