#!/usr/bin/env python3
"""Preprocessing Pipeline for TruthLens AI Public Dataset (Fakeddit).

Performs reproducible, non-destructive preprocessing:
1. Validates raw input files from data/raw/fakeddit/.
2. Strips whitespace and normalizes text claims.
3. Standardizes labels to binary 0 (REAL) and 1 (FAKE).
4. Converts images to RGB and standardizes dimensions to 224x224.
5. Saves processed images into data/processed/fakeddit/ without modifying data/raw/.
6. Emits normalized master metadata into data/metadata/fakeddit_manifest.csv.
7. Saves preprocessing run metadata to data/metadata/preprocessing_metadata.json.
"""

import os
import sys
import csv
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "dataset_config.yaml"

# 6-way label mapping according to official Fakeddit taxonomy (LREC 2020)
TAXONOMY_MAP = {
    "fakeddit_class_0": "authentic",
    "fakeddit_class_1": "satire_parody",
    "fakeddit_class_2": "misleading_content",
    "fakeddit_class_3": "manipulated_content",
    "fakeddit_class_4": "false_connection_semantic_mismatch",
    "fakeddit_class_5": "imposter_content",
    "none": "authentic",
}


def compute_md5(file_path: Path) -> str:
    """Computes streaming MD5 hash."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_text(raw_text: str) -> str:
    """Normalizes whitespace and removes control characters."""
    if not raw_text:
        return ""
    # Collapse consecutive whitespace
    text = re.sub(r"\s+", " ", str(raw_text)).strip()
    return text


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "dataset": {
            "raw_dir": "data/raw",
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
            "metadata_dir": "data/metadata",
        },
        "modalities": {
            "visual": {"image_size": 224}
        }
    }


def main():
    print("=" * 60)
    print("TruthLens AI — Public Dataset Preprocessing Pipeline")
    print("=" * 60)

    cfg = load_config()
    raw_dir = BASE_DIR / cfg["dataset"].get("raw_dir", "data/raw")
    processed_dir = BASE_DIR / cfg["dataset"].get("processed_dir", "data/processed")
    metadata_dir = BASE_DIR / cfg["dataset"].get("metadata_dir", "data/metadata")
    interim_dir = BASE_DIR / cfg["dataset"].get("interim_dir", "data/interim")

    target_size = cfg.get("modalities", {}).get("visual", {}).get("image_size", 224)

    raw_fakeddit = raw_dir / "fakeddit"
    processed_fakeddit = processed_dir / "fakeddit"
    processed_fakeddit.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Reference manifest from experiments/external_dataset
    source_manifest_path = BASE_DIR / "experiments" / "external_dataset" / "external_dataset_manifest.csv"
    if not source_manifest_path.exists():
        print(f"[FAIL] Source manifest not found: {source_manifest_path}")
        return 1

    with open(source_manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        source_rows = [r for r in reader if r.get("source_dataset") == "fakeddit"]

    print(f"Loaded {len(source_rows)} source metadata records for Fakeddit.")

    processed_records: List[Dict[str, Any]] = []
    skipped_count = 0

    for idx, row in enumerate(source_rows, start=1):
        sample_id = row["sample_id"]
        raw_img_filename = f"{sample_id}.jpg"
        raw_img_path = raw_fakeddit / raw_img_filename

        if not raw_img_path.exists():
            print(f"[WARN] Raw file missing for {sample_id} at {raw_img_path}, skipping.")
            skipped_count += 1
            continue

        # 1. Clean Text
        normalized_text = clean_text(row.get("text", ""))

        # 2. Validate Label
        try:
            label_val = int(row.get("label", 0))
            assert label_val in (0, 1), f"Invalid binary label {label_val}"
        except Exception as e:
            print(f"[WARN] Invalid label for {sample_id}: {e}")
            skipped_count += 1
            continue

        # 3. Manipulation Taxonomy
        raw_manip = row.get("manipulation_type", "none")
        normalized_manip = TAXONOMY_MAP.get(raw_manip, raw_manip)

        # 4. Process Image (Non-destructive to raw)
        try:
            with Image.open(raw_img_path) as img:
                img_rgb = img.convert("RGB")
                orig_w, orig_h = img_rgb.size
                resized_img = img_rgb.resize((target_size, target_size), Image.Resampling.BILINEAR)

                out_img_filename = f"{sample_id}_224x224.jpg"
                out_img_path = processed_fakeddit / out_img_filename
                resized_img.save(out_img_path, "JPEG", quality=95)

            raw_md5 = compute_md5(raw_img_path)
            proc_md5 = compute_md5(out_img_path)

            record = {
                "sample_id": sample_id,
                "label": label_val,
                "label_name": "REAL" if label_val == 0 else "FAKE",
                "text": normalized_text,
                "raw_image_path": str(raw_img_path.relative_to(BASE_DIR)),
                "processed_image_path": str(out_img_path.relative_to(BASE_DIR)),
                "modality": "text+image",
                "source": "r/Fakeddit (LREC 2020)",
                "manipulation_type": normalized_manip,
                "original_width": orig_w,
                "original_height": orig_h,
                "processed_width": target_size,
                "processed_height": target_size,
                "raw_md5": raw_md5,
                "processed_md5": proc_md5,
            }
            processed_records.append(record)

        except Exception as e:
            print(f"[ERROR] Failed to process image for {sample_id}: {e}")
            skipped_count += 1
            continue

    print(f"\nSuccessfully processed: {len(processed_records)} records (Skipped: {skipped_count})")

    # 5. Save master processed metadata CSV
    out_csv = metadata_dir / "fakeddit_manifest.csv"
    fieldnames = [
        "sample_id",
        "label",
        "label_name",
        "text",
        "raw_image_path",
        "processed_image_path",
        "modality",
        "source",
        "manipulation_type",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "raw_md5",
        "processed_md5",
    ]
    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_records)
    print(f"Master manifest written to: {out_csv.relative_to(BASE_DIR)}")

    # 6. Save JSON format for inspection
    out_json = metadata_dir / "fakeddit_metadata.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=2)
    print(f"Metadata JSON written to: {out_json.relative_to(BASE_DIR)}")

    # 7. Save Preprocessing Run Metadata
    prep_meta = {
        "dataset_name": cfg["dataset"].get("name", "Fakeddit"),
        "total_processed_samples": len(processed_records),
        "target_resolution": [target_size, target_size],
        "color_space": "RGB",
        "resampling_filter": "BILINEAR",
        "jpeg_quality": 95,
        "normalization": cfg.get("modalities", {}).get("visual", {}).get("normalization", {}),
        "source_manifest": str(source_manifest_path.relative_to(BASE_DIR)),
        "master_manifest": str(out_csv.relative_to(BASE_DIR)),
    }
    prep_meta_path = metadata_dir / "preprocessing_metadata.json"
    with open(prep_meta_path, "w", encoding="utf-8") as f:
        json.dump(prep_meta, f, indent=2)
    print(f"Preprocessing metadata logged to: {prep_meta_path.relative_to(BASE_DIR)}")

    print("\n[PASS] Preprocessing pipeline completed successfully.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
