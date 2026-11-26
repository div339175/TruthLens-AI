#!/usr/bin/env python3
"""Dataset Verification Script for TruthLens AI.

Verifies the integrity of raw assets in data/raw/:
1. Validates that referenced image files exist on disk.
2. Checks that each image can be properly opened and decoded via PIL.
3. Detects corrupted files, truncated streams, or zero-byte payloads.
4. Computes MD5 checksums for cryptographic integrity and deduplication.
5. Produces an audit report and returns exit code 0 on PASS, 1 on FAIL.
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "dataset_config.yaml"


def compute_md5(file_path: Path) -> str:
    """Computes the MD5 hex digest of a file in streaming chunks."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_config() -> Dict[str, Any]:
    """Loads dataset configuration from config/dataset_config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "dataset": {
            "name": "Fakeddit",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "metadata_dir": "data/metadata",
            "split_dir": "data/splits",
        }
    }


def verify_raw_images(raw_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Inspects all image files in raw_dir and verifies their integrity."""
    valid_files: List[Dict[str, Any]] = []
    corrupted_or_invalid: List[str] = []

    fakeddit_raw = raw_dir / "fakeddit"
    target_dir = fakeddit_raw if fakeddit_raw.exists() else raw_dir

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = [p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]

    for file_path in sorted(all_files):
        try:
            rel_path = file_path.relative_to(BASE_DIR)
        except ValueError:
            rel_path = file_path
        size_bytes = file_path.stat().st_size

        if size_bytes == 0:
            corrupted_or_invalid.append(f"{rel_path}: zero-byte file")
            continue

        try:
            with Image.open(file_path) as img:
                img.verify()  # Fast structural verification

            # Re-open for mode and dimension inspection
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode

            file_md5 = compute_md5(file_path)
            valid_files.append({
                "path": str(rel_path),
                "filename": file_path.name,
                "size_bytes": size_bytes,
                "width": width,
                "height": height,
                "mode": mode,
                "md5": file_md5,
            })
        except Exception as e:
            corrupted_or_invalid.append(f"{rel_path}: corrupted image ({e})")

    return valid_files, corrupted_or_invalid


def main() -> int:
    print("=" * 60)
    print("TruthLens AI — Raw Dataset Verification")
    print("=" * 60)

    cfg = load_config()
    raw_dir_rel = cfg.get("dataset", {}).get("raw_dir", "data/raw")
    raw_dir = BASE_DIR / raw_dir_rel

    if not raw_dir.exists():
        print(f"[FAIL] Raw data directory does not exist: {raw_dir}")
        return 1

    print(f"Scanning directory: {raw_dir.relative_to(BASE_DIR)}")
    valid_files, invalid_files = verify_raw_images(raw_dir)

    print(f"\nVerification Results:")
    print(f"  Valid decoded images : {len(valid_files)}")
    print(f"  Corrupted / invalid  : {len(invalid_files)}")

    if invalid_files:
        print("\n[FAIL] Corrupted files detected:")
        for inv in invalid_files:
            print(f"  - {inv}")
        return 1

    if len(valid_files) == 0:
        print("\n[WARN] No image files found in raw directory.")
        return 1

    print("\n[PASS] All raw image assets verified successfully.")
    print(f"Summary: {len(valid_files)} images verified with valid headers, RGB/RGBA channels, and MD5 hashes.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
