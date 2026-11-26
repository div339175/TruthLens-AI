#!/usr/bin/env python3
"""Verification Script for Fakeddit External Benchmark.

Verifies the integrity, completeness, and balance of the 60-sample
external Fakeddit generalization benchmark in data/splits/fakeddit_manifest.csv.
"""

import sys
import csv
import hashlib
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "splits" / "fakeddit_manifest.csv"


def compute_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_fakeddit():
    if not MANIFEST_PATH.exists():
        print(f"[FAIL] Manifest missing at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, mode="r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    total_samples = len(records)
    real_count = sum(1 for r in records if r["label"].upper() == "REAL")
    fake_count = sum(1 for r in records if r["label"].upper() == "FAKE")
    invalid_labels = sum(1 for r in records if r["label"].upper() not in ("REAL", "FAKE"))

    missing_text = sum(1 for r in records if not r.get("text", "").strip())
    missing_images = 0
    corrupted_images = 0

    seen_ids = set()
    duplicate_ids = 0

    seen_hashes = set()
    duplicate_images = 0

    for r in records:
        sid = r["sample_id"]
        if sid in seen_ids:
            duplicate_ids += 1
        seen_ids.add(sid)

        img_rel = r.get("image_path", "")
        img_path = BASE_DIR / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)

        if not img_path.exists():
            missing_images += 1
            continue

        try:
            with Image.open(img_path) as img:
                img.verify()
            file_hash = compute_md5(img_path)
            if file_hash in seen_hashes:
                duplicate_images += 1
            seen_hashes.add(file_hash)
        except Exception:
            corrupted_images += 1

    all_passed = (
        total_samples == 60 and
        real_count == 30 and
        fake_count == 30 and
        invalid_labels == 0 and
        missing_text == 0 and
        missing_images == 0 and
        corrupted_images == 0 and
        duplicate_ids == 0 and
        duplicate_images == 0
    )

    print("====================================")
    print("FAKEDIT EXTERNAL BENCHMARK CHECK")
    print("====================================")
    print()
    print(f"Total samples: {total_samples}")
    print(f"REAL: {real_count}")
    print(f"FAKE: {fake_count}")
    print()
    print(f"Missing text: {missing_text}")
    print(f"Missing images: {missing_images}")
    print(f"Corrupted images: {corrupted_images}")
    print(f"Duplicate IDs: {duplicate_ids}")
    print(f"Duplicate images: {duplicate_images}")
    print(f"Invalid labels: {invalid_labels}")
    print()
    print(f"Status: {'PASS' if all_passed else 'FAIL'}")
    print("====================================")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    verify_fakeddit()
