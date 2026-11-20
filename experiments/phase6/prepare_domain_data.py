"""Domain-Adaptive Training Data Preparation for TruthLens AI (Phase 6).

Fetches disjoint Fakeddit train and val text splits from ams-99/fakeddit_9k,
merges with in-domain training text to prevent catastrophic forgetting,
and strictly verifies ZERO leakage against the Phase 5 external test set.
"""

import io
import csv
import random
from pathlib import Path
from typing import Dict, Any, List
import requests
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "experiments" / "phase6" / "data"
PHASE5_MANIFEST = BASE_DIR / "experiments" / "external_dataset" / "external_dataset_manifest.csv"
COURSE_DATASET = BASE_DIR / "ml" / "data" / "metadata" / "dataset.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 TruthLens/1.0"}


def prepare_domain_adaptive_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("PREPARING PHASE 6 DOMAIN-ADAPTIVE TEXT DATASETS")
    print("=" * 70)

    # 1. Load Phase 5 Test Set for Strict Leakage Check
    p5_df = pd.read_csv(PHASE5_MANIFEST)
    test_texts = set(p5_df["text"].dropna().str.lower().str.strip())
    print(f"[AUDIT] Phase 5 Test Texts to protect: {len(test_texts)}")

    # 2. Ingest Fakeddit Training Text (300 samples: 150 Real, 150 Fake)
    print("\n[INFO] Fetching Fakeddit train split...")
    url_train = "https://huggingface.co/datasets/ams-99/fakeddit_9k/raw/main/train_balanced.csv"
    r_tr = requests.get(url_train, headers=HEADERS, timeout=15)
    r_tr.raise_for_status()

    reader_tr = csv.DictReader(io.StringIO(r_tr.text))
    rows_tr = list(reader_tr)

    rng_tr = random.Random(42)
    rng_tr.shuffle(rows_tr)

    train_recs = []
    tr_real, tr_fake = 0, 0
    for row in rows_tr:
        text = str(row.get("text", "")).strip()
        label_6way = int(row.get("6_way_label", 0))
        if not text or len(text) < 5:
            continue

        # Strict check against Phase 5 test texts
        if text.lower().strip() in test_texts:
            continue

        is_real = (label_6way == 0)
        target_label = 0 if is_real else 1

        if is_real and tr_real < 150:
            train_recs.append({
                "sample_id": f"fakeddit_train_real_{tr_real+1:03d}",
                "text": text,
                "label": target_label,
                "source": "fakeddit_train",
            })
            tr_real += 1
        elif not is_real and tr_fake < 150:
            train_recs.append({
                "sample_id": f"fakeddit_train_fake_{tr_fake+1:03d}",
                "text": text,
                "label": target_label,
                "source": "fakeddit_train",
            })
            tr_fake += 1

        if tr_real >= 150 and tr_fake >= 150:
            break

    print(f"[+] Selected Fakeddit Train: {len(train_recs)} samples ({tr_real} Real, {tr_fake} Fake)")

    # 3. Ingest Fakeddit Validation Text (100 samples: 50 Real, 50 Fake)
    print("\n[INFO] Fetching Fakeddit val split...")
    url_val = "https://huggingface.co/datasets/ams-99/fakeddit_9k/raw/main/val_balanced.csv"
    r_val = requests.get(url_val, headers=HEADERS, timeout=15)
    r_val.raise_for_status()

    reader_val = csv.DictReader(io.StringIO(r_val.text))
    rows_val = list(reader_val)

    rng_val = random.Random(42)
    rng_val.shuffle(rows_val)

    val_recs = []
    val_real, val_fake = 0, 0
    train_texts = {r["text"].lower().strip() for r in train_recs}

    for row in rows_val:
        text = str(row.get("text", "")).strip()
        label_6way = int(row.get("6_way_label", 0))
        if not text or len(text) < 5:
            continue

        # Strict check against Phase 5 test texts and train texts
        if text.lower().strip() in test_texts or text.lower().strip() in train_texts:
            continue

        is_real = (label_6way == 0)
        target_label = 0 if is_real else 1

        if is_real and val_real < 50:
            val_recs.append({
                "sample_id": f"fakeddit_val_real_{val_real+1:03d}",
                "text": text,
                "label": target_label,
                "source": "fakeddit_val",
            })
            val_real += 1
        elif not is_real and val_fake < 50:
            val_recs.append({
                "sample_id": f"fakeddit_val_fake_{val_fake+1:03d}",
                "text": text,
                "label": target_label,
                "source": "fakeddit_val",
            })
            val_fake += 1

        if val_real >= 50 and val_fake >= 50:
            break

    print(f"[+] Selected Fakeddit Val: {len(val_recs)} samples ({val_real} Real, {val_fake} Fake)")

    # 4. Ingest Course In-Domain Train and Val for Rehearsal
    df_course = pd.read_csv(COURSE_DATASET)
    course_train = df_course[df_course["split"] == "train"]
    course_val = df_course[df_course["split"] == "val"]

    rehearsal_train = [
        {"sample_id": row["id"], "text": row["text"], "label": int(row["label"]), "source": "course_in_domain"}
        for _, row in course_train.iterrows()
    ]
    rehearsal_val = [
        {"sample_id": row["id"], "text": row["text"], "label": int(row["label"]), "source": "course_in_domain"}
        for _, row in course_val.iterrows()
    ]

    # Combine
    combined_train = train_recs + rehearsal_train
    all_train_texts = {r["text"].lower().strip() for r in combined_train}

    # Ensure validation is strictly disjoint from all training texts
    combined_val = [r for r in (val_recs + rehearsal_val) if r["text"].lower().strip() not in all_train_texts]

    # Shuffle deterministically
    random.Random(42).shuffle(combined_train)
    random.Random(42).shuffle(combined_val)

    # Save to disk
    train_csv = DATA_DIR / "domain_adaptive_train.csv"
    val_csv = DATA_DIR / "domain_adaptive_val.csv"

    pd.DataFrame(combined_train).to_csv(train_csv, index=False)
    pd.DataFrame(combined_val).to_csv(val_csv, index=False)

    print(f"\n[SUCCESS] Saved Combined Train: {len(combined_train)} samples -> {train_csv}")
    print(f"[SUCCESS] Saved Combined Val:   {len(combined_val)} samples -> {val_csv}")

    # Final Verification of 0.0% leakage
    all_train_texts = {r["text"].lower().strip() for r in combined_train}
    all_val_texts = {r["text"].lower().strip() for r in combined_val}

    leakage_train_test = all_train_texts.intersection(test_texts)
    leakage_val_test = all_val_texts.intersection(test_texts)
    leakage_train_val = all_train_texts.intersection(all_val_texts)

    print("\n--- ZERO-LEAKAGE AUDIT ---")
    print(f"Leakage Train <-> Test: {len(leakage_train_test)} (Expected: 0)")
    print(f"Leakage Val <-> Test:   {len(leakage_val_test)} (Expected: 0)")
    print(f"Leakage Train <-> Val:  {len(leakage_train_val)} (Expected: 0)")

    assert len(leakage_train_test) == 0, "DATA LEAKAGE DETECTED IN TRAIN!"
    assert len(leakage_val_test) == 0, "DATA LEAKAGE DETECTED IN VAL!"
    assert len(leakage_train_val) == 0, "DATA LEAKAGE DETECTED TRAIN <-> VAL!"
    print("VERIFICATION PASS: 0.00% DATA LEAKAGE GUARANTEED")
    print("=" * 70)


if __name__ == "__main__":
    prepare_domain_adaptive_data()
