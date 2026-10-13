"""Dataset splitting utility implementing stratified partition without data leakage."""

import csv
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split


def split_metadata_csv(
    input_csv: Path,
    output_csv: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Dict[str, int]:
    """Reads input_csv, splits rows into train/val/test with stratification, and writes output_csv."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    rows: List[Dict[str, Any]] = []
    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Input metadata file {input_csv} is empty.")

    labels = [r.get("label", "0") for r in rows]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_rows, temp_rows, train_labels, temp_labels = train_test_split(
        rows,
        labels,
        test_size=val_test_ratio,
        stratify=labels if len(set(labels)) > 1 else None,
        random_state=random_state,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_rows, test_rows, _, _ = train_test_split(
        temp_rows,
        temp_labels,
        test_size=relative_test_ratio,
        stratify=temp_labels if len(set(temp_labels)) > 1 else None,
        random_state=random_state,
    )

    for r in train_rows:
        r["split"] = "train"
    for r in val_rows:
        r["split"] = "val"
    for r in test_rows:
        r["split"] = "test"

    all_split_rows = train_rows + val_rows + test_rows

    fieldnames = list(rows[0].keys())
    if "split" not in fieldnames:
        fieldnames.append("split")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_split_rows)

    return {
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "total": len(all_split_rows),
    }
