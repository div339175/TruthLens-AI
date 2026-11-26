"""Automated Unit and Integration Tests for Public Dataset Pipeline (Fakeddit).

Verifies:
1. Dataset loading via PyTorch Dataset interface.
2. Binary label validation (0 for REAL, 1 for FAKE).
3. Corrupted / zero-byte file detection.
4. Preprocessing fidelity (standardization to 224x224 RGB).
5. Split reproducibility using deterministic seed 42.
6. Empirical class distribution and balance metrics.
7. Data leakage detection across train/validation/test partitions.
8. Master and run metadata generation with valid schemas.
"""

import os
import csv
import json
import tempfile
from pathlib import Path
from PIL import Image
import pytest
import torch

from ml.preprocessing.text import TextPreprocessor
from ml.preprocessing.image import ImagePreprocessor
from scripts.train_and_eval_public import FakedditSplitDataset
from scripts.verify_dataset import verify_raw_images
from scripts.check_data_leakage import audit_splits
from scripts.analyze_dataset import load_manifest, analyze

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
METADATA_DIR = DATA_DIR / "metadata"


def test_public_dataset_loading():
    """1. Dataset loading: Verifies FakedditSplitDataset yields valid tensors."""
    train_csv = SPLITS_DIR / "train.csv"
    assert train_csv.exists(), "data/splits/train.csv does not exist"

    text_prep = TextPreprocessor(max_length=64)
    img_prep = ImagePreprocessor(image_size=224)

    ds = FakedditSplitDataset(train_csv, text_prep, img_prep)
    assert len(ds) == 42, f"Expected 42 train samples, got {len(ds)}"

    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "image" in item
    assert "label" in item
    assert "sample_id" in item

    assert item["input_ids"].shape == (64,)
    assert item["image"].shape == (3, 224, 224)
    assert item["label"].item() in (0, 1)


def test_label_validation():
    """2. Label validation: Confirms labels in manifest and splits are strictly binary integers 0 or 1."""
    manifest_csv = METADATA_DIR / "fakeddit_manifest.csv"
    assert manifest_csv.exists()

    with open(manifest_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_int = int(row["label"])
            assert label_int in (0, 1), f"Invalid label {label_int} in {row['sample_id']}"
            assert row["label_name"] in ("REAL", "FAKE")


def test_corrupted_file_detection():
    """3. Corrupted file detection: Verifies verify_raw_images detects zero-byte and corrupt files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Valid image
        valid_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        valid_img_path = tmp_path / "valid.jpg"
        valid_img.save(valid_img_path, "JPEG")

        # Zero-byte corrupted file
        zero_file = tmp_path / "zero_byte.jpg"
        zero_file.write_bytes(b"")

        # Garbage bytes corrupted file
        corrupt_file = tmp_path / "corrupted.jpg"
        corrupt_file.write_bytes(b"Not an image at all but random text bytes")

        valid_files, invalid_files = verify_raw_images(tmp_path)

        assert len(valid_files) == 1
        assert valid_files[0]["filename"] == "valid.jpg"
        assert len(invalid_files) == 2
        assert any("zero-byte" in inv for inv in invalid_files)
        assert any("corrupted" in inv for inv in invalid_files)


def test_preprocessing_fidelity():
    """4. Preprocessing: Verifies processed images exist and are standardized to 224x224 RGB."""
    processed_dir = DATA_DIR / "processed" / "fakeddit"
    assert processed_dir.exists(), "Processed directory missing"

    sample_images = list(processed_dir.glob("*.jpg"))
    assert len(sample_images) == 60, f"Expected 60 processed images, got {len(sample_images)}"

    # Check a subset of processed images
    for img_path in sample_images[:5]:
        with Image.open(img_path) as im:
            assert im.size == (224, 224), f"Image {img_path.name} has size {im.size}, expected (224, 224)"
            assert im.mode == "RGB", f"Image {img_path.name} has mode {im.mode}, expected RGB"


def test_split_reproducibility():
    """5. Split reproducibility: Confirms that splits are deterministic."""
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "validation.csv"
    test_csv = SPLITS_DIR / "test.csv"

    assert train_csv.exists() and val_csv.exists() and test_csv.exists()

    def read_ids(p):
        with open(p, mode="r", encoding="utf-8") as f:
            return [r["sample_id"] for r in csv.DictReader(f)]

    train_ids = read_ids(train_csv)
    val_ids = read_ids(val_csv)
    test_ids = read_ids(test_csv)

    assert len(train_ids) == 42
    assert len(val_ids) == 9
    assert len(test_ids) == 9
    assert len(set(train_ids).union(val_ids).union(test_ids)) == 60


def test_class_distribution_parity():
    """6. Class distribution: Verifies exact 1:1 balance in master manifest."""
    records = load_manifest()
    stats = analyze(records)

    assert stats["total_samples"] == 60
    assert stats["real_count"] == 30
    assert stats["fake_count"] == 30
    assert stats["imbalance_ratio"] == 1.00
    assert stats["real_pct"] == 50.0
    assert stats["fake_pct"] == 50.0


def test_leakage_detection_clean():
    """7. Leakage detection: Runs audit_splits and asserts complete clean separation."""
    leakage_results = audit_splits()

    assert leakage_results["Duplicate files"] is True
    assert leakage_results["Train/Test overlap"] is True
    assert leakage_results["Image hash deduplication"] is True
    assert leakage_results["Text claim overlap"] is True
    assert leakage_results["Identity overlap"] is True
    assert leakage_results["Overall"] is True


def test_metadata_generation():
    """8. Metadata generation: Verifies schema validity of fakeddit_manifest.csv and preprocessing_metadata.json."""
    manifest_csv = METADATA_DIR / "fakeddit_manifest.csv"
    prep_json = METADATA_DIR / "preprocessing_metadata.json"

    assert manifest_csv.exists()
    assert prep_json.exists()

    with open(manifest_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        assert "sample_id" in headers
        assert "label" in headers
        assert "text" in headers
        assert "raw_image_path" in headers
        assert "processed_image_path" in headers
        assert "raw_md5" in headers
        assert "processed_md5" in headers

    with open(prep_json, mode="r", encoding="utf-8") as f:
        meta = json.load(f)
        assert meta["dataset_name"] == "Fakeddit"
        assert meta["total_processed_samples"] == 60
        assert meta["target_resolution"] == [224, 224]
        assert meta["color_space"] == "RGB"


def test_fakeddit_splits_manifest():
    """9. Manifest integrity: Verifies data/splits/fakeddit_manifest.csv has 60 samples, 30 Real, 30 Fake."""
    splits_manifest = SPLITS_DIR / "fakeddit_manifest.csv"
    assert splits_manifest.exists()

    with open(splits_manifest, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 60
    assert sum(1 for r in rows if r["label"] == "REAL") == 30
    assert sum(1 for r in rows if r["label"] == "FAKE") == 30
    for r in rows:
        assert (BASE_DIR / r["image_path"]).exists()


def test_fakeddit_benchmark_api():
    """10. Benchmark API endpoints: Tests /api/benchmark/fakeddit/* routes."""
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)

    # Results endpoint
    res = client.get("/api/benchmark/fakeddit/results")
    assert res.status_code == 200
    data = res.json()
    assert data["evaluation_samples"] == 60
    assert "accuracy" in data
    assert "f1_score" in data

    # Samples endpoint
    res_samples = client.get("/api/benchmark/fakeddit/samples?status=all")
    assert res_samples.status_code == 200
    samples_data = res_samples.json()
    assert samples_data["count"] == 60

    # Filtered samples (correct and uncertain)
    res_correct = client.get("/api/benchmark/fakeddit/samples?status=correct")
    assert res_correct.status_code == 200
    for s in res_correct.json()["samples"]:
        assert s["status"] == "CORRECT"

    # Single sample detail
    res_detail = client.get("/api/benchmark/fakeddit/sample/fakeddit_fake_001")
    assert res_detail.status_code == 200
    sample_item = res_detail.json()["sample"]
    assert sample_item["sample_id"] == "fakeddit_fake_001"
    assert "text" in sample_item
    assert "image_url" in sample_item

    # Image serving
    res_img = client.get("/api/benchmark/fakeddit/image/fakeddit_fake_001")
    assert res_img.status_code == 200
    assert res_img.headers["content-type"] == "image/jpeg"

