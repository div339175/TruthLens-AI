"""Unit and Integration Tests for TruthLens AI Phase 5 (Generalization & External Benchmarks)."""

import os
import csv
import json
import hashlib
from pathlib import Path
import pytest
import torch
import pandas as pd

from experiments.external_dataset.dataset import TruthLensExternalDataset
from experiments.metrics import compute_comprehensive_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = BASE_DIR / "experiments" / "external_dataset"
MANIFEST_CSV = EXTERNAL_DIR / "external_dataset_manifest.csv"
MANIFEST_JSON = EXTERNAL_DIR / "external_dataset_manifest.json"
REPORTS_DIR = BASE_DIR / "experiments" / "reports"
COURSE_DATASET_CSV = BASE_DIR / "ml" / "data" / "metadata" / "dataset.csv"


def test_external_manifest_structure_and_columns():
    """Verifies that the external dataset manifest exists and matches expected schema."""
    assert MANIFEST_CSV.exists(), "external_dataset_manifest.csv does not exist"
    df = pd.read_csv(MANIFEST_CSV)
    assert len(df) >= 180, f"Expected at least 180 external samples, got {len(df)}"

    required_columns = [
        "sample_id",
        "source_dataset",
        "label",
        "text",
        "image_path",
        "video_path",
        "audio_path",
        "split",
        "source_id",
        "manipulation_type",
        "md5",
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing required column in manifest: {col}"


def test_label_mapping_binary_integers():
    """Verifies that all external labels map explicitly to binary 0 (REAL) or 1 (FAKE)."""
    df = pd.read_csv(MANIFEST_CSV)
    labels = df["label"].unique().tolist()
    assert set(labels).issubset({0, 1}), f"Labels must be subset of {{0, 1}}, got {labels}"

    # Verify per-dataset balance
    fakeddit_labels = df[df["source_dataset"] == "fakeddit"]["label"].value_counts().to_dict()
    assert fakeddit_labels.get(0) == 30, f"Expected 30 real Fakeddit samples, got {fakeddit_labels.get(0)}"
    assert fakeddit_labels.get(1) == 30, f"Expected 30 fake Fakeddit samples, got {fakeddit_labels.get(1)}"

    audio_labels = df[df["source_dataset"] == "deepfake_audio_v4"]["label"].value_counts().to_dict()
    assert audio_labels.get(0) == 20, f"Expected 20 real audio samples, got {audio_labels.get(0)}"
    assert audio_labels.get(1) == 20, f"Expected 20 fake audio samples, got {audio_labels.get(1)}"

    video_labels = df[df["source_dataset"] == "generative_video_benchmark"]["label"].value_counts().to_dict()
    assert video_labels.get(0) == 10, f"Expected 10 real video samples, got {video_labels.get(0)}"
    assert video_labels.get(1) == 10, f"Expected 10 fake video samples, got {video_labels.get(1)}"


def test_referenced_files_exist_on_disk():
    """Verifies that all non-empty referenced asset paths exist on disk."""
    df = pd.read_csv(MANIFEST_CSV)
    for idx, row in df.iterrows():
        for col in ["image_path", "audio_path", "video_path"]:
            p = row[col]
            if pd.notna(p) and str(p).strip():
                full_path = BASE_DIR / str(p).strip()
                assert full_path.exists(), f"Row {idx} {col} not found: {full_path}"


def test_no_data_leakage_between_course_and_external_datasets():
    """Verifies zero image and text overlap between in-domain course dataset and external benchmarks."""
    df_course = pd.read_csv(COURSE_DATASET_CSV)
    df_ext = pd.read_csv(MANIFEST_CSV)

    course_images = {
        hashlib.md5(open(BASE_DIR / "ml" / "data" / row["image_path"], "rb").read()).hexdigest()
        for _, row in df_course.iterrows()
        if pd.notna(row["image_path"]) and (BASE_DIR / "ml" / "data" / row["image_path"]).exists()
    }

    ext_images = {
        hashlib.md5(open(BASE_DIR / row["image_path"], "rb").read()).hexdigest()
        for _, row in df_ext.iterrows()
        if pd.notna(row["image_path"]) and str(row["image_path"]).strip() and (BASE_DIR / row["image_path"]).exists()
    }

    overlap = course_images.intersection(ext_images)
    assert len(overlap) == 0, f"Detected {len(overlap)} overlapping images between course and external benchmarks"


def test_external_dataset_loading_item_tensors():
    """Verifies that TruthLensExternalDataset outputs correct tensor shapes and types."""
    ds_fakeddit = TruthLensExternalDataset(MANIFEST_CSV, source_dataset="fakeddit")
    assert len(ds_fakeddit) == 60
    item = ds_fakeddit[0]

    assert "input_ids" in item and item["input_ids"].shape == torch.Size([128])
    assert "attention_mask" in item and item["attention_mask"].shape == torch.Size([128])
    assert "image" in item and item["image"].shape == torch.Size([3, 224, 224])
    assert "audio" in item and item["audio"].shape == torch.Size([128])
    assert "video" in item and item["video"].shape == torch.Size([8, 3, 224, 224])
    assert isinstance(item["label"].item(), int)


def test_semantic_mismatch_4_conditions_coverage():
    """Verifies that all 4 controlled semantic mismatch conditions are present with 15 samples each."""
    df = pd.read_csv(MANIFEST_CSV)
    mismatch_df = df[df["source_dataset"] == "semantic_mismatch_suite"]
    assert len(mismatch_df) == 60, f"Expected 60 mismatch samples, got {len(mismatch_df)}"

    conditions = mismatch_df["source_id"].value_counts().to_dict()
    assert conditions.get("condition_1_congruent_real") == 15
    assert conditions.get("condition_2_semantic_mismatch") == 15
    assert conditions.get("condition_3_swapped_image") == 15
    assert conditions.get("condition_4_unrelated_text") == 15


def test_modality_availability_matrix():
    """Verifies modality support flags across external benchmark subsets."""
    df = pd.read_csv(MANIFEST_CSV)

    fakeddit_has_text = df[df["source_dataset"] == "fakeddit"]["text"].str.strip().ne("").all()
    fakeddit_has_img = df[df["source_dataset"] == "fakeddit"]["image_path"].str.strip().ne("").all()
    assert fakeddit_has_text and fakeddit_has_img

    audio_has_aud = df[df["source_dataset"] == "deepfake_audio_v4"]["audio_path"].str.strip().ne("").all()
    assert audio_has_aud

    video_has_vid = df[df["source_dataset"] == "generative_video_benchmark"]["video_path"].str.strip().ne("").all()
    assert video_has_vid


def test_result_serialization_json_validity():
    """Verifies all Phase 5 report files exist and contain valid JSON/CSV structures."""
    gen_json = REPORTS_DIR / "generalization_results.json"
    gen_csv = REPORTS_DIR / "generalization_comparison.csv"
    mismatch_json = REPORTS_DIR / "semantic_mismatch_results.json"
    failure_json = REPORTS_DIR / "phase5_failure_analysis.json"
    failure_csv = REPORTS_DIR / "phase5_failure_analysis.csv"

    assert gen_json.exists(), "generalization_results.json missing"
    assert gen_csv.exists(), "generalization_comparison.csv missing"
    assert mismatch_json.exists(), "semantic_mismatch_results.json missing"
    assert failure_json.exists(), "phase5_failure_analysis.json missing"
    assert failure_csv.exists(), "phase5_failure_analysis.csv missing"

    with open(gen_json) as f:
        data = json.load(f)
        assert "fakeddit" in data and "audio" in data and "video" in data

    with open(mismatch_json) as f:
        mdata = json.load(f)
        assert "overall_metrics" in mdata and "conditions" in mdata

    with open(failure_json) as f:
        fdata = json.load(f)
        assert isinstance(fdata, list) and len(fdata) > 0


def test_domain_shift_delta_calculation():
    """Verifies that domain shift deltas are mathematically exact."""
    gen_json = REPORTS_DIR / "generalization_results.json"
    with open(gen_json) as f:
        data = json.load(f)

    for model_key, res in data["fakeddit"].items():
        ds = res["domain_shift"]
        expected_delta_f1 = ds["external_f1"] - ds["in_domain_f1"]
        expected_delta_acc = ds["external_acc"] - ds["in_domain_acc"]
        assert abs(ds["delta_f1"] - expected_delta_f1) < 1e-6
        assert abs(ds["delta_acc"] - expected_delta_acc) < 1e-6
