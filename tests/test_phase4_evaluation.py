"""Tests for Phase 4 Experimental Evaluation, Metrics, Baselines, and Ablation Framework."""

import os
import json
import pytest
import numpy as np
import torch
from pathlib import Path

from experiments.metrics import compute_comprehensive_metrics
from experiments.baselines import evaluate_text_baseline, evaluate_video_baseline, evaluate_audio_baseline
from experiments.fusion_comparison import evaluate_multimodal_variant
from experiments.ablation import run_ablation_study
from experiments.dataset_audit import audit_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "experiments" / "reports"
MODELS_DIR = BASE_DIR / "models"


# 1. Metric Calculation
def test_compute_comprehensive_metrics_accuracy_and_f1():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_prob = [0.1, 0.8, 0.4, 0.9]

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob)

    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 0.5  # 2 out of 4 correct
    assert metrics["precision"] == 0.5  # 1 TP / 2 predicted positive
    assert metrics["recall"] == 0.5     # 1 TP / 2 actual positive
    assert metrics["f1"] == 0.5
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None


# 2. Confusion Matrix Generation
def test_confusion_matrix_structure():
    # 2 TN (0->0), 1 FP (0->1), 1 FN (1->0), 2 TP (1->1)
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]

    metrics = compute_comprehensive_metrics(y_true, y_pred)
    cm = metrics["confusion_matrix"]

    assert cm["tn"] == 2
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 2
    assert cm["matrix"] == [[2, 1], [1, 2]]


# 3. Class-Wise Metrics and Support
def test_class_wise_metrics_and_support():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 1]

    metrics = compute_comprehensive_metrics(y_true, y_pred)
    class_metrics = metrics["class_metrics"]

    assert "REAL (Class 0)" in class_metrics
    assert "FAKE (Class 1)" in class_metrics

    real = class_metrics["REAL (Class 0)"]
    fake = class_metrics["FAKE (Class 1)"]

    assert real["support"] == 3
    assert fake["support"] == 2
    assert real["recall"] == round(2 / 3, 4)
    assert fake["recall"] == 1.0


# 4. Length Validation
def test_metric_calculation_length_mismatch_raises_error():
    y_true = [0, 1, 0]
    y_pred = [0, 1]

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_comprehensive_metrics(y_true, y_pred)


# 5. Empty Input Handling
def test_empty_metric_handling():
    metrics = compute_comprehensive_metrics([], [])
    assert metrics["sample_count"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["roc_auc"] is None


# 6. Reproducibility Seed
def test_reproducibility_seed_deterministic_outputs():
    device = torch.device("cpu")
    # Evaluate twice on test split
    res1 = evaluate_text_baseline(device, split="test")
    res2 = evaluate_text_baseline(device, split="test")

    preds1 = res1["raw_predictions"]["y_pred"]
    preds2 = res2["raw_predictions"]["y_pred"]
    assert preds1 == preds2, "Predictions differed across deterministic runs with same seed!"

    probs1 = res1["raw_predictions"]["y_prob"]
    probs2 = res2["raw_predictions"]["y_prob"]
    assert probs1 == probs2, "Probabilities differed across deterministic runs with same seed!"


# 7. Unavailable Modality Handling (No Fabricated Results)
def test_unavailable_modality_handling():
    device = torch.device("cpu")
    audio_res = evaluate_audio_baseline(device)
    video_res = evaluate_video_baseline(device)

    assert audio_res["status"] == "NOT_AVAILABLE"
    assert audio_res["metrics"] is None
    assert "0 labeled audio" in audio_res["reason"]

    assert video_res["status"] == "TECHNICAL_VERIFICATION_ONLY"
    assert video_res["metrics"] is None
    assert "0 labeled video" in video_res["reason"]


# 8. Result Serialization and File Validity
def test_result_serialization_json_validity():
    audit_res = audit_dataset()
    assert audit_res["total_samples"] >= 300
    assert audit_res["class_balance"]["real"] > 0
    assert audit_res["class_balance"]["fake"] > 0

    json_path = REPORTS_DIR / "dataset_audit.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_samples"] == audit_res["total_samples"]


# 9. Ablation Configuration and Delta Metrics
def test_ablation_configuration_deltas():
    device = torch.device("cpu")
    res = run_ablation_study(device_str="cpu", split="test")

    assert "full_model" in res
    assert "without_text" in res
    assert "without_visual" in res
    assert "without_audio" in res

    assert res["without_audio"]["status"] == "NOT_AVAILABLE"
    assert res["without_audio"]["metrics"] is None

    assert isinstance(res["full_model"]["delta_f1"], float)
    assert isinstance(res["without_text"]["delta_f1"], float)
    assert isinstance(res["without_visual"]["delta_f1"], float)
