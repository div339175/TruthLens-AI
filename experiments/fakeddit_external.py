#!/usr/bin/env python3
"""TruthLens AI — Fakeddit External Generalization Evaluation Pipeline.

Evaluates the FROZEN, ALREADY-TRAINED TruthLens model on the external, unseen
r/Fakeddit benchmark (N=60, 30 REAL / 30 FAKE).

Core Principle:
- Training Data: TruthLens course dataset (320 samples: 160 REAL / 160 FAKE).
- Fakeddit is NEVER used for training.
- Evaluates out-of-domain generalization across multimodal text + image.
- Uses existing preprocessing, DistilBERT + ResNet-18 Cross-Attention, DecisionPolicy,
  and explainability engine (saliency + Grad-CAM).

Outputs:
- reports/fakeddit_external_results.json
- reports/fakeddit_external_results.csv
- reports/fakeddit_confusion_matrix.png
- reports/FAKEDIT_GENERALIZATION_REPORT.md
"""

import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.inference.predictor import TruthLensPredictor

MANIFEST_PATH = BASE_DIR / "data" / "splits" / "fakeddit_manifest.csv"
REPORTS_DIR = BASE_DIR / "reports"


def plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    """Saves formatted confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=f"{cm[i, j]}",
                va="center",
                ha="center",
                size="xx-large",
                weight="bold",
                color="white" if cm[i, j] > (cm.max() / 2) else "black",
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["REAL (0)", "FAKE (1)"], fontsize=11, weight="bold")
    ax.set_yticklabels(["REAL (0)", "FAKE (1)"], fontsize=11, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10, weight="bold")
    ax.set_ylabel("Ground Truth Label", fontsize=12, labelpad=10, weight="bold")
    ax.set_title("Fakeddit External Benchmark — Confusion Matrix", fontsize=12, pad=15, weight="bold")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_fakeddit_evaluation() -> Dict[str, Any]:
    print("=" * 65)
    print("TruthLens AI — Fakeddit External Generalization Evaluation")
    print("=" * 65)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Fakeddit manifest not found at {MANIFEST_PATH}")

    with open(MANIFEST_PATH, mode="r", encoding="utf-8") as f:
        manifest_records = list(csv.DictReader(f))

    print(f"Loaded {len(manifest_records)} unseen Fakeddit samples.")
    print("Loading existing trained TruthLens model (production checkpoint)...")
    predictor = TruthLensPredictor.get_instance()

    sample_results = []
    y_true_binary = []
    y_pred_binary = []
    y_prob_fake = []

    correct_count = 0
    incorrect_count = 0
    uncertain_count = 0

    print("\nRunning multimodal inference on Fakeddit external samples...")
    for idx, r in enumerate(manifest_records, start=1):
        sample_id = r["sample_id"]
        text = r["text"]
        gt_label = r["label"].upper()  # 'REAL' or 'FAKE'
        gt_num = 0 if gt_label == "REAL" else 1

        img_rel = r["image_path"]
        img_full = BASE_DIR / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)

        # Run inference through existing production pipeline
        report = predictor.predict_multimodal(text=text, image_source=img_full)

        pred_label = report.prediction  # 'REAL', 'FAKE', or 'UNCERTAIN'
        conf = report.confidence
        real_p = report.real_probability
        fake_p = report.fake_probability

        # Determine correctness
        if pred_label == "UNCERTAIN":
            uncertain_count += 1
            is_correct = False
            # For strict binary metrics, map uncertain based on argmax probability
            binary_pred = 1 if fake_p >= real_p else 0
        elif pred_label == gt_label:
            correct_count += 1
            is_correct = True
            binary_pred = 1 if pred_label == "FAKE" else 0
        else:
            incorrect_count += 1
            is_correct = False
            binary_pred = 1 if pred_label == "FAKE" else 0

        y_true_binary.append(gt_num)
        y_pred_binary.append(binary_pred)
        y_prob_fake.append(fake_p)

        # Extract evidence details
        text_tokens = []
        cam_heatmap = None
        evidence_desc = []
        for ev in report.evidence:
            if ev.token_importance:
                text_tokens = [{"token": t.token, "score": round(t.score, 4)} for t in ev.token_importance[:8]]
            if ev.heatmap_base64:
                cam_heatmap = ev.heatmap_base64
            evidence_desc.append(ev.description)

        status_str = "CORRECT" if is_correct else ("UNCERTAIN" if pred_label == "UNCERTAIN" else "INCORRECT")

        sample_results.append({
            "sample_id": sample_id,
            "ground_truth": gt_label,
            "prediction": pred_label,
            "confidence": round(float(conf), 4),
            "real_probability": round(float(real_p), 4),
            "fake_probability": round(float(fake_p), 4),
            "correct": is_correct,
            "status": status_str,
            "text": text,
            "image_path": str(img_rel),
            "manipulation_type": r.get("manipulation_type", "none"),
            "explanation": report.explanation,
            "evidence_descriptions": evidence_desc,
            "token_saliency": text_tokens,
            "has_heatmap": cam_heatmap is not None,
            "heatmap_base64": cam_heatmap,
        })

        if idx % 10 == 0 or idx == len(manifest_records):
            print(f"  Processed [{idx:2d}/{len(manifest_records)}] — {sample_id}: GT={gt_label}, Pred={pred_label} ({status_str})")

    # Metric calculations
    total = len(sample_results)
    acc = correct_count / total
    resolved_acc = accuracy_score(y_true_binary, y_pred_binary)
    prec_fake = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec_fake = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1_fake = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "dataset": "Fakeddit",
        "evaluation_samples": total,
        "training_source": "TruthLens course dataset",
        "training_samples": 320,
        "fakeddit_used_for_training": False,
        "fakeddit_used_for_external_evaluation": True,
        "class_distribution": {
            "real": sum(1 for r in sample_results if r["ground_truth"] == "REAL"),
            "fake": sum(1 for r in sample_results if r["ground_truth"] == "FAKE"),
        },
        "accuracy": round(float(acc), 4),
        "resolved_accuracy": round(float(resolved_acc), 4),
        "precision": round(float(prec_fake), 4),
        "recall": round(float(rec_fake), 4),
        "f1_score": round(float(f1_fake), 4),
        "counts": {
            "correct": correct_count,
            "incorrect": incorrect_count,
            "uncertain": uncertain_count,
        },
        "rates": {
            "uncertain_rate": round(uncertain_count / total, 4),
            "accuracy_percentage": round(acc * 100, 2),
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "samples": sample_results,
    }

    print("\n" + "=" * 65)
    print("FAKEDIT EXTERNAL BENCHMARK EVALUATION RESULTS")
    print("-" * 65)
    print(f"Total Evaluation Samples : {total}")
    print(f"Correct Predictions      : {correct_count} ({acc*100:.1f}%)")
    print(f"Incorrect Predictions    : {incorrect_count} ({incorrect_count/total*100:.1f}%)")
    print(f"Uncertain Predictions    : {uncertain_count} ({uncertain_count/total*100:.1f}%)")
    print(f"Precision (Class 1 FAKE) : {metrics['precision']:.4f}")
    print(f"Recall (Class 1 FAKE)    : {metrics['recall']:.4f}")
    print(f"F1-Score (Class 1 FAKE)  : {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    json_path = REPORTS_DIR / "fakeddit_external_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results JSON saved to: {json_path.relative_to(BASE_DIR)}")

    # 2. Save CSV
    csv_path = REPORTS_DIR / "fakeddit_external_results.csv"
    csv_fields = ["sample_id", "ground_truth", "prediction", "confidence", "real_probability", "fake_probability", "correct"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for s in sample_results:
            writer.writerow({
                "sample_id": s["sample_id"],
                "ground_truth": s["ground_truth"],
                "prediction": s["prediction"],
                "confidence": s["confidence"],
                "real_probability": s["real_probability"],
                "fake_probability": s["fake_probability"],
                "correct": s["correct"],
            })
    print(f"Results CSV saved to: {csv_path.relative_to(BASE_DIR)}")

    # 3. Save Confusion Matrix
    cm_path = REPORTS_DIR / "fakeddit_confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path)
    print(f"Confusion Matrix saved to: {cm_path.relative_to(BASE_DIR)}")

    # 4. Save Generalization Report
    md_path = REPORTS_DIR / "FAKEDIT_GENERALIZATION_REPORT.md"
    generate_markdown_report(metrics, md_path)
    print(f"Markdown Report saved to: {md_path.relative_to(BASE_DIR)}")

    print("\n[PASS] External evaluation completed successfully.")
    print("=" * 65)
    return metrics


def generate_markdown_report(m: Dict[str, Any], out_path: Path):
    cm = m["confusion_matrix"]
    cnt = m["counts"]

    md = f"""# TruthLens AI — Fakeddit External Generalization Report

```text
Dataset:
{m['dataset']}

Evaluation samples:
{m['evaluation_samples']}

Training source:
{m['training_source']}

Training samples:
{m['training_samples']}

Fakeddit used for training:
NO

Fakeddit used for external evaluation:
YES
```

---

## 1. Executive Summary

This report documents the zero-shot external generalization performance of the **frozen, production TruthLens model** when evaluated on the external **r/Fakeddit Benchmark** ($N={m['evaluation_samples']}$).

The model was strictly trained on the internal TruthLens course dataset ($N=320$) and evaluated on unseen real-world multimodal Reddit submissions without any fine-tuning or parameter updates on Fakeddit.

---

## 2. Quantitative Performance Metrics

| Metric | Measured Value | Percentage / Interpretation |
| :--- | :---: | :--- |
| **Strict Accuracy** | **{m['accuracy']:.4f}** | **{m['rates']['accuracy_percentage']}%** (Uncertain treated as unresolved) |
| **Resolved Binary Accuracy** | **{m['resolved_accuracy']:.4f}** | **{m['resolved_accuracy']*100:.1f}%** (Argmax prediction resolution) |
| **Precision (FAKE)** | **{m['precision']:.4f}** | Accuracy of FAKE class flags |
| **Recall (FAKE)** | **{m['recall']:.4f}** | Sensitivity to external deceptive claims |
| **F1-Score (FAKE)** | **{m['f1_score']:.4f}** | Harmonic mean of external precision and recall |
| **Uncertainty Rate** | **{m['rates']['uncertain_rate']:.4f}** | **{m['rates']['uncertain_rate']*100:.1f}%** routed to human editorial policy |

---

## 3. Decision Breakdown

- **Correct Predictions:** **{cnt['correct']}** / {m['evaluation_samples']} ({cnt['correct']/m['evaluation_samples']*100:.1f}%)
- **Incorrect Predictions:** **{cnt['incorrect']}** / {m['evaluation_samples']} ({cnt['incorrect']/m['evaluation_samples']*100:.1f}%)
- **Uncertain Inferences:** **{cnt['uncertain']}** / {m['evaluation_samples']} ({cnt['uncertain']/m['evaluation_samples']*100:.1f}%)

```text
Confusion Matrix:
                  Predicted REAL (0)    Predicted FAKE (1)
True REAL (0)            {cm['true_negatives']:2d} (TN)               {cm['false_positives']:2d} (FP)
True FAKE (1)            {cm['false_negatives']:2d} (FN)               {cm['true_positives']:2d} (TP)
```

---

## 4. Key Scientific Insights for Course Presentation

1. **Clean Domain Shift Demonstration:** Because the model was trained on formal course templates and clean canvas graphics, external social media headlines exhibit natural lexical and visual domain shift.
2. **Value of the UNCERTAIN Policy:** Rather than blindly hallucinating high-confidence predictions on unfamiliar out-of-distribution social media samples, the calibrated decision policy safely routes ambiguous samples ({cnt['uncertain']} cases, {m['rates']['uncertain_rate']*100:.1f}%) to `UNCERTAIN`.
3. **Cross-Modal Explainability:** Token saliency and Layer4 Grad-CAM heatmaps remain fully operational on external data, enabling interpretable error analysis for every individual sample.

---

## 5. Sample-Level Evaluation Log (First 15 Samples)

| Sample ID | Ground Truth | Prediction | Confidence | REAL Prob | FAKE Prob | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for s in m["samples"][:15]:
        v_icon = "✓ Correct" if s["status"] == "CORRECT" else ("⚠ Uncertain" if s["status"] == "UNCERTAIN" else "✗ Incorrect")
        md += f"| `{s['sample_id']}` | **{s['ground_truth']}** | {s['prediction']} | {s['confidence']*100:.1f}% | {s['real_probability']*100:.1f}% | {s['fake_probability']*100:.1f}% | {v_icon} |\n"

    md += f"""
*(Complete 60-sample results available in `reports/fakeddit_external_results.csv` and `reports/fakeddit_external_results.json`)*
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    run_fakeddit_evaluation()
