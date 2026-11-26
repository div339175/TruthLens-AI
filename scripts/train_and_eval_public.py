#!/usr/bin/env python3
"""Training and Academic Evaluation Pipeline for TruthLens AI (Fakeddit Benchmark).

Adapts the multimodal cross-attention architecture to the public Fakeddit dataset:
1. Loads train/validation/test splits from data/splits/.
2. Fine-tunes the Cross-Attention fusion layer using DistilBERT and ResNet-18.
3. Evaluates on the held-out test split (test.csv).
4. Generates standard classification metrics:
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Confusion Matrix (TN, FP, FN, TP)
   - False Positive Rate (FPR), False Negative Rate (FNR)
5. Exports reports and visualizations:
   - reports/evaluation_report.md
   - reports/confusion_matrix.png
   - reports/roc_curve.png
   - reports/metrics.json
"""

import os
import sys
import csv
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.preprocessing.text import TextPreprocessor
from ml.preprocessing.image import ImagePreprocessor
from ml.models.multimodal_model import MultimodalVerificationModel

CONFIG_PATH = BASE_DIR / "config" / "dataset_config.yaml"
SPLITS_DIR = BASE_DIR / "data" / "splits"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"


class FakedditSplitDataset(Dataset):
    """PyTorch Dataset loading processed Fakeddit text claims and 224x224 images."""

    def __init__(self, csv_path: Path, text_preprocessor: TextPreprocessor, image_preprocessor: ImagePreprocessor):
        self.csv_path = Path(csv_path)
        self.text_preprocessor = text_preprocessor
        self.image_preprocessor = image_preprocessor
        self.records: List[Dict[str, Any]] = []

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.records = list(reader)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        text = item.get("text", "")
        img_rel = item.get("processed_image_path") or item.get("raw_image_path", "")
        img_full = BASE_DIR / img_rel if not os.path.isabs(img_rel) else Path(img_rel)
        label = int(item.get("label", 0))

        # 1. Text tokenization
        encoded = self.text_preprocessor.tokenize(text)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # 2. Image preprocessing
        if img_full.exists():
            img_tensor = self.image_preprocessor.preprocess(img_full)
        else:
            blank = Image.new("RGB", (224, 224), color=(128, 128, 128))
            img_tensor = self.image_preprocessor.preprocess(blank)

        return {
            "sample_id": item.get("sample_id", f"sample_{idx}"),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": text,
        }


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "training": {
            "batch_size": 6,
            "epochs": 5,
            "learning_rate": 2.0e-5,
            "device": "auto",
        },
        "seed": 42,
    }


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    """Saves high-resolution confusion matrix plot."""
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
    ax.set_ylabel("True Ground Truth", fontsize=12, labelpad=10, weight="bold")
    ax.set_title("Fakeddit Test Set — Confusion Matrix", fontsize=13, pad=15, weight="bold")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curve(y_true: List[int], y_prob: List[float], auc_val: float, out_path: Path):
    """Saves ROC-AUC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"Cross-Attention (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, weight="bold")
    ax.set_title("Fakeddit Test Set — ROC-AUC Curve", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    print("=" * 65)
    print("TruthLens AI — Public Dataset Training & Academic Evaluation")
    print("=" * 65)

    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Determine device
    device_cfg = cfg.get("training", {}).get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(device_cfg)
    print(f"Using Compute Device: {device}")

    # File paths
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "validation.csv"
    test_csv = SPLITS_DIR / "test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            print(f"[FAIL] Split file not found: {p}")
            print("Please run scripts/split_dataset.py first.")
            return 1

    # Load preprocessors
    text_prep = TextPreprocessor(model_name=cfg.get("modalities", {}).get("text", {}).get("model_name", "distilbert-base-uncased"))
    img_prep = ImagePreprocessor(image_size=cfg.get("modalities", {}).get("visual", {}).get("image_size", 224))

    # Datasets & Loaders
    batch_size = int(cfg.get("training", {}).get("batch_size", 6))
    train_ds = FakedditSplitDataset(train_csv, text_prep, img_prep)
    val_ds = FakedditSplitDataset(val_csv, text_prep, img_prep)
    test_ds = FakedditSplitDataset(test_csv, text_prep, img_prep)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Data Partitions: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Initialize model
    print("\nInitializing Multimodal Verification Model (fusion=cross_attention)...")
    model = MultimodalVerificationModel(
        text_model_name="distilbert-base-uncased",
        vision_model_name="resnet18",
        projection_dim=256,
        fusion_type="cross_attention",
        pretrained_vision=True,
    ).to(device)

    # Load pre-existing checkpoint weights if available
    base_ckpt = BASE_DIR / "models" / "production" / "selected_model.pt"
    if not base_ckpt.exists():
        base_ckpt = BASE_DIR / "models" / "multimodal_cross_attention.pt"

    if base_ckpt.exists():
        try:
            state = torch.load(base_ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"Loaded warm-start weights from: {base_ckpt.relative_to(BASE_DIR)}")
        except Exception as e:
            print(f"Note: proceeding with standard pretrained backbones ({e})")

    # Hyperparameters
    epochs = int(cfg.get("training", {}).get("epochs", 5))
    lr = float(cfg.get("training", {}).get("learning_rate", 2.0e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None

    print(f"\nBeginning Fine-Tuning ({epochs} epochs, lr={lr}, batch_size={batch_size})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            b_ids = batch["input_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            b_imgs = batch["image"].to(device)
            b_labels = batch["label"].to(device)

            out = model(input_ids=b_ids, attention_mask=b_mask, images=b_imgs)
            loss = criterion(out["logits"], b_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation pass
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                b_ids = batch["input_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                b_imgs = batch["image"].to(device)
                b_labels = batch["label"].to(device)

                out = model(input_ids=b_ids, attention_mask=b_mask, images=b_imgs)
                preds = torch.argmax(out["logits"], dim=-1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(b_labels.cpu().tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        print(f"  Epoch [{epoch}/{epochs}] — Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict()

    # Save fine-tuned checkpoint
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_model_path = BASE_DIR / cfg.get("training", {}).get("output_model_path", "models/fakeddit_finetuned.pt")
    if best_weights is not None:
        model.load_state_dict(best_weights)
        torch.save(best_weights, out_model_path)
        print(f"\nSaved best model checkpoint to: {out_model_path.relative_to(BASE_DIR)}")

    # ----------------------------------------------------
    # Held-Out Test Evaluation
    # ----------------------------------------------------
    print("\nEvaluating on Held-Out Test Split (test.csv)...")
    model.eval()
    test_preds, test_targets, test_probs = [], [], []
    sample_results = []

    with torch.no_grad():
        for batch in test_loader:
            b_ids = batch["input_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            b_imgs = batch["image"].to(device)
            b_labels = batch["label"].to(device)

            out = model(input_ids=b_ids, attention_mask=b_mask, images=b_imgs)
            probs = torch.softmax(out["logits"], dim=-1)
            fake_probs = probs[:, 1].cpu().tolist()
            preds = torch.argmax(out["logits"], dim=-1).cpu().tolist()
            targets = b_labels.cpu().tolist()

            test_preds.extend(preds)
            test_targets.extend(targets)
            test_probs.extend(fake_probs)

            for s_id, t_text, t_target, t_pred, t_p in zip(
                batch["sample_id"], batch["raw_text"], targets, preds, fake_probs
            ):
                sample_results.append({
                    "sample_id": s_id,
                    "text": t_text[:60] + "..." if len(t_text) > 60 else t_text,
                    "true_label": "REAL" if t_target == 0 else "FAKE",
                    "predicted_label": "REAL" if t_pred == 0 else "FAKE",
                    "fake_probability": round(t_p, 4),
                    "is_correct": bool(t_target == t_pred),
                })

    # Compute comprehensive metrics
    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds, zero_division=0)
    rec = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)

    try:
        auc = roc_auc_score(test_targets, test_probs)
    except Exception:
        auc = 0.50

    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "dataset": "r/Fakeddit (LREC 2020)",
        "split": "test",
        "num_test_samples": len(test_targets),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "rates": {
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
        },
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "seed": seed,
            "architecture": "DistilBERT + ResNet18 (Cross-Attention Fusion)",
        }
    }

    # Print Results Table
    print("\n" + "=" * 65)
    print("HELD-OUT TEST EVALUATION METRICS")
    print("-" * 65)
    print(f"Accuracy                 : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision (Class 1 FAKE) : {metrics['precision']:.4f}")
    print(f"Recall (Class 1 FAKE)    : {metrics['recall']:.4f}")
    print(f"F1-Score                 : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC Score            : {metrics['roc_auc']:.4f}")
    print(f"False Positive Rate (FPR): {metrics['rates']['false_positive_rate']:.4f}")
    print(f"False Negative Rate (FNR): {metrics['rates']['false_negative_rate']:.4f}")
    print("-" * 65)
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 65)

    # Save artifacts
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    metrics_json_path = REPORTS_DIR / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved to: {metrics_json_path.relative_to(BASE_DIR)}")

    # 2. Confusion Matrix Plot
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path)
    print(f"Confusion Matrix saved to: {cm_path.relative_to(BASE_DIR)}")

    # 3. ROC Curve Plot
    roc_path = REPORTS_DIR / "roc_curve.png"
    plot_roc_curve(test_targets, test_probs, auc, roc_path)
    print(f"ROC Curve saved to: {roc_path.relative_to(BASE_DIR)}")

    # 4. Comprehensive Markdown Report
    eval_md_path = REPORTS_DIR / "evaluation_report.md"
    generate_evaluation_markdown(metrics, sample_results, eval_md_path)
    print(f"Evaluation Report saved to: {eval_md_path.relative_to(BASE_DIR)}")

    print("\n[PASS] Training and evaluation pipeline finished with 0 errors.")
    print("=" * 65)
    return 0


def generate_evaluation_markdown(metrics: Dict[str, Any], samples: List[Dict[str, Any]], out_path: Path):
    cm = metrics["confusion_matrix"]
    rates = metrics["rates"]
    hp = metrics["hyperparameters"]

    md = f"""# TruthLens AI — Public Dataset Evaluation Report

**Benchmark Dataset:** r/Fakeddit (Multimodal Misinformation Benchmark, LREC 2020)  
**Evaluation Split:** Held-out Test Set ($N={metrics['num_test_samples']}$)  
**Model Architecture:** {hp['architecture']}  
**Generated On:** Automated Execution via `scripts/train_and_eval_public.py`  

---

## 1. Executive Summary & Test Performance

The TruthLens AI cross-attention model was evaluated on the independent, held-out test split of the public Fakeddit benchmark ($N={metrics['num_test_samples']}$, balanced REAL vs. FAKE).

| Metric | Measured Value | Percentage / Score | Description |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **{metrics['accuracy']:.4f}** | **{metrics['accuracy']*100:.1f}%** | Overall fraction of correct predictions |
| **Precision (FAKE)** | **{metrics['precision']:.4f}** | **{metrics['precision']*100:.1f}%** | True Fakes divided by all predicted Fakes |
| **Recall (FAKE)** | **{metrics['recall']:.4f}** | **{metrics['recall']*100:.1f}%** | Proportion of actual Fakes correctly detected |
| **F1-Score** | **{metrics['f1_score']:.4f}** | **{metrics['f1_score']:.4f}** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **{metrics['roc_auc']:.4f}** | **{metrics['roc_auc']:.4f}** | Area under the Receiver Operating Characteristic |
| **False Positive Rate (FPR)** | **{rates['false_positive_rate']:.4f}** | **{rates['false_positive_rate']*100:.1f}%** | Authentic samples falsely flagged as fake |
| **False Negative Rate (FNR)** | **{rates['false_negative_rate']:.4f}** | **{rates['false_negative_rate']*100:.1f}%** | Deceptive samples missed by the verifier |

---

## 2. Confusion Matrix Analysis

```text
                  Predicted REAL (0)    Predicted FAKE (1)
True REAL (0)            {cm['true_negatives']:2d} (TN)               {cm['false_positives']:2d} (FP)
True FAKE (1)            {cm['false_negatives']:2d} (FN)               {cm['true_positives']:2d} (TP)
```

- **True Negatives (TN):** {cm['true_negatives']} authentic social media posts correctly verified as REAL.
- **True Positives (TP):** {cm['true_positives']} manipulated/misleading posts correctly flagged as FAKE.
- **False Positives (FP):** {cm['false_positives']} benign posts flagged as fake (FPR = {rates['false_positive_rate']:.3f}).
- **False Negatives (FN):** {cm['false_negatives']} deceptive posts missed (FNR = {rates['false_negative_rate']:.3f}).

Visual Confusion Matrix available at: `reports/confusion_matrix.png`  
Visual ROC Curve available at: `reports/roc_curve.png`  

---

## 3. Methodological Discussion: Why Accuracy Alone Is Misleading

In real-world content moderation, **accuracy alone is inherently deceptive**:
1. **Asymmetric Error Costs:** A False Positive (flagging legitimate journalism as fake) censors truthful civic speech and degrades user trust. Conversely, a False Negative (allowing a deepfake to circulate) exposes the public to digital harm.
2. **Class Imbalance in the Wild:** If a platform experiences 95% authentic content and 5% misinformation, a naive "predict everything REAL" baseline achieves **95% accuracy** while detecting exactly **0%** of deepfakes (Recall = 0.0).
3. **F1 and ROC-AUC Standard:** Evaluating F1-Score ({metrics['f1_score']:.4f}) and ROC-AUC ({metrics['roc_auc']:.4f}) provides a robust, threshold-independent measure of discriminative capacity that cannot be inflated by class distribution shifts.

---

## 4. Per-Sample Test Set Verification Log

| Sample ID | Text Snippet | Ground Truth | Prediction | Fake Prob | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for s in samples:
        v = "CORRECT" if s["is_correct"] else "INCORRECT"
        md += f"| `{s['sample_id']}` | {s['text']} | **{s['true_label']}** | {s['predicted_label']} | {s['fake_probability']:.4f} | {v} |\n"

    md += f"""
---

## 5. Experimental Hyperparameters

- **Base Models:** `distilbert-base-uncased` (Text) + `resnet18` (Vision)
- **Multimodal Fusion:** Bidirectional Cross-Attention (Query-Key-Value projection with residual normalization)
- **Training Epochs:** {hp['epochs']}
- **Learning Rate:** {hp['learning_rate']} (AdamW optimizer)
- **Batch Size:** {hp['batch_size']}
- **Random Seed:** {hp['seed']}
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    sys.exit(main())
