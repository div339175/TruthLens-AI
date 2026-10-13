"""Comprehensive model evaluation on test split with metrics and confusion matrix."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.datasets.dataset import TruthLensMultimodalDataset
from ml.models.multimodal_model import MultimodalVerificationModel
from ml.evaluation.metrics import compute_classification_metrics
from ml.evaluation.confusion_matrix import compute_confusion_matrix

RESULTS_DIR = BASE_DIR / "reports" / "results"
MODELS_DIR = BASE_DIR / "models"


def evaluate_model(
    checkpoint_path: Optional[str] = None,
    fusion_type: str = "cross_attention",
    split: str = "test",
    device_str: str = "auto",
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)

    metadata_path = BASE_DIR / "ml" / "data" / "metadata" / "dataset.csv"
    dataset = TruthLensMultimodalDataset(metadata_path, split=split)
    if len(dataset) == 0:
        print(f"[WARN] Split '{split}' is empty. Falling back to all samples.")
        dataset = TruthLensMultimodalDataset(metadata_path, split=None)

    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = MultimodalVerificationModel(
        fusion_type=fusion_type,
        projection_dim=256,
        pretrained_vision=True,
    ).to(device)

    # Load checkpoint if given or available
    ckpt_file = Path(checkpoint_path) if checkpoint_path else MODELS_DIR / "multimodal_baseline.pt"
    if ckpt_file.exists():
        model.load_state_dict(torch.load(ckpt_file, map_location=device), strict=False)
        print(f"[INFO] Evaluated using checkpoint: {ckpt_file}")
    else:
        print("[INFO] Evaluated using initialized model backbone.")

    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in loader:
            y_true.extend(batch["label"].tolist())
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
            probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(out["logits"], dim=-1).cpu().tolist()

            y_prob.extend(probs)
            y_pred.extend(preds)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    cm = compute_confusion_matrix(y_true, y_pred)

    report = {
        "dataset_split": split,
        "sample_count": len(y_true),
        "fusion_type": fusion_type,
        "metrics": metrics,
        "confusion_matrix": cm,
    }

    out_file = RESULTS_DIR / "evaluation_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\nModel Evaluation Summary")
    print("========================")
    print(f"Samples evaluated: {len(y_true)}")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score:  {metrics['f1'] * 100:.2f}%")
    print(f"Confusion Matrix: {cm['matrix']}")
    print(f"Report saved to: {out_file}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TruthLens Multimodal Model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fusion", type=str, default="cross_attention")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    evaluate_model(checkpoint_path=args.checkpoint, fusion_type=args.fusion, split=args.split)
