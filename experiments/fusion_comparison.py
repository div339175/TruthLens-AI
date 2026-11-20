"""Fusion Strategy Comparison for TruthLens AI (Phase 4).

Compares Concatenation Fusion vs Bidirectional Cross-Attention on identical evaluation data:
- EXP_MULTIMODAL_CONCAT_001: Concatenation Fusion (models/multimodal_concat.pt)
- EXP_MULTIMODAL_CA_001: Bidirectional Cross-Attention (models/multimodal_cross_attention.pt)
"""

import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.datasets.dataset import TruthLensMultimodalDataset
from ml.models.multimodal_model import MultimodalVerificationModel
from experiments.metrics import compute_comprehensive_metrics

REPORTS_DIR = BASE_DIR / "experiments" / "reports"
MODELS_DIR = BASE_DIR / "models"
METADATA_PATH = BASE_DIR / "ml" / "data" / "metadata" / "dataset.csv"


def evaluate_multimodal_variant(
    experiment_id: str,
    fusion_type: str,
    checkpoint_file: Path,
    device: torch.device,
    split: str = "test",
) -> Dict[str, Any]:
    """Evaluates a multimodal fusion variant on identical test split."""
    torch.manual_seed(42)
    dataset = TruthLensMultimodalDataset(METADATA_PATH, split=split)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = MultimodalVerificationModel(
        fusion_type=fusion_type,
        projection_dim=256,
        pretrained_vision=True,
    ).to(device)

    if checkpoint_file.exists():
        model.load_state_dict(torch.load(checkpoint_file, map_location=device), strict=False)
        ckpt_status = f"Loaded from {checkpoint_file}"
    else:
        ckpt_status = f"Warning: checkpoint not found at {checkpoint_file}"

    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            y_true.extend(batch["label"].tolist())
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            y_prob.extend(probs)
            y_pred.extend(preds)

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob)

    return {
        "experiment_id": experiment_id,
        "fusion_type": fusion_type,
        "modalities": "text+visual",
        "checkpoint": ckpt_status,
        "split": split,
        "metrics": metrics,
        "raw_predictions": {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        },
    }


def run_fusion_comparison(device_str: str = "auto", split: str = "test") -> Dict[str, Any]:
    """Runs fusion comparison between Concat and Cross-Attention."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)

    print(f"\n[INFO] Running Phase 4 Fusion Comparison on device={device}, split={split}...")

    # 1. Concat Fusion
    concat_res = evaluate_multimodal_variant(
        experiment_id="EXP_MULTIMODAL_CONCAT_001",
        fusion_type="concat",
        checkpoint_file=MODELS_DIR / "multimodal_concat.pt",
        device=device,
        split=split,
    )

    # 2. Bidirectional Cross-Attention
    ca_res = evaluate_multimodal_variant(
        experiment_id="EXP_MULTIMODAL_CA_001",
        fusion_type="cross_attention",
        checkpoint_file=MODELS_DIR / "multimodal_cross_attention.pt",
        device=device,
        split=split,
    )

    results = {
        "concatenation": concat_res,
        "cross_attention": ca_res,
    }

    # Save JSON report
    out_json = REPORTS_DIR / "fusion_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        summary = {
            k: {
                "experiment_id": v["experiment_id"],
                "fusion_type": v["fusion_type"],
                "modalities": v["modalities"],
                "metrics": v["metrics"],
            }
            for k, v in results.items()
        }
        json.dump(summary, f, indent=2)

    # Save CSV table
    out_csv = REPORTS_DIR / "fusion_comparison.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Model", "Modalities", "Fusion", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in results.items():
            m = v["metrics"]
            writer.writerow({
                "Model": v["experiment_id"],
                "Modalities": v["modalities"],
                "Fusion": v["fusion_type"],
                "Accuracy": f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1": f"{m['f1']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A",
                "PR-AUC": f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A",
            })

    print(f"[INFO] Fusion comparison complete. Output saved to {out_json} and {out_csv}")
    return results


if __name__ == "__main__":
    run_fusion_comparison()
