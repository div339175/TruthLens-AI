"""Ablation experiments comparing unimodal and multimodal variants on test set."""

import os
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.datasets.dataset import TruthLensMultimodalDataset
from ml.models.multimodal_model import MultimodalVerificationModel
from ml.evaluation.metrics import compute_classification_metrics

RESULTS_DIR = BASE_DIR / "reports" / "results"


def run_ablation_study(device_str: str = "auto") -> List[Dict[str, Any]]:
    """Evaluates ablation variants across modalities and fusion methods on test split."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)

    metadata_path = BASE_DIR / "ml" / "data" / "metadata" / "dataset.csv"
    test_dataset = TruthLensMultimodalDataset(metadata_path, split="test")

    if len(test_dataset) == 0:
        print("[WARN] Test dataset split is empty. Running evaluation on all samples.")
        test_dataset = TruthLensMultimodalDataset(metadata_path, split=None)

    if len(test_dataset) == 0:
        print("[ERROR] No dataset available for ablation study.")
        return []

    loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    variants = [
        {"name": "Text_Only", "modalities": "text", "fusion": "concat", "use_text": True, "use_img": False},
        {"name": "Vision_Only", "modalities": "visual", "fusion": "concat", "use_text": False, "use_img": True},
        {"name": "Multimodal_Concat", "modalities": "text+visual", "fusion": "concat", "use_text": True, "use_img": True},
        {"name": "Multimodal_CrossAttn", "modalities": "text+visual", "fusion": "cross_attention", "use_text": True, "use_img": True},
    ]

    results = []

    print("\nRunning Ablation Evaluation")
    print("=" * 60)

    for v in variants:
        model = MultimodalVerificationModel(
            fusion_type=v["fusion"],
            projection_dim=256,
            pretrained_vision=True,
        ).to(device)

        if v["name"] == "Text_Only":
            ckpt_file = BASE_DIR / "experiments" / "baseline" / "text_baseline.pt"
            if ckpt_file.exists():
                data = torch.load(ckpt_file, map_location=device)
                model.text_encoder.load_state_dict(data["encoder_state"], strict=False)
                model.text_classifier.load_state_dict(data["classifier_state"], strict=False)
        elif v["name"] == "Vision_Only":
            ckpt_file = BASE_DIR / "experiments" / "baseline" / "vision_baseline.pt"
            if ckpt_file.exists():
                data = torch.load(ckpt_file, map_location=device)
                model.vision_encoder.load_state_dict(data["encoder_state"], strict=False)
                model.vision_classifier.load_state_dict(data["classifier_state"], strict=False)
        elif v["name"] == "Multimodal_Concat":
            ckpt_file = BASE_DIR / "models" / "multimodal_concat.pt"
            if ckpt_file.exists():
                model.load_state_dict(torch.load(ckpt_file, map_location=device), strict=False)
        elif v["name"] == "Multimodal_CrossAttn":
            ckpt_file = BASE_DIR / "models" / "multimodal_cross_attention.pt"
            if ckpt_file.exists():
                model.load_state_dict(torch.load(ckpt_file, map_location=device), strict=False)

        model.eval()

        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for batch in loader:
                labels = batch["label"].tolist()
                y_true.extend(labels)

                input_ids = batch["input_ids"].to(device) if v["use_text"] else None
                att_mask = batch["attention_mask"].to(device) if v["use_text"] else None
                images = batch["image"].to(device) if v["use_img"] else None

                out = model(input_ids=input_ids, attention_mask=att_mask, images=images)
                logits = out["logits"]
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                preds = torch.argmax(logits, dim=-1).cpu().tolist()

                y_prob.extend(probs)
                y_pred.extend(preds)

        m = compute_classification_metrics(y_true, y_pred, y_prob)
        entry = {
            "experiment": v["name"],
            "modalities": v["modalities"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        }
        results.append(entry)
        print(f"[{v['name']:<22}] Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1']:.4f}")

    # Write genuine results to CSV
    csv_file = RESULTS_DIR / "ablation_results.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["experiment", "modalities", "accuracy", "precision", "recall", "f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[INFO] Ablation results saved to: {csv_file}\n")
    return results


if __name__ == "__main__":
    run_ablation_study()
