"""Multimodal training script for TruthLens AI combining text and visual streams."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.datasets.dataset import TruthLensMultimodalDataset
from ml.models.multimodal_model import MultimodalVerificationModel
from ml.training.losses import VerificationLoss

MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "multimodal"


def train_multimodal(
    fusion_type: str = "cross_attention",
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 2e-5,
    device_str: str = "auto",
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)

    print(f"[INFO] Starting TruthLens Multimodal Training ({fusion_type}) on device: {device}")

    metadata_path = BASE_DIR / "ml" / "data" / "metadata" / "dataset.csv"
    train_dataset = TruthLensMultimodalDataset(metadata_path, split="train")
    val_dataset = TruthLensMultimodalDataset(metadata_path, split="val")

    if len(train_dataset) == 0:
        print("[ERROR] No training data found. Please run scripts/prepare_data.py first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None

    # Instantiate model
    model = MultimodalVerificationModel(
        fusion_type=fusion_type,
        projection_dim=256,
        pretrained_vision=True,
    ).to(device)

    # Train fusion and classification heads while keeping backbone feature extractors mostly frozen initially
    # Enable gradients for projection heads (text, vision, audio), fusion, and classifiers
    for param in model.text_encoder.projection.parameters():
        param.requires_grad = True
    for param in model.vision_encoder.projection.parameters():
        param.requires_grad = True
    for param in model.audio_projection.parameters():
        param.requires_grad = True
    for param in model.audio_classifier.parameters():
        param.requires_grad = True
    for param in model.trimodal_fusion.parameters():
        param.requires_grad = True
    for param in model.fusion_module.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = VerificationLoss()

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            targets = batch["label"].to(device)

            has_aud = batch.get("has_audio")
            aud_tensor = batch.get("audio")
            if aud_tensor is not None and has_aud is not None and has_aud.sum() > 0:
                audio_features = aud_tensor.to(device)
            else:
                audio_features = None

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                audio_features=audio_features,
            )
            logits = out["logits"]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        train_acc = round(correct / max(total, 1), 4)
        avg_loss = round(total_loss / max(total, 1), 4)

        # Validation
        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    v_in = batch["input_ids"].to(device)
                    v_mask = batch["attention_mask"].to(device)
                    v_img = batch["image"].to(device)
                    v_tar = batch["label"].to(device)

                    v_has_aud = batch.get("has_audio")
                    v_aud = batch.get("audio")
                    v_audio_feat = v_aud.to(device) if (v_aud is not None and v_has_aud is not None and v_has_aud.sum() > 0) else None

                    v_out = model(
                        input_ids=v_in,
                        attention_mask=v_mask,
                        images=v_img,
                        audio_features=v_audio_feat,
                    )
                    val_loss += criterion(v_out["logits"], v_tar).item() * len(v_tar)
                    v_preds = torch.argmax(v_out["logits"], dim=-1)
                    val_correct += (v_preds == v_tar).sum().item()
                    val_total += len(v_tar)

            val_acc = round(val_correct / max(val_total, 1), 4)
            val_loss = round(val_loss / max(val_total, 1), 4)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    # Save to standard checkpoint location: models/multimodal_baseline.pt
    main_ckpt = MODELS_DIR / "multimodal_baseline.pt"
    torch.save(model.state_dict(), main_ckpt)
    print(f"[INFO] Model weights saved to: {main_ckpt}")

    # Save experiment report
    exp_ckpt = EXPERIMENTS_DIR / f"multimodal_{fusion_type}.pt"
    torch.save(model.state_dict(), exp_ckpt)

    metrics_file = EXPERIMENTS_DIR / f"multimodal_{fusion_type}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "fusion_type": fusion_type,
            "epochs": epochs,
            "final_train_acc": train_acc,
            "final_val_acc": val_acc,
            "history": history,
        }, f, indent=2)

    print(f"[INFO] Multimodal training metrics recorded in: {metrics_file}")


def verify_audio_gradient_flow(model: MultimodalVerificationModel, device: torch.device) -> Dict[str, float]:
    """Diagnostic check confirming audio projection & classifier receive non-zero gradients."""
    model.train()
    for param in model.audio_projection.parameters():
        param.requires_grad = True
    for param in model.audio_classifier.parameters():
        param.requires_grad = True
    for param in model.trimodal_fusion.parameters():
        param.requires_grad = True

    model.zero_grad()
    dummy_audio = torch.randn(2, 128, device=device)
    dummy_targets = torch.tensor([0, 1], dtype=torch.long, device=device)

    out = model(audio_features=dummy_audio)
    loss = torch.nn.functional.cross_entropy(out["logits"], dummy_targets)
    loss.backward()

    grad_norms = {}
    for name, param in model.audio_projection.named_parameters():
        if param.grad is not None:
            grad_norms[f"audio_projection.{name}"] = float(param.grad.norm().item())
    for name, param in model.audio_classifier.named_parameters():
        if param.grad is not None:
            grad_norms[f"audio_classifier.{name}"] = float(param.grad.norm().item())

    model.zero_grad()
    return grad_norms


def main():
    parser = argparse.ArgumentParser(description="Train TruthLens Multimodal Model")
    parser.add_argument("--fusion", choices=["cross_attention", "concat"], default="cross_attention")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_multimodal(
        fusion_type=args.fusion,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
