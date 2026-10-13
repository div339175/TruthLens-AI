"""Baseline training script for unimodal (Text-only or Vision-only) models."""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.datasets.dataset import TruthLensMultimodalDataset
from ml.models.text_encoder import TextEncoder
from ml.models.vision_encoder import VisionEncoder
from ml.models.classifier import ClassificationHead
from ml.training.losses import VerificationLoss

MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "baseline"


def train_unimodal(
    modality: str = "text",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    device_str: str = "auto",
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)

    print(f"[INFO] Training unimodal {modality.upper()} baseline on device: {device}")

    metadata_path = BASE_DIR / "ml" / "data" / "metadata" / "dataset.csv"
    train_dataset = TruthLensMultimodalDataset(metadata_path, split="train")
    val_dataset = TruthLensMultimodalDataset(metadata_path, split="val")

    if len(train_dataset) == 0:
        print("[ERROR] No training samples found. Please run scripts/prepare_data.py first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None

    # Model definition
    if modality == "text":
        encoder = TextEncoder(projection_dim=256, fine_tune=True).to(device)
    else:
        encoder = VisionEncoder(projection_dim=256, fine_tune=True).to(device)

    classifier = ClassificationHead(in_features=256, num_classes=2).to(device)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    criterion = VerificationLoss()

    best_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()
        total_train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            targets = batch["label"].to(device)

            if modality == "text":
                input_ids = batch["input_ids"].to(device)
                att_mask = batch["attention_mask"].to(device)
                feat = encoder(input_ids, att_mask)["projected_pooled"]
            else:
                images = batch["image"].to(device)
                feat = encoder(images)["projected_pooled"]

            logits = classifier(feat)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        train_acc = round(correct / max(total, 1), 4)
        avg_train_loss = round(total_train_loss / max(total, 1), 4)

        # Validation
        val_loss, val_acc = 0.0, 0.0
        if val_loader and len(val_loader) > 0:
            encoder.eval()
            classifier.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    targets = batch["label"].to(device)
                    if modality == "text":
                        feat = encoder(batch["input_ids"].to(device), batch["attention_mask"].to(device))["projected_pooled"]
                    else:
                        feat = encoder(batch["image"].to(device))["projected_pooled"]
                    v_logits = classifier(feat)
                    val_loss += criterion(v_logits, targets).item() * len(targets)
                    val_correct += (torch.argmax(v_logits, dim=-1) == targets).sum().item()
                    val_total += len(targets)
            val_acc = round(val_correct / max(val_total, 1), 4)
            val_loss = round(val_loss / max(val_total, 1), 4)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    # Save checkpoint
    ckpt_path = EXPERIMENTS_DIR / f"{modality}_baseline.pt"
    torch.save({
        "encoder_state": encoder.state_dict(),
        "classifier_state": classifier.state_dict(),
        "modality": modality,
    }, ckpt_path)

    # Save metrics
    with open(EXPERIMENTS_DIR / f"{modality}_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[INFO] Saved {modality} baseline checkpoint to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TruthLens Unimodal Baseline")
    parser.add_argument("--modality", choices=["text", "vision"], default="text")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_unimodal(modality=args.modality, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
