"""Domain-Adaptive Text Fine-Tuning Pipeline for TruthLens AI (Phase 6).

Fine-tunes the DistilBERT text encoder on joint in-domain + social media (Fakeddit) text
to mitigate the severe domain shift identified in Phase 5 (-0.6667 F1 degradation).

Saves to: models/phase6/text_domain_adapted.pt
Does NOT overwrite any Phase 1-5 checkpoint.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.preprocessing.text import TextPreprocessor
from ml.models.text_encoder import TextEncoder
from ml.models.classifier import ClassificationHead
from ml.training.losses import VerificationLoss
from experiments.metrics import compute_comprehensive_metrics

DATA_DIR = BASE_DIR / "experiments" / "phase6" / "data"
RESULTS_DIR = BASE_DIR / "experiments" / "phase6" / "results"
CONFIGS_DIR = BASE_DIR / "experiments" / "phase6" / "configs"
MODELS_P6_DIR = BASE_DIR / "models" / "phase6"
BASELINES_DIR = BASE_DIR / "experiments" / "baseline"


class DomainTextDataset(Dataset):
    """PyTorch Dataset loading domain adaptation text CSV."""

    def __init__(self, csv_path: Path, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.preprocessor = TextPreprocessor(max_length=max_length)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = int(row["label"])

        encoded = self.preprocessor.tokenize(text)
        return {
            "sample_id": row["sample_id"],
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": text,
        }


def train_text_domain_adaptation(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-5,
    seed: int = 42,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Fine-tunes DistilBERT with domain adaptation."""
    torch.manual_seed(seed)
    MODELS_P6_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("\n" + "=" * 70)
    print("TRUTHLENS AI — PHASE 6 DOMAIN-ADAPTIVE TEXT FINE-TUNING")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Seed: {seed}")
    print("=" * 70)

    train_csv = DATA_DIR / "domain_adaptive_train.csv"
    val_csv = DATA_DIR / "domain_adaptive_val.csv"

    train_dataset = DomainTextDataset(train_csv)
    val_dataset = DomainTextDataset(val_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model: fine_tune=True to enable transformer gradients
    encoder = TextEncoder(projection_dim=256, fine_tune=True, dropout_p=0.3).to(device)
    classifier = ClassificationHead(in_features=256, num_classes=2, dropout_p=0.3).to(device)

    # Warm-start from baseline text checkpoint if present
    base_ckpt = BASELINES_DIR / "text_baseline.pt"
    if base_ckpt.exists():
        data = torch.load(base_ckpt, map_location=device)
        encoder.load_state_dict(data["encoder_state"], strict=False)
        classifier.load_state_dict(data["classifier_state"], strict=False)
        print(f"[INFO] Initialized weights from baseline: {base_ckpt}")

    # Set up differential learning rates: lower for transformer, higher for projection/head
    optimizer = torch.optim.AdamW([
        {"params": encoder.transformer.parameters(), "lr": lr * 0.5},
        {"params": encoder.projection.parameters(), "lr": lr},
        {"params": classifier.parameters(), "lr": lr},
    ], weight_decay=0.01)

    criterion = VerificationLoss(label_smoothing=0.05)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = []

    out_ckpt = MODELS_P6_DIR / "text_domain_adapted.pt"

    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()

        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            out = encoder(input_ids=ids, attention_mask=mask)
            logits = classifier(out["projected_pooled"])

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        encoder.eval()
        classifier.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_true, val_pred, val_prob = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                out = encoder(input_ids=ids, attention_mask=mask)
                logits = classifier(out["projected_pooled"])

                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)

                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                preds = torch.argmax(logits, dim=-1).cpu().tolist()

                val_true.extend(labels.cpu().tolist())
                val_pred.extend(preds)
                val_prob.extend(probs)

                val_correct += sum(1 for p, t in zip(preds, labels.cpu().tolist()) if p == t)
                val_total += len(labels)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_metrics = compute_comprehensive_metrics(val_true, val_pred, val_prob)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_metrics['f1']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": val_metrics["f1"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "classifier_state": classifier.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_metrics["f1"],
                "config": {"lr": lr, "batch_size": batch_size, "epochs": epochs, "seed": seed},
            }, out_ckpt)
            print(f"  [+] Saved new best model checkpoint to: {out_ckpt}")

    # Save Config and History
    config_dict = {
        "experiment_id": "EXP_PHASE6_TEXT_001",
        "model": "DistilBERT Domain Adapted",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "checkpoint_path": str(out_ckpt),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
    }

    with open(CONFIGS_DIR / "text_adaptation_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(RESULTS_DIR / "text_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("DOMAIN-ADAPTIVE FINE-TUNING COMPLETE")
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}% | Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {out_ckpt}")
    print("=" * 70)

    return config_dict


if __name__ == "__main__":
    train_text_domain_adaptation()
