"""Classification head and confidence/uncertainty decision policy for TruthLens AI."""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """MLP classifier predicting binary logits (REAL=0, FAKE=1) from fused representations."""

    def __init__(self, in_features: int = 256, num_classes: int = 2, dropout_p: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Args:

        features: [B, in_features]

        Returns:
            logits: [B, 2] (class 0: REAL, class 1: FAKE)
        """
        return self.classifier(features)


class DecisionPolicy:
    """Computes final prediction (REAL, FAKE, or UNCERTAIN) and calibrated confidence scores."""

    def __init__(self, uncertainty_threshold: float = 0.65):
        self.uncertainty_threshold = uncertainty_threshold

    def evaluate(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Convert binary classification logits into probabilities and apply uncertainty policy.

        Args:
            logits: [B, 2] or [2]

        Returns:
            Dict containing prediction ('REAL', 'FAKE', 'UNCERTAIN'),
            confidence (float), real_probability (float), fake_probability (float),
            uncertain (bool).
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probs = F.softmax(logits, dim=-1)
        real_prob = float(probs[0, 0].item())
        fake_prob = float(probs[0, 1].item())

        if fake_prob >= self.uncertainty_threshold:
            prediction = "FAKE"
            confidence = fake_prob
            is_uncertain = False
        elif real_prob >= self.uncertainty_threshold:
            prediction = "REAL"
            confidence = real_prob
            is_uncertain = False
        else:
            prediction = "UNCERTAIN"
            confidence = max(real_prob, fake_prob)
            is_uncertain = True

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "uncertain": is_uncertain,
            "threshold_used": self.uncertainty_threshold,
        }
