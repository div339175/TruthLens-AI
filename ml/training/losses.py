"""Loss functions for unimodal and multimodal deepfake verification."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VerificationLoss(nn.Module):
    """Standard CrossEntropy loss with optional class weighting and label smoothing."""

    def __init__(self, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.05):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


class CrossModalContrastiveLoss(nn.Module):
    """Symmetric InfoNCE contrastive loss aligning text and visual embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_embeds: torch.Tensor, vision_embeds: torch.Tensor) -> torch.Tensor:
        """Args:

        text_embeds: [B, D]
        vision_embeds: [B, D]
        """
        # Normalize embeddings
        text_norm = F.normalize(text_embeds, p=2, dim=-1)
        vision_norm = F.normalize(vision_embeds, p=2, dim=-1)

        # Cosine similarity matrix [B, B]
        sim_matrix = torch.matmul(text_norm, vision_norm.T) / self.temperature

        # Ground truth diagonal alignment
        labels = torch.arange(text_embeds.size(0), device=text_embeds.device)
        loss_t2v = F.cross_entropy(sim_matrix, labels)
        loss_v2t = F.cross_entropy(sim_matrix.T, labels)

        return (loss_t2v + loss_v2t) / 2.0
