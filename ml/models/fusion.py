"""Feature-level concatenation fusion module for multimodal representations."""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Concatenates projected text and visual features and maps them through an MLP."""

    def __init__(self, projection_dim: int = 256, hidden_dim: int = 256, dropout_p: float = 0.2):
        super().__init__()
        # Input dimension is 2 * projection_dim (text + vision)
        # Extensible to 3 * projection_dim for audio in future
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, text_feature: torch.Tensor, vision_feature: torch.Tensor) -> torch.Tensor:
        """Concatenate features along channel dimension and pass through MLP.

        Args:
            text_feature: [B, projection_dim]
            vision_feature: [B, projection_dim]

        Returns:
            fused_representation: [B, projection_dim]
        """
        combined = torch.cat([text_feature, vision_feature], dim=-1)
        return self.mlp(combined)
