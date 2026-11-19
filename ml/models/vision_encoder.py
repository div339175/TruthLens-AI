"""Vision Encoder module using torchvision ResNet18 backbone."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class VisionEncoder(nn.Module):
    """Convolutional vision encoder with spatial feature extraction and projection head."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        projection_dim: int = 256,
        fine_tune: bool = False,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
            self.embed_dim = 512

            # Break into feature extractor (up to layer4) and pooling
            self.conv_stem = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,  # [B, 512, 7, 7] for 224x224 input
            )
            self.avgpool = base_model.avgpool  # [B, 512, 1, 1]
        else:
            raise ValueError(f"Unsupported vision backbone: {backbone_name}")

        if not fine_tune:
            for param in self.conv_stem.parameters():
                param.requires_grad = False

        # Projection layer to shared multimodal space
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: [B, 3, H, W]

        Returns:
            Dict containing:
                - 'spatial_features': [B, 512, 7, 7]
                - 'spatial_sequence': [B, 49, 512] (flattened 7x7 spatial tokens)
                - 'pooled_features': [B, 512]
                - 'projected_pooled': [B, projection_dim]
                - 'projected_sequence': [B, 49, projection_dim]
        """
        spatial = self.conv_stem(images)  # [B, 512, 7, 7]
        b, c, h, w = spatial.shape

        # Flatten spatial grid into sequence for cross-attention
        spatial_seq = spatial.flatten(2).permute(0, 2, 1)  # [B, 49, 512]

        pooled = self.avgpool(spatial).flatten(1)  # [B, 512]
        proj_pooled = self.projection(pooled)  # [B, projection_dim]
        proj_seq = self.projection(spatial_seq)  # [B, 49, projection_dim]

        return {
            "spatial_features": spatial,
            "spatial_sequence": spatial_seq,
            "pooled_features": pooled,
            "projected_pooled": proj_pooled,
            "projected_sequence": proj_seq,
        }

    def pool_video_frames(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Pool features across sampled video frames [N, D] -> [1, D].

        Uses mean pooling over temporal frame dimension.
        """
        return torch.mean(frame_features, dim=0, keepdim=True)

    def encode_video_frames(self, video_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode N video frames individually through ResNet18 and apply temporal feature pooling.

        Pipeline:
            [N, 3, H, W] -> ResNet18 -> [N, 512] -> Temporal Mean Pooling -> [1, 512]
            [N, 512] -> Projection -> [N, projection_dim] -> Temporal Mean Pooling -> [1, projection_dim]

        Args:
            video_frames: [N, 3, H, W] batch of sampled frames.

        Returns:
            Dict containing:
                - 'frame_features': [N, 512] individual frame feature vectors
                - 'projected_frames': [N, projection_dim] individual projected frame embeddings
                - 'pooled_features': [1, 512] temporally mean-pooled features
                - 'projected_pooled': [1, projection_dim] temporally mean-pooled video embedding
                - 'projected_sequence': [1, 49, projection_dim] temporally mean-pooled spatial sequence
        """
        if video_frames.dim() == 5:
            # Shape [B, N, C, H, W]
            b, n, c, h, w = video_frames.shape
            flat_frames = video_frames.view(b * n, c, h, w)
            frame_out = self.forward(flat_frames)

            proj_frames = frame_out["projected_pooled"].view(b, n, -1)
            raw_frames = frame_out["pooled_features"].view(b, n, -1)
            proj_seq = frame_out["projected_sequence"].view(b, n, 49, -1)

            pooled_proj = torch.mean(proj_frames, dim=1)  # [B, projection_dim]
            pooled_raw = torch.mean(raw_frames, dim=1)    # [B, 512]
            pooled_seq = torch.mean(proj_seq, dim=1)      # [B, 49, projection_dim]

            return {
                "frame_features": raw_frames,
                "projected_frames": proj_frames,
                "pooled_features": pooled_raw,
                "projected_pooled": pooled_proj,
                "projected_sequence": pooled_seq,
                "spatial_features": frame_out["spatial_features"],
            }

        # video_frames is [N, 3, 224, 224]
        frame_out = self.forward(video_frames)
        frame_features = frame_out["pooled_features"]       # [N, 512]
        projected_frames = frame_out["projected_pooled"]     # [N, projection_dim]

        # Temporal feature-level pooling across frame dimension (dim 0)
        pooled_features = torch.mean(frame_features, dim=0, keepdim=True)       # [1, 512]
        projected_pooled = torch.mean(projected_frames, dim=0, keepdim=True)     # [1, projection_dim]
        projected_sequence = torch.mean(frame_out["projected_sequence"], dim=0, keepdim=True)  # [1, 49, projection_dim]

        return {
            "spatial_features": frame_out["spatial_features"],
            "spatial_sequence": frame_out["spatial_sequence"],
            "frame_features": frame_features,
            "projected_frames": projected_frames,
            "pooled_features": pooled_features,
            "projected_pooled": projected_pooled,
            "projected_sequence": projected_sequence,
        }
