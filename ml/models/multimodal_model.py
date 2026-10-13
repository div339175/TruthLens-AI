"""Multimodal Deepfake Verification Model combining Text and Vision backbones."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from ml.models.text_encoder import TextEncoder
from ml.models.vision_encoder import VisionEncoder
from ml.models.fusion import ConcatFusion
from ml.models.cross_attention import BidirectionalCrossAttention
from ml.models.classifier import ClassificationHead


class MultimodalVerificationModel(nn.Module):
    """End-to-end multimodal architecture for deepfake detection.

    Supports:
    - Text + Image / Video input
    - Text-only input
    - Image-only input
    - Concat or Cross-Attention fusion
    """

    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        vision_model_name: str = "resnet18",
        projection_dim: int = 256,
        fusion_type: str = "cross_attention",
        pretrained_vision: bool = True,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.projection_dim = projection_dim

        # Modality Encoders
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            projection_dim=projection_dim,
            dropout_p=dropout_p,
        )
        self.vision_encoder = VisionEncoder(
            backbone_name=vision_model_name,
            pretrained=pretrained_vision,
            projection_dim=projection_dim,
            dropout_p=dropout_p,
        )

        # Fusion Layers
        if fusion_type == "concat":
            self.fusion_module = ConcatFusion(
                projection_dim=projection_dim,
                hidden_dim=projection_dim,
                dropout_p=dropout_p,
            )
        elif fusion_type == "cross_attention":
            self.fusion_module = BidirectionalCrossAttention(
                embed_dim=projection_dim,
                num_heads=4,
                dropout=dropout_p,
            )
        else:
            raise ValueError(f"Unsupported fusion_type '{fusion_type}'. Choose 'concat' or 'cross_attention'.")

        # Classification Heads
        # Main multimodal head
        self.classifier = ClassificationHead(in_features=projection_dim, num_classes=2, dropout_p=dropout_p)

        # Unimodal heads (for baseline evaluation and unimodal fallbacks)
        self.text_classifier = ClassificationHead(in_features=projection_dim, num_classes=2, dropout_p=dropout_p)
        self.vision_classifier = ClassificationHead(in_features=projection_dim, num_classes=2, dropout_p=dropout_p)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass accommodating multimodal or unimodal inputs.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            images: [B, 3, H, W]

        Returns:
            Dict with 'logits', 'fused_features', 'text_features', 'vision_features', 'cross_attn'
        """
        has_text = input_ids is not None
        has_vision = images is not None

        if not has_text and not has_vision:
            raise ValueError("At least one modality (text or images) must be provided.")

        text_out = None
        vision_out = None
        cross_attn_details = None

        if has_text:
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        if has_vision:
            vision_out = self.vision_encoder(images=images)

        # Case 1: Both modalities present -> Full Multimodal Fusion
        if has_text and has_vision:
            if self.fusion_type == "concat":
                fused = self.fusion_module(
                    text_feature=text_out["projected_pooled"],
                    vision_feature=vision_out["projected_pooled"],
                )
            else:  # cross_attention
                # Invert mask: PyTorch MultiheadAttention key_padding_mask expects True where values should be ignored
                pad_mask = (attention_mask == 0) if attention_mask is not None else None
                ca_out = self.fusion_module(
                    text_seq=text_out["projected_sequence"],
                    vision_seq=vision_out["projected_sequence"],
                    text_padding_mask=pad_mask,
                )
                fused = ca_out["fused_embedding"]
                cross_attn_details = {
                    "text_to_vision_attn": ca_out["text_to_vision_attn"],
                    "vision_to_text_attn": ca_out["vision_to_text_attn"],
                }

            logits = self.classifier(fused)
            return {
                "logits": logits,
                "fused_features": fused,
                "text_features": text_out["projected_pooled"],
                "vision_features": vision_out["projected_pooled"],
                "cross_attention": cross_attn_details,
                "modalities": ["text", "visual"],
            }

        # Case 2: Text-only branch
        elif has_text:
            fused = text_out["projected_pooled"]
            logits = self.text_classifier(fused)
            return {
                "logits": logits,
                "fused_features": fused,
                "text_features": fused,
                "vision_features": None,
                "cross_attention": None,
                "modalities": ["text"],
            }

        # Case 3: Vision-only branch
        else:
            fused = vision_out["projected_pooled"]
            logits = self.vision_classifier(fused)
            return {
                "logits": logits,
                "fused_features": fused,
                "text_features": None,
                "vision_features": fused,
                "cross_attention": None,
                "modalities": ["visual"],
            }
