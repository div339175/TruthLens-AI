"""Multimodal Deepfake Verification Model combining Text, Vision, and Audio backbones."""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

from ml.models.text_encoder import TextEncoder
from ml.models.vision_encoder import VisionEncoder
from ml.models.fusion import ConcatFusion
from ml.models.cross_attention import BidirectionalCrossAttention
from ml.models.classifier import ClassificationHead


class MultimodalVerificationModel(nn.Module):
    """End-to-end multimodal architecture for deepfake and misinformation verification.

    Supports:
    - Text + Visual (Image/Video)
    - Text + Audio
    - Visual + Audio
    - Tri-modal: Text + Visual + Audio
    - Unimodal baselines (Text-only, Visual-only, Audio-only)
    - Feature Concat & Bidirectional Cross-Attention fusion
    """

    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        vision_model_name: str = "resnet18",
        audio_feature_dim: int = 128,
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
        # Audio Projection Layer
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(projection_dim, projection_dim),
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

        # Tri-modal aggregation layer
        self.trimodal_fusion = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(projection_dim, projection_dim),
        )

        # Classification Heads
        # Main multimodal head
        self.classifier = ClassificationHead(in_features=projection_dim, num_classes=2, dropout_p=dropout_p)

        # Unimodal & specialized heads
        self.text_classifier = ClassificationHead(in_features=projection_dim, num_classes=2, dropout_p=dropout_p)
        self.vision_classifier = ClassificationHead(in_features=projection_dim, num_classes=2, dropout_p=dropout_p)
        self.audio_classifier = ClassificationHead(in_features=projection_dim, num_classes=2, dropout_p=dropout_p)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass accommodating multimodal (bi-modal, tri-modal) or unimodal inputs.

        Supports:
        - images: [B, 3, H, W] for static images
        - video_frames: [N, 3, H, W] for video frame sequences (applies temporal feature pooling)
        """
        has_text = input_ids is not None
        has_image = images is not None
        has_video = video_frames is not None
        has_vision = has_image or has_video
        has_audio = audio_features is not None

        if not has_text and not has_vision and not has_audio:
            raise ValueError("At least one modality (text, images, video_frames, or audio) must be provided.")

        text_out = None
        vision_out = None
        audio_out = None
        cross_attn_details = None
        modalities_used: List[str] = []

        if has_text:
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            modalities_used.append("text")

        if has_video:
            # Process N frames individually through ResNet18 and apply temporal feature pooling
            vision_out = self.vision_encoder.encode_video_frames(video_frames=video_frames)
            modalities_used.append("video")
        elif has_image:
            vision_out = self.vision_encoder(images=images)
            modalities_used.append("visual")

        if has_audio:
            if audio_features.dim() == 1:
                audio_features = audio_features.unsqueeze(0)
            audio_out = self.audio_projection(audio_features)
            modalities_used.append("audio")

        # Case 1: Tri-modal (Text + Vision + Audio)
        if has_text and has_vision and has_audio:
            t_feat = text_out["projected_pooled"]
            v_feat = vision_out["projected_pooled"]
            a_feat = audio_out
            tri_combined = torch.cat([t_feat, v_feat, a_feat], dim=-1)
            fused = self.trimodal_fusion(tri_combined)
            logits = self.classifier(fused)
            return {
                "logits": logits,
                "fused_features": fused,
                "text_features": t_feat,
                "vision_features": v_feat,
                "audio_features": a_feat,
                "cross_attention": None,
                "modalities": modalities_used,
            }

        # Case 2: Text + Vision
        elif has_text and has_vision:
            if self.fusion_type == "concat":
                fused = self.fusion_module(
                    text_feature=text_out["projected_pooled"],
                    vision_feature=vision_out["projected_pooled"],
                )
            else:  # cross_attention
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
                "frame_features": vision_out.get("projected_frames"),
                "video_frame_features": vision_out.get("projected_frames"),
                "audio_features": None,
                "cross_attention": cross_attn_details,
                "modalities": modalities_used,
            }

        # Case 3: Text + Audio
        elif has_text and has_audio:
            t_feat = text_out["projected_pooled"]
            a_feat = audio_out
            fused = self.fusion_module(t_feat, a_feat) if hasattr(self.fusion_module, "mlp") else self.trimodal_fusion(torch.cat([t_feat, a_feat, t_feat], dim=-1))
            logits = self.classifier(fused)
            return {
                "logits": logits,
                "fused_features": fused,
                "text_features": t_feat,
                "vision_features": None,
                "audio_features": a_feat,
                "cross_attention": None,
                "modalities": modalities_used,
            }

        # Case 4: Vision + Audio
        elif has_vision and has_audio:
            v_feat = vision_out["projected_pooled"]
            a_feat = audio_out
            fused = self.fusion_module(v_feat, a_feat) if hasattr(self.fusion_module, "mlp") else self.trimodal_fusion(torch.cat([v_feat, a_feat, v_feat], dim=-1))
            logits = self.classifier(fused)
            return {
                "logits": logits,
                "fused_features": fused,
                "text_features": None,
                "vision_features": v_feat,
                "audio_features": a_feat,
                "cross_attention": None,
                "modalities": modalities_used,
            }

        # Case 5: Text-only
        elif has_text:
            fused = text_out["projected_pooled"]
            logits = self.text_classifier(fused)
            return {
                "logits": logits,
                "fused_features": fused,
                "text_features": fused,
                "vision_features": None,
                "audio_features": None,
                "cross_attention": None,
                "modalities": ["text"],
            }

        # Case 6: Vision-only
        elif has_vision:
            fused = vision_out["projected_pooled"]
            logits = self.vision_classifier(fused)
            return {
                "logits": logits,
                "fused_features": fused,
                "text_features": None,
                "vision_features": fused,
                "frame_features": vision_out.get("projected_frames"),
                "video_frame_features": vision_out.get("projected_frames"),
                "audio_features": None,
                "cross_attention": None,
                "modalities": modalities_used,
            }

        # Case 7: Audio-only
        else:
            fused = audio_out
            logits = self.audio_classifier(fused)
            return {
                "logits": logits,
                "fused_features": fused,
                "text_features": None,
                "vision_features": None,
                "audio_features": fused,
                "cross_attention": None,
                "modalities": ["audio"],
            }
