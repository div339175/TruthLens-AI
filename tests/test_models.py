"""Tests for model architectures, fusion mechanisms, and decision policy."""

import torch
import pytest

from ml.models.text_encoder import TextEncoder
from ml.models.vision_encoder import VisionEncoder
from ml.models.fusion import ConcatFusion
from ml.models.cross_attention import BidirectionalCrossAttention
from ml.models.classifier import ClassificationHead, DecisionPolicy
from ml.models.multimodal_model import MultimodalVerificationModel


def test_text_encoder_forward():
    encoder = TextEncoder(projection_dim=128)
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones(2, 16)

    out = encoder(input_ids, attention_mask)
    assert out["pooled_output"].shape == (2, 768)
    assert out["projected_pooled"].shape == (2, 128)
    assert out["projected_sequence"].shape == (2, 16, 128)


def test_vision_encoder_forward():
    encoder = VisionEncoder(projection_dim=128, pretrained=False)
    images = torch.randn(2, 3, 224, 224)

    out = encoder(images)
    assert out["pooled_features"].shape == (2, 512)
    assert out["projected_pooled"].shape == (2, 128)
    assert out["spatial_sequence"].shape == (2, 49, 512)
    assert out["projected_sequence"].shape == (2, 49, 128)


def test_concat_fusion():
    fusion = ConcatFusion(projection_dim=128, hidden_dim=128)
    text_feat = torch.randn(2, 128)
    vis_feat = torch.randn(2, 128)

    fused = fusion(text_feat, vis_feat)
    assert fused.shape == (2, 128)


def test_cross_attention_fusion():
    ca = BidirectionalCrossAttention(embed_dim=128, num_heads=4)
    text_seq = torch.randn(2, 16, 128)
    vision_seq = torch.randn(2, 49, 128)

    out = ca(text_seq=text_seq, vision_seq=vision_seq)
    assert out["fused_embedding"].shape == (2, 128)
    assert "text_to_vision_attn" in out
    assert "vision_to_text_attn" in out


def test_decision_policy_real_fake_uncertain():
    policy = DecisionPolicy(uncertainty_threshold=0.65)

    # Clear fake (e.g. logits yielding fake_prob = 0.9)
    logits_fake = torch.tensor([[ -2.0, 2.0 ]])
    res_fake = policy.evaluate(logits_fake)
    assert res_fake["prediction"] == "FAKE"
    assert res_fake["uncertain"] is False
    assert res_fake["fake_probability"] > 0.65

    # Clear real
    logits_real = torch.tensor([[ 2.0, -2.0 ]])
    res_real = policy.evaluate(logits_real)
    assert res_real["prediction"] == "REAL"
    assert res_real["uncertain"] is False
    assert res_real["real_probability"] > 0.65

    # Ambiguous / close to 0.50 (uncertain)
    logits_ambiguous = torch.tensor([[ 0.05, -0.05 ]])
    res_ambiguous = policy.evaluate(logits_ambiguous)
    assert res_ambiguous["prediction"] == "UNCERTAIN"
    assert res_ambiguous["uncertain"] is True


def test_multimodal_model_end_to_end():
    model = MultimodalVerificationModel(
        projection_dim=128,
        fusion_type="cross_attention",
        pretrained_vision=False,
    )
    input_ids = torch.randint(0, 1000, (2, 16))
    att_mask = torch.ones(2, 16)
    images = torch.randn(2, 3, 224, 224)

    out = model(input_ids=input_ids, attention_mask=att_mask, images=images)
    assert out["logits"].shape == (2, 2)
    assert out["modalities"] == ["text", "visual"]
    assert "cross_attention" in out
