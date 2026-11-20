"""Automated test suite for Phase 6 Contrastive Cross-Modal Alignment."""

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

from ml.training.losses import CrossModalContrastiveLoss
from ml.models.multimodal_model import MultimodalVerificationModel

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE6_MODELS = BASE_DIR / "models" / "phase6"


class TestPhase6ContrastiveLoss:
    """Verifies properties and mathematical formulation of CrossModalContrastiveLoss."""

    def test_contrastive_loss_shape_and_value(self):
        """Tests that contrastive loss computes a positive scalar loss."""
        loss_fn = CrossModalContrastiveLoss(temperature=0.07)
        batch_size = 4
        dim = 256
        t_feat = torch.randn(batch_size, dim, requires_grad=True)
        v_feat = torch.randn(batch_size, dim, requires_grad=True)

        loss = loss_fn(t_feat, v_feat)
        assert loss.ndim == 0, "Loss must be scalar"
        assert loss.item() > 0.0, "Loss must be positive"

    def test_contrastive_loss_symmetry(self):
        """Tests that contrastive loss is symmetric between modalities."""
        loss_fn = CrossModalContrastiveLoss(temperature=0.07)
        t_feat = torch.randn(4, 128)
        v_feat = torch.randn(4, 128)

        loss_t2v_v2t = loss_fn(t_feat, v_feat)
        loss_v2t_t2v = loss_fn(v_feat, t_feat)
        assert torch.isclose(loss_t2v_v2t, loss_v2t_t2v, atol=1e-5), "Contrastive loss must be symmetric"

    def test_aligned_pairs_produce_lower_loss(self):
        """Tests that perfectly aligned pairs produce strictly lower loss than random pairs."""
        loss_fn = CrossModalContrastiveLoss(temperature=0.07)
        feat = torch.randn(4, 256)
        aligned_loss = loss_fn(feat, feat)

        unaligned_feat = torch.randn(4, 256)
        unaligned_loss = loss_fn(feat, unaligned_feat)

        assert aligned_loss.item() < unaligned_loss.item(), "Aligned pairs must yield lower loss"

    def test_contrastive_gradients(self):
        """Verifies that gradients flow back through both text and vision features."""
        loss_fn = CrossModalContrastiveLoss(temperature=0.07)
        t_feat = torch.randn(4, 64, requires_grad=True)
        v_feat = torch.randn(4, 64, requires_grad=True)

        loss = loss_fn(t_feat, v_feat)
        loss.backward()

        assert t_feat.grad is not None, "Text features must receive gradients"
        assert v_feat.grad is not None, "Vision features must receive gradients"
        assert torch.isfinite(t_feat.grad).all(), "Gradients must be finite"
        assert torch.isfinite(v_feat.grad).all(), "Gradients must be finite"


class TestPhase6ContrastiveCheckpoints:
    """Verifies that Phase 6 contrastive models exist and load correctly."""

    def test_contrastive_checkpoint_exists(self):
        """Ensures best contrastive checkpoint exists."""
        ckpt = PHASE6_MODELS / "multimodal_contrastive.pt"
        assert ckpt.exists(), f"Expected checkpoint at {ckpt}"
        assert ckpt.stat().st_size > 10_000, "Checkpoint file must be non-empty"

    def test_contrastive_model_forward_pass(self):
        """Verifies forward pass through contrastive multimodal model produces expected output keys."""
        ckpt = PHASE6_MODELS / "multimodal_contrastive.pt"
        if not ckpt.exists():
            pytest.skip("multimodal_contrastive.pt not found")

        model = MultimodalVerificationModel(fusion_type="cross_attention", projection_dim=256)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16)
        images = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, images=images)

        assert "logits" in out
        assert "text_features" in out
        assert "vision_features" in out
        assert "cross_attention" in out
        assert out["logits"].shape == (2, 2)
        assert out["text_features"].shape == (2, 256)
        assert out["vision_features"].shape == (2, 256)
